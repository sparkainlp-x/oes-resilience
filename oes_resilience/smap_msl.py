# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""NASA SMAP/MSL adapter and preregistered evaluation on real public telemetry (v0.5).

Data: the SMAP/MSL telemetry anomaly dataset of Hundman et al. (2018), "Detecting
Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD '18,
https://doi.org/10.1145/3219819.3219845 (code and labels: https://github.com/khundman/telemanom).
The raw data is **never bundled**: ``fetch`` downloads it into a cache outside the
repository and verifies every extracted file against a committed SHA-256 manifest.

Protocol (locked in ``reports/smap_msl_protocol.json`` before any test-split scoring):

* Univariate per channel: only column 0 of each ``.npy`` file (the telemetry value) is
  used; the one-hot command columns are ignored by every method.
* Every method sees the same input: ``z = (x - mean_train) / std_train`` (ddof 1, floor
  ``MIN_SCALE``), with statistics from the channel's anomaly-free train split only.
* Methods: ``oes32`` (the reference score over a trailing window of 32 samples, i.e. the
  32-channel block becomes 32 consecutive samples), ``maxabs`` (max|z| over the same
  window), ``zscore`` (|z_t|), ``ewma`` and ``cusum`` (the built-in recursions on ``z``).
* Thresholds: :func:`oes_resilience.scorecard.calibrate_threshold` on the train-split
  scores (unit = one sample) at a target false-alarm rate; no test label is used.
* Metrics: event-level precision/recall/F1 (an event is detected if any alarm falls inside
  its labelled window; a false alarm is a maximal run of alarms that touches no labelled
  window), point-adjusted F1 (secondary; known to be inflated), latency, false alarms per
  1000 test samples; channel bootstrap CIs, paired bootstrap and Wilcoxon tests.

This module only reads local files and public URLs. It is an offline evaluation, not an
operational alarm or a claim about any spacecraft system.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import core
from .core import Config, atomic_write, csv_bytes, json_bytes
from .detectors import MIN_SCALE, MaxAbsDetector, OES32Detector
from .scorecard import calibrate_threshold

DATASETS = ("SMAP", "MSL")
METHODS = ("oes32", "maxabs", "zscore", "ewma", "cusum")
CANDIDATE = "oes32"
BASELINES = tuple(m for m in METHODS if m != CANDIDATE)
PROTOCOL_SCHEMA = "oes-resilience/smap-msl-protocol/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "reports" / "smap_msl_protocol.json"
DEFAULT_MANIFEST = REPO_ROOT / "reports" / "smap_msl_data.sha256"
#: Public sources, tried in order. The telemanom README now points to the Kaggle mirror;
#: the original S3 bucket is kept as a fallback (it returned HTTP 403 on 2026-10-03).
DEFAULT_URLS = (
    "https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
    "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
)
_MEMBER = re.compile(r"(?:^|/)(train|test)/([A-Za-z]-\d+)\.npy$")
_LABELS = re.compile(r"(?:^|/)labeled_anomalies\.csv$")
_CHANNEL = re.compile(r"^[A-Za-z]-\d+$")


class DataError(ValueError):
    """Missing, malformed or unverified SMAP/MSL data."""


# --------------------------------------------------------------------------
# Fetch and verify
# --------------------------------------------------------------------------


def read_manifest(path: str | Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest into ``{relative_path: sha256}``."""
    entries: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise DataError(f"{path}:{number}: expected '<sha256>  <path>'")
        name = parts[1].lstrip("*")
        if name.startswith("/") or ".." in Path(name).parts:
            raise DataError(f"{path}:{number}: unsafe path {name!r}")
        entries[name] = parts[0]
    if not entries:
        raise DataError(f"{path}: manifest is empty")
    return entries


def manifest_text(data_dir: str | Path) -> str:
    """``sha256sum``-style manifest of every train/test ``.npy`` and the label CSV under ``data_dir``."""
    root = Path(data_dir)
    names = sorted(str(p.relative_to(root)) for p in root.glob("t*/*.npy") if p.parent.name in ("train", "test"))
    names.append("labeled_anomalies.csv")
    return "".join(f"{core.sha256_file(root / name)}  {name}\n" for name in names)


def verify_data(data_dir: str | Path, manifest: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Check every file named in the manifest; raise :class:`DataError` on any mismatch."""
    root = Path(data_dir)
    expected = read_manifest(manifest)
    missing = [name for name in expected if not (root / name).is_file()]
    bad = [name for name, digest in expected.items() if name not in missing and core.sha256_file(root / name) != digest]
    if missing or bad:
        raise DataError(f"data verification failed: {len(missing)} missing, {len(bad)} with a different SHA-256 "
                        f"(first: {(missing + bad)[:3]})")
    return {"files": len(expected), "manifest_sha256": core.sha256_file(Path(manifest))}


def extract_archive(archive: str | Path, data_dir: str | Path) -> int:
    """Extract only ``train/*.npy``, ``test/*.npy`` and ``labeled_anomalies.csv`` (flattened, zip-slip safe)."""
    root = Path(data_dir)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            match = _MEMBER.search(info.filename)
            if match:
                target = root / match.group(1) / f"{match.group(2)}.npy"
            elif _LABELS.search(info.filename):
                target = root / "labeled_anomalies.csv"
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            count += 1
    if count == 0:
        raise DataError(f"{archive}: no SMAP/MSL train/test files found in the archive")
    return count


def download(url: str, target: Path, timeout: float = 120.0) -> str:
    """Stream ``url`` to ``target`` (via a ``.part`` file); return the SHA-256 of the bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": f"oes-resilience/{core.__version__}"})
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as sink:  # noqa: S310
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            sink.write(chunk)
    partial.replace(target)
    return digest.hexdigest()


def fetch(
    data_dir: str | Path,
    manifest: str | Path = DEFAULT_MANIFEST,
    urls: Sequence[str] = DEFAULT_URLS,
    archive: str | Path | None = None,
    keep_archive: bool = True,
) -> dict[str, Any]:
    """Download (or take a local ``archive``), extract and verify the dataset into ``data_dir``.

    Verification is per extracted file against ``manifest`` (the archive itself is
    re-packed by the mirror, so its hash is recorded but not required to match).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    source = str(archive) if archive is not None else None
    if archive is None:
        target = root / "smap_msl_download.zip"
        for url in urls:
            try:
                archive_sha = download(url, target)
                source = url
                break
            except OSError as exc:  # urllib.error.URLError and HTTPError are OSError subclasses
                errors.append(f"{url}: {exc}")
        else:
            raise DataError("all download sources failed: " + "; ".join(errors))
        archive = target
    else:
        archive_sha = core.sha256_file(Path(archive))
    extracted = extract_archive(archive, root)
    if not keep_archive and archive == root / "smap_msl_download.zip":
        Path(archive).unlink()
    result = verify_data(root, manifest)
    return {**result, "source": source, "archive_sha256": archive_sha, "extracted": extracted, "errors": errors}


# --------------------------------------------------------------------------
# Labels and channels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One telemetry channel: its spacecraft, ID, merged labelled windows (inclusive) and classes."""

    spacecraft: str
    chan_id: str
    events: tuple[tuple[int, int], ...]
    classes: tuple[str, ...]
    num_values: int
    label_rows: int


def merge_windows(windows: Sequence[tuple[int, int, str]]) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    """Merge overlapping or adjacent inclusive windows; the merged class is ``+``-joined if they differ."""
    merged: list[list[Any]] = []
    for start, end, klass in sorted(windows):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].add(klass)
        else:
            merged.append([start, end, {klass}])
    return tuple((s, e) for s, e, _ in merged), tuple("+".join(sorted(k)) for _, _, k in merged)


def _parse_class_list(text: str, count: int, where: str) -> list[str]:
    items = [item.strip() for item in text.strip().strip("[]").split(",")]
    if len(items) != count or not all(re.fullmatch(r"[a-z]+", item) for item in items):
        raise DataError(f"{where}: class list {text!r} does not match {count} windows")
    return items


def load_labels(path: str | Path) -> list[Channel]:
    """Read ``labeled_anomalies.csv``; rows sharing (spacecraft, chan_id) are pooled and merged."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), 2):
            where = f"{path}:{number}"
            try:
                spacecraft, chan_id = row["spacecraft"].strip(), row["chan_id"].strip()
                windows = ast.literal_eval(row["anomaly_sequences"])
                num_values = int(row["num_values"])
                klass = row["class"]
            except (KeyError, ValueError, SyntaxError, AttributeError) as exc:
                raise DataError(f"{where}: malformed row ({exc})") from None
            if spacecraft not in DATASETS or not _CHANNEL.match(chan_id):
                raise DataError(f"{where}: unexpected spacecraft {spacecraft!r} or channel {chan_id!r}")
            if not isinstance(windows, list) or not windows or not all(
                isinstance(w, list) and len(w) == 2 and all(isinstance(v, int) for v in w) and 0 <= w[0] <= w[1]
                < num_values for w in windows
            ):
                raise DataError(f"{where}: invalid anomaly_sequences {row['anomaly_sequences']!r}")
            classes = _parse_class_list(klass, len(windows), where)
            entry = grouped.setdefault((spacecraft, chan_id), {"windows": [], "num_values": num_values, "rows": 0})
            if entry["num_values"] != num_values:
                raise DataError(f"{where}: num_values differs between rows of {chan_id}")
            entry["windows"].extend((s, e, c) for (s, e), c in zip(windows, classes, strict=True))
            entry["rows"] += 1
    channels = []
    for (spacecraft, chan_id), entry in grouped.items():
        events, classes = merge_windows(entry["windows"])
        channels.append(Channel(spacecraft, chan_id, events, classes, entry["num_values"], entry["rows"]))
    return channels


def load_series(data_dir: str | Path, channel: Channel) -> tuple[np.ndarray, np.ndarray]:
    """Column 0 (the telemetry value) of the channel's train and test arrays, as float64."""
    root = Path(data_dir)
    series = []
    for split in ("train", "test"):
        array = np.load(root / split / f"{channel.chan_id}.npy", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] < 2 or not np.all(np.isfinite(array[:, 0])):
            raise DataError(f"{split}/{channel.chan_id}.npy: expected a finite 2-D array, got {array.shape}")
        series.append(np.asarray(array[:, 0], dtype=np.float64))
    if series[1].size != channel.num_values:
        raise DataError(f"test/{channel.chan_id}.npy has {series[1].size} values, labels say {channel.num_values}")
    return series[0], series[1]


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


def standardize(train: np.ndarray, *series: np.ndarray) -> tuple[list[np.ndarray], float, float]:
    """Standardise with train mean and train std (ddof 1, floored at ``MIN_SCALE``)."""
    mu = float(np.mean(train))
    sigma = max(float(np.std(train, ddof=1)), MIN_SCALE)
    return [(np.asarray(s, dtype=np.float64) - mu) / sigma for s in (train, *series)], mu, sigma


def trailing_windows(z: np.ndarray, window: int) -> np.ndarray:
    """``(n,) -> (n, window)``: row t holds ``z[t-window+1 .. t]``, NaN-padded before the first sample."""
    padded = np.concatenate([np.full(window - 1, np.nan), np.asarray(z, dtype=np.float64)])
    return np.lib.stride_tricks.sliding_window_view(padded, window).copy()


def _window_scores(detector_class: type, z: np.ndarray, window: int) -> np.ndarray:
    detector = detector_class(Config(channels=window, block_size=window))
    frames = trailing_windows(z, window)
    head = min(window - 1, z.size)  # partial windows: mask-aware scoring over observed samples only
    scores = np.empty(z.size)
    scores[:head] = detector.score(frames[:head])[:, 0]
    if z.size > head:
        scores[head:] = detector.score(frames[head:])[:, 0]  # full windows: exact reference path
    return scores


def ewma_scores(z: np.ndarray, lam: float) -> np.ndarray:
    """Same recursion as :class:`~oes_resilience.detectors.EWMADetector`, state 0 at the first sample."""
    out = np.empty(z.size)
    norm = math.sqrt(lam / (2.0 - lam))
    state = 0.0
    for t, value in enumerate(z):
        state = lam * value + (1.0 - lam) * state
        out[t] = abs(state) / norm
    return out


def cusum_scores(z: np.ndarray, k: float) -> np.ndarray:
    """Same recursion as :class:`~oes_resilience.detectors.CUSUMDetector`, sums 0 at the first sample."""
    out = np.empty(z.size)
    upper = lower = 0.0
    for t, value in enumerate(z):
        upper = max(0.0, upper + value - k)
        lower = max(0.0, lower - value - k)
        out[t] = max(upper, lower)
    return out


def method_scores(method: str, z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Per-sample scores of ``method`` on one standardised segment (recursions restart per segment)."""
    if method == "oes32":
        return _window_scores(OES32Detector, z, int(params["window"]))
    if method == "maxabs":
        return _window_scores(MaxAbsDetector, z, int(params["window"]))
    if method == "zscore":
        return np.abs(z)
    if method == "ewma":
        return ewma_scores(z, float(params["lam"]))
    if method == "cusum":
        return cusum_scores(z, float(params["k"]))
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def alarm_runs(alarms: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive alarms as inclusive ``(start, end)`` pairs."""
    padded = np.concatenate([[0], np.asarray(alarms, dtype=np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def label_mask(n: int, events: Sequence[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in events:
        mask[start : end + 1] = True
    return mask


def channel_metrics(alarms: np.ndarray, events: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Event-level and point-adjusted counts for one channel's test split.

    An event is detected if any alarm falls in ``[start, end]``; latency = first such
    alarm index minus ``start``. A false alarm is a maximal run of alarms that touches no
    labelled window. Point adjustment marks every point of a detected event as predicted.
    """
    alarms = np.asarray(alarms, dtype=bool)
    n = alarms.size
    truth = label_mask(n, events)
    detected, latencies = [], []
    for start, end in events:
        hits = np.flatnonzero(alarms[start : end + 1])
        detected.append(bool(hits.size))
        if hits.size:
            latencies.append(int(hits[0]))
    fp_runs = sum(1 for s, e in alarm_runs(alarms) if not truth[s : e + 1].any())
    adjusted = alarms.copy()
    for (start, end), hit in zip(events, detected, strict=True):
        if hit:
            adjusted[start : end + 1] = True
    tp_events = int(sum(detected))
    return {
        "n_test": n,
        "n_events": len(events),
        "tp_events": tp_events,
        "fp_runs": fp_runs,
        "detected": detected,
        "latencies": latencies,
        "pa_tp": int((adjusted & truth).sum()),
        "pa_fp": int((adjusted & ~truth).sum()),
        "pa_fn": int((~adjusted & truth).sum()),
        "normal_samples": int((~truth).sum()),
        "normal_alarm_samples": int((alarms & ~truth).sum()),
        "alarm_samples": int(alarms.sum()),
    }


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1; each is 0 when its denominator is 0 (F1 is 0 when TP is 0)."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def event_f1_from_counts(tp: np.ndarray, n_events: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Vectorised event F1 = 2TP / (2TP + FP + FN) with FN = events - TP (0 when TP = 0)."""
    tp, n_events, fp = (np.asarray(a, dtype=np.float64) for a in (tp, n_events, fp))
    denominator = tp + fp + n_events
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=(tp > 0) & (denominator > 0))


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test; zero differences are dropped (Wilcoxon's method).

    Exact null distribution when there are 1..50 non-zero differences and no tied
    magnitudes; otherwise the normal approximation with tie and continuity corrections.
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = int(d.size)
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "p_value": 1.0, "method": "no non-zero differences"}
    magnitude = np.abs(d)
    order = np.argsort(magnitude, kind="mergesort")
    ranks = np.empty(n)
    sorted_mag = magnitude[order]
    i = 0
    ties = []
    while i < n:
        j = i
        while j + 1 < n and math.isclose(sorted_mag[j + 1], sorted_mag[i], rel_tol=1e-9, abs_tol=1e-12):
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    if not ties and n <= 50:
        total = n * (n + 1) // 2
        counts = np.zeros(total + 1)
        counts[0] = 1.0
        for r in range(1, n + 1):
            counts[r:] = counts[r:] + counts[: total + 1 - r].copy()
        cdf = np.cumsum(counts) / counts.sum()
        w = int(round(w_plus))
        lower = cdf[w]
        upper = 1.0 - (cdf[w - 1] if w > 0 else 0.0)
        return {"n_nonzero": n, "w_plus": w_plus, "p_value": float(min(1.0, 2 * min(lower, upper))),
                "method": "exact"}
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t**3 - t for t in ties) / 48.0
    if variance <= 0:  # pragma: no cover - only if every magnitude is tied and n == 1
        return {"n_nonzero": n, "w_plus": w_plus, "p_value": 1.0, "method": "degenerate"}
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return {"n_nonzero": n, "w_plus": w_plus, "p_value": float(min(1.0, p)), "method": "normal approximation"}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    """Load the locked protocol JSON and return it with the SHA-256 of its exact bytes."""
    raw = Path(path).read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise DataError(f"{path}: not a {PROTOCOL_SCHEMA} protocol")
    params = protocol.get("methods")
    if not isinstance(params, dict) or tuple(params) != METHODS or protocol.get("candidate") != CANDIDATE:
        raise DataError(f"{path}: methods must be {list(METHODS)} with candidate {CANDIDATE!r}")
    calibration = protocol.get("calibration", {})
    alphas = [calibration.get("primary_target_fp"), *calibration.get("secondary_target_fp", [])]
    if not all(isinstance(a, (int, float)) and 0 <= a < 1 for a in alphas) or alphas[0] is None:
        raise DataError(f"{path}: calibration targets must be in [0, 1)")
    boot = protocol.get("uncertainty", {})
    if not isinstance(boot.get("bootstrap_resamples"), int) or not isinstance(boot.get("seed"), int):
        raise DataError(f"{path}: uncertainty.bootstrap_resamples and uncertainty.seed must be integers")
    return protocol, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _r(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def score_channel(train: np.ndarray, test: np.ndarray, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Train and test scores of every method on identically standardised input."""
    (z_train, z_test), mu, sigma = standardize(train, test)
    scores = {
        m: (method_scores(m, z_train, protocol["methods"][m]), method_scores(m, z_test, protocol["methods"][m]))
        for m in METHODS
    }
    return {"scores": scores, "mu": mu, "sigma": sigma, "constant_train": bool(np.std(train) == 0.0)}


def _alphas(protocol: Mapping[str, Any]) -> list[float]:
    calibration = protocol["calibration"]
    return [float(calibration["primary_target_fp"]), *map(float, calibration.get("secondary_target_fp", []))]


def evaluate_channels(
    data_dir: str | Path, channels: Sequence[Channel], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Per-channel, per-target, per-method records (thresholds from train only)."""
    records = []
    for channel in sorted(channels, key=lambda c: (DATASETS.index(c.spacecraft), c.chan_id)):
        train, test = load_series(data_dir, channel)
        scored = score_channel(train, test, protocol)
        for alpha in _alphas(protocol):
            for method in METHODS:
                train_scores, test_scores = scored["scores"][method]
                threshold = calibrate_threshold({"train": train_scores}, alpha)["threshold"]
                alarms = test_scores >= threshold
                metrics = channel_metrics(alarms, channel.events)
                records.append({
                    "dataset": channel.spacecraft, "channel": channel.chan_id, "target_fp": alpha, "method": method,
                    "threshold": threshold, "n_train": int(train.size), "train_alarm_rate": float(
                        np.mean(train_scores >= threshold)), "constant_train": scored["constant_train"],
                    "classes": list(channel.classes), **metrics,
                })
    return records


def _pooled(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(r["tp_events"] for r in rows)
    events = sum(r["n_events"] for r in rows)
    fp = sum(r["fp_runs"] for r in rows)
    n_test = sum(r["n_test"] for r in rows)
    precision, recall, f1 = prf(tp, fp, events - tp)
    pa_p, pa_r, pa_f1 = prf(sum(r["pa_tp"] for r in rows), sum(r["pa_fp"] for r in rows),
                            sum(r["pa_fn"] for r in rows))
    latencies = [lat for r in rows for lat in r["latencies"]]
    by_class: dict[str, list[int]] = {}
    for r in rows:
        for klass, hit in zip(r["classes"], r["detected"], strict=True):
            by_class.setdefault(klass, []).append(int(hit))
    per_channel_f1 = [prf(r["tp_events"], r["fp_runs"], r["n_events"] - r["tp_events"])[2] for r in rows]
    return {
        "channels": len(rows), "events": events, "tp_events": tp, "fp_runs": fp, "n_test": n_test,
        "event_precision": _r(precision), "event_recall": _r(recall), "event_f1": _r(f1),
        "pa_precision": _r(pa_p), "pa_recall": _r(pa_r), "pa_f1": _r(pa_f1),
        "latency_mean": _r(np.mean(latencies)) if latencies else None,
        "latency_median": _r(np.median(latencies)) if latencies else None,
        "false_alarms_per_1k": _r(1000.0 * fp / n_test) if n_test else None,
        "test_normal_alarm_rate": _r(sum(r["normal_alarm_samples"] for r in rows)
                                     / max(sum(r["normal_samples"] for r in rows), 1)),
        "train_alarm_rate_mean": _r(np.mean([r["train_alarm_rate"] for r in rows])),
        "mean_channel_event_f1": _r(np.mean(per_channel_f1)),
        "recall_by_class": {k: _r(np.mean(v)) for k, v in sorted(by_class.items())},
    }


def _bootstrap_indices(n_channels: int, resamples: int, seed: int, dataset_index: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(dataset_index,)))
    return rng.integers(0, n_channels, size=(resamples, n_channels))


def analyse(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Pooled metrics, channel-bootstrap CIs, paired comparisons and verdicts per dataset and target."""
    unc = protocol["uncertainty"]
    resamples, seed, level = int(unc["bootstrap_resamples"]), int(unc["seed"]), float(unc.get("ci_level", 0.95))
    tail = (1.0 - level) / 2.0
    alpha_test = float(protocol["success_criteria"]["significance_level"])
    subsets = {"all_channels": lambda r: True, "excluding_constant_train": lambda r: not r["constant_train"]}
    summary, comparisons = [], []
    for subset, keep in subsets.items():
        for alpha in _alphas(protocol):
            block = []
            for d_index, dataset in enumerate(DATASETS):
                rows = [r for r in records if r["dataset"] == dataset and r["target_fp"] == alpha and keep(r)]
                channel_ids = sorted({r["channel"] for r in rows})
                if not channel_ids:
                    continue
                idx = _bootstrap_indices(len(channel_ids), resamples, seed, d_index)
                counts, per_channel_f1 = {}, {}
                for method in METHODS:
                    by_channel = {r["channel"]: r for r in rows if r["method"] == method}
                    ordered = [by_channel[c] for c in channel_ids]
                    tp = np.array([r["tp_events"] for r in ordered])
                    ev = np.array([r["n_events"] for r in ordered])
                    fp = np.array([r["fp_runs"] for r in ordered])
                    counts[method] = event_f1_from_counts(tp[idx].sum(1), ev[idx].sum(1), fp[idx].sum(1))
                    per_channel_f1[method] = event_f1_from_counts(tp, ev, fp)
                    pooled = _pooled(ordered)
                    low, high = np.quantile(counts[method], [tail, 1.0 - tail])
                    summary.append({"subset": subset, "target_fp": alpha, "dataset": dataset, "method": method,
                                    **pooled, "event_f1_ci_low": _r(low), "event_f1_ci_high": _r(high)})
                pooled_f1 = {m: next(s["event_f1"] for s in summary if s["subset"] == subset and
                                     s["target_fp"] == alpha and s["dataset"] == dataset and s["method"] == m)
                             for m in METHODS}
                for baseline in BASELINES:
                    diff = counts[CANDIDATE] - counts[baseline]
                    low, high = np.quantile(diff, [tail, 1.0 - tail])
                    p_boot = min(1.0, 2.0 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0))))
                    test = wilcoxon_signed_rank(per_channel_f1[CANDIDATE] - per_channel_f1[baseline])
                    block.append({
                        "subset": subset, "target_fp": alpha, "dataset": dataset, "candidate": CANDIDATE,
                        "baseline": baseline, "channels": len(channel_ids),
                        "candidate_event_f1": pooled_f1[CANDIDATE], "baseline_event_f1": pooled_f1[baseline],
                        "delta_event_f1": _r(pooled_f1[CANDIDATE] - pooled_f1[baseline]),
                        "delta_ci_low": _r(low), "delta_ci_high": _r(high), "paired_bootstrap_p": _r(p_boot),
                        "mean_channel_f1_delta": _r(np.mean(per_channel_f1[CANDIDATE] - per_channel_f1[baseline])),
                        "wilcoxon_n_nonzero": test["n_nonzero"], "wilcoxon_w_plus": _r(test["w_plus"]),
                        "wilcoxon_p": _r(test["p_value"]), "wilcoxon_method": test["method"],
                    })
            for row, adjusted in zip(block, holm([row["wilcoxon_p"] for row in block]), strict=True):
                row["wilcoxon_p_holm"] = _r(adjusted)
                significant = adjusted < alpha_test
                if significant and row["delta_ci_low"] > 0:
                    row["verdict"] = "oes32 better"
                elif significant and row["delta_ci_high"] < 0:
                    row["verdict"] = "oes32 worse"
                else:
                    row["verdict"] = "no significant difference"
            comparisons.extend(block)
    return {"summary": summary, "comparisons": comparisons}


def headline(analysis: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-stated success criterion to the primary analysis."""
    primary = float(protocol["calibration"]["primary_target_fp"])
    rows = [c for c in analysis["comparisons"] if c["subset"] == "all_channels" and c["target_fp"] == primary]
    per_dataset = {d: [r["verdict"] for r in rows if r["dataset"] == d] for d in DATASETS}
    wins_all = [d for d, v in per_dataset.items() if v and all(x == "oes32 better" for x in v)]
    losses = sorted({f"{r['dataset']}:{r['baseline']}" for r in rows if r["verdict"] == "oes32 worse"})
    met = bool(wins_all) and not losses
    return {"primary_target_fp": primary, "criterion_met": met, "datasets_where_oes32_beats_all_baselines": wins_all,
            "significant_losses": losses, "criterion": protocol["success_criteria"]["headline"]}


def per_channel_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        p, rec, f1 = prf(r["tp_events"], r["fp_runs"], r["n_events"] - r["tp_events"])
        rows.append({
            "dataset": r["dataset"], "channel": r["channel"], "target_fp": r["target_fp"], "method": r["method"],
            "threshold": f"{r['threshold']:.6g}", "constant_train": int(r["constant_train"]), "n_train": r["n_train"],
            "n_test": r["n_test"], "n_events": r["n_events"], "tp_events": r["tp_events"], "fp_runs": r["fp_runs"],
            "event_precision": _r(p), "event_recall": _r(rec), "event_f1": _r(f1),
            "latency_first": r["latencies"][0] if r["latencies"] else None,
            "false_alarms_per_1k": _r(1000.0 * r["fp_runs"] / r["n_test"]),
            "train_alarm_rate": _r(r["train_alarm_rate"]),
            "test_normal_alarm_rate": _r(r["normal_alarm_samples"] / max(r["normal_samples"], 1)),
        })
    return rows


PER_CHANNEL_FIELDS = (
    "dataset", "channel", "target_fp", "method", "threshold", "constant_train", "n_train", "n_test", "n_events",
    "tp_events", "fp_runs", "event_precision", "event_recall", "event_f1", "latency_first", "false_alarms_per_1k",
    "train_alarm_rate", "test_normal_alarm_rate",
)
SUMMARY_FIELDS = (
    "subset", "target_fp", "dataset", "method", "channels", "events", "tp_events", "fp_runs", "n_test",
    "event_precision", "event_recall", "event_f1", "event_f1_ci_low", "event_f1_ci_high", "pa_precision", "pa_recall",
    "pa_f1", "latency_mean", "latency_median", "false_alarms_per_1k", "test_normal_alarm_rate",
    "train_alarm_rate_mean", "mean_channel_event_f1", "recall_point", "recall_contextual",
)
COMPARISON_FIELDS = (
    "subset", "target_fp", "dataset", "candidate", "baseline", "channels", "candidate_event_f1", "baseline_event_f1",
    "delta_event_f1", "delta_ci_low", "delta_ci_high", "paired_bootstrap_p", "mean_channel_f1_delta",
    "wilcoxon_n_nonzero", "wilcoxon_w_plus", "wilcoxon_p", "wilcoxon_p_holm", "wilcoxon_method", "verdict",
)


def summary_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in analysis["summary"]:
        row = {k: s.get(k) for k in SUMMARY_FIELDS if k in s}
        row["recall_point"] = s["recall_by_class"].get("point")
        row["recall_contextual"] = s["recall_by_class"].get("contextual")
        rows.append(row)
    return rows


def build_results(
    data_dir: str | Path, protocol: Mapping[str, Any], protocol_sha256: str, labels: str | Path | None = None,
    manifest_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the locked evaluation and return the JSON-ready results (deterministic)."""
    labels = Path(labels) if labels is not None else Path(data_dir) / "labeled_anomalies.csv"
    channels = load_labels(labels)
    records = evaluate_channels(data_dir, channels, protocol)
    analysis = analyse(records, protocol)
    return {
        "project": core.PROJECT, "version": core.__version__, "evaluation": "nasa-smap-msl",
        "data_kind": "REAL public telemetry (NASA SMAP/MSL, Hundman et al. 2018)",
        "protocol_id": protocol.get("protocol_id"), "protocol_sha256": protocol_sha256,
        "data_manifest": dict(manifest_info or {}), "labels_sha256": core.sha256_file(labels),
        "channels": {d: sorted(c.chan_id for c in channels if c.spacecraft == d) for d in DATASETS},
        "events": {d: sum(len(c.events) for c in channels if c.spacecraft == d) for d in DATASETS},
        "merged_label_rows": sorted(f"{c.spacecraft}:{c.chan_id}" for c in channels if c.label_rows > 1),
        "constant_train_channels": sorted({f"{r['dataset']}:{r['channel']}" for r in records if r["constant_train"]}),
        "headline": headline(analysis, protocol), **analysis, "per_channel": per_channel_rows(records),
    }


# --------------------------------------------------------------------------
# Plot (dependency-free, deterministic SVG)
# --------------------------------------------------------------------------


def svg_plot(results: Mapping[str, Any], target_fp: float, subset: str = "all_channels") -> str:
    """Grouped bar chart of pooled event-level F1 with bootstrap 95% CIs, one panel per dataset."""
    width, height, left, top, panel_w, plot_h = 760, 360, 60, 50, 330, 230
    colors = {"oes32": "#d62728", "maxabs": "#7f7f7f", "zscore": "#1f77b4", "ewma": "#2ca02c", "cusum": "#9467bd"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2:.0f}" y="20" text-anchor="middle" font-size="13">NASA SMAP/MSL (real public data): '
           f'pooled event-level F1, 95% channel-bootstrap CI, train-calibrated target FP {target_fp:g}</text>']
    for p, dataset in enumerate(DATASETS):
        x0 = left + p * (panel_w + 40)
        rows = {s["method"]: s for s in results["summary"]
                if s["subset"] == subset and s["target_fp"] == target_fp and s["dataset"] == dataset}
        out.append(f'<text x="{x0 + panel_w / 2:.0f}" y="{top - 8}" text-anchor="middle" font-size="12">'
                   f'{dataset} ({rows[CANDIDATE]["channels"] if rows else 0} channels)</text>')
        out.append(f'<line x1="{x0}" y1="{top + plot_h}" x2="{x0 + panel_w}" y2="{top + plot_h}" stroke="black"/>')
        out.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + plot_h}" stroke="black"/>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + plot_h * (1 - tick)
            out.append(f'<text x="{x0 - 5}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
            out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#ddd"/>')
        bar = panel_w / (len(METHODS) * 1.5)
        for i, method in enumerate(METHODS):
            if method not in rows:
                continue
            s = rows[method]
            bx = x0 + bar * 0.5 + i * bar * 1.5
            bh = plot_h * s["event_f1"]
            out.append(f'<rect x="{bx:.1f}" y="{top + plot_h - bh:.1f}" width="{bar:.1f}" height="{bh:.1f}" '
                       f'fill="{colors[method]}"/>')
            cx = bx + bar / 2
            y_low, y_high = top + plot_h * (1 - s["event_f1_ci_low"]), top + plot_h * (1 - s["event_f1_ci_high"])
            out.append(f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" stroke="black"/>')
            out.append(f'<text x="{cx:.1f}" y="{top + plot_h + 14}" text-anchor="middle">{method}</text>')
            out.append(f'<text x="{cx:.1f}" y="{min(y_high, top + plot_h - bh) - 4:.1f}" text-anchor="middle">'
                       f'{s["event_f1"]:.2f}</text>')
    out.append(f'<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" fill="#555">Event detected if any '
               'alarm falls inside its labelled window; false alarm = alarm run touching no window.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Outputs and CLI
# --------------------------------------------------------------------------


def write_results(results: Mapping[str, Any], out_dir: str | Path, prefix: str = "smap_msl") -> dict[str, Path]:
    """Write results JSON, summary/comparison/per-channel CSVs, the SVG plot and SHA256SUMS."""
    out = Path(out_dir)
    primary = float(results["headline"]["primary_target_fp"])
    payloads = {
        f"{prefix}_results.json": json_bytes(results),
        f"{prefix}_summary.csv": csv_bytes(summary_rows(results), SUMMARY_FIELDS),
        f"{prefix}_comparisons.csv": csv_bytes(results["comparisons"], COMPARISON_FIELDS),
        f"{prefix}_per_channel.csv": csv_bytes(results["per_channel"], PER_CHANNEL_FIELDS),
        f"{prefix}_event_f1.svg": svg_plot(results, primary).encode("utf-8"),
    }
    paths, sums = {}, []
    for name, payload in payloads.items():
        sums.append(f"{atomic_write(out / name, payload)}  {name}\n")
        paths[name] = out / name
    atomic_write(out / "SHA256SUMS", "".join(sorted(sums, key=lambda line: line.split()[1])).encode("utf-8"))
    paths["SHA256SUMS"] = out / "SHA256SUMS"
    return paths


def run_evaluation(data_dir: str | Path, protocol_path: str | Path, manifest: str | Path | None) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    info = verify_data(data_dir, manifest) if manifest is not None else {"verified": False}
    return build_results(data_dir, protocol, protocol_sha, Path(data_dir) / "labeled_anomalies.csv", info)


def train_diagnostics(data_dir: str | Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Development aid that touches the TRAIN split only: thresholds and train alarm rates."""
    rows = []
    for channel in load_labels(Path(data_dir) / "labeled_anomalies.csv"):
        train = np.asarray(np.load(Path(data_dir) / "train" / f"{channel.chan_id}.npy")[:, 0], dtype=np.float64)
        (z_train,), _, _ = standardize(train)
        for method in METHODS:
            scores = method_scores(method, z_train, protocol["methods"][method])
            for alpha in _alphas(protocol):
                threshold = calibrate_threshold({"train": scores}, alpha)["threshold"]
                rows.append({"dataset": channel.spacecraft, "channel": channel.chan_id, "method": method,
                             "target_fp": alpha, "threshold": threshold,
                             "train_alarm_rate": float(np.mean(scores >= threshold))})
    return rows


def add_smap_parser(sub: Any) -> argparse.ArgumentParser:
    parser = sub.add_parser("smap-msl", help="NASA SMAP/MSL real-data adapter: fetch, verify, evaluate (v0.5)")
    ops = parser.add_subparsers(dest="smap_command", required=True)
    fetch_p = ops.add_parser("fetch", help="download, extract and SHA-256-verify the public dataset into a cache")
    fetch_p.add_argument("--data-dir", type=Path, required=True, help="cache directory (keep it outside the repo)")
    fetch_p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fetch_p.add_argument("--archive", type=Path, help="use an already downloaded zip instead of downloading")
    fetch_p.add_argument("--url", action="append", help="override the download URL(s)")
    fetch_p.add_argument("--no-keep-archive", action="store_true", help="delete the downloaded zip after extraction")
    verify_p = ops.add_parser("verify", help="re-check the cached files against the SHA-256 manifest")
    verify_p.add_argument("--data-dir", type=Path, required=True)
    verify_p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    dev = ops.add_parser("train-diagnostics", help="development aid on the TRAIN split only (no test data)")
    dev.add_argument("--data-dir", type=Path, required=True)
    dev.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ev = ops.add_parser("evaluate", help="run the locked protocol on the test split and write the results")
    ev.add_argument("--data-dir", type=Path, required=True)
    ev.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ev.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ev.add_argument("--skip-verify", action="store_true", help="do not verify data hashes (tests and fixtures only)")
    ev.add_argument("--expect-protocol-sha256", help="refuse to run unless the protocol file has this SHA-256")
    ev.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ev.add_argument("--prefix", default="smap_msl")
    ev.add_argument("--verify", action="store_true", help="recompute and exit 4 unless the results hash matches")
    ev.add_argument("--quiet", action="store_true")
    return parser


def cmd_smap(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Handler for ``smap-msl``."""
    try:
        if args.smap_command == "fetch":
            info = fetch(args.data_dir, args.manifest, tuple(args.url) if args.url else DEFAULT_URLS, args.archive,
                         keep_archive=not args.no_keep_archive)
            print(json.dumps(info, indent=2, sort_keys=True))
            return core.EXIT_OK
        if args.smap_command == "verify":
            print(json.dumps(verify_data(args.data_dir, args.manifest), indent=2, sort_keys=True))
            return core.EXIT_OK
        if args.smap_command == "train-diagnostics":
            protocol, _ = load_protocol(args.protocol)
            rows = train_diagnostics(args.data_dir, protocol)
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            return core.EXIT_OK
        _, sha = load_protocol(args.protocol)
        if args.expect_protocol_sha256 and sha != args.expect_protocol_sha256:
            raise DataError(f"protocol SHA-256 {sha} does not match the expected {args.expect_protocol_sha256}")
        manifest = None if args.skip_verify else args.manifest
        started = time.perf_counter()
        results = run_evaluation(args.data_dir, args.protocol, manifest)
        paths = write_results(results, args.out_dir, args.prefix)
        atomic_write(Path(args.out_dir) / f"{args.prefix}_run_metadata.json",
                     json_bytes(core.run_metadata(argv, time.perf_counter() - started)))
        if args.verify:
            again = json_bytes(run_evaluation(args.data_dir, args.protocol, manifest))
            if again != paths[f"{args.prefix}_results.json"].read_bytes():
                print("error: SMAP/MSL results hash mismatch on recomputation", file=sys.stderr)
                return core.EXIT_NOT_REPRODUCIBLE
        if not args.quiet:
            for row in results["comparisons"]:
                if row["subset"] == "all_channels" and row["target_fp"] == results["headline"]["primary_target_fp"]:
                    print(f"{row['dataset']:>4} oes32 {row['candidate_event_f1']:.3f} vs {row['baseline']:<6} "
                          f"{row['baseline_event_f1']:.3f}  delta CI [{row['delta_ci_low']:.3f}, "
                          f"{row['delta_ci_high']:.3f}]  Holm p={row['wilcoxon_p_holm']:.3g}  {row['verdict']}")
            print(f"headline criterion met: {results['headline']['criterion_met']}")
            for path in paths.values():
                print(path)
        return core.EXIT_OK
    except (DataError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return core.EXIT_USAGE


__all__ = [
    "BASELINES", "CANDIDATE", "DATASETS", "METHODS", "Channel", "DataError", "alarm_runs", "analyse",
    "build_results", "channel_metrics", "cusum_scores", "ewma_scores", "fetch", "holm", "load_labels",
    "load_protocol", "load_series", "manifest_text", "merge_windows", "method_scores", "prf", "standardize",
    "svg_plot", "trailing_windows", "verify_data", "wilcoxon_signed_rank", "write_results",
]
