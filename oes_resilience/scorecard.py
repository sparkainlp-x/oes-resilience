# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Detector comparison: equal-footing threshold calibration and a scorecard.

Two tracks:

* **frame**: the unchanged v0.1 single-frame regimes, using the exact v0.1
  evaluation trials. Only frame (non-temporal) detectors apply.
* **stream**: multi-step streams (:mod:`oes_resilience.streams`). Every detector
  applies; frame detectors score each step on its own.

Fairness: every detector is fitted on the same clean data (stable + noisy, from a
seed stream disjoint from evaluation), and its threshold is calibrated the same way
on a third, held-out clean set. For each clean regime, the threshold is the smallest
value that at most ``floor(target_fp * n)`` calibration units reach, where a unit is
one frame (frame track) or one whole stream after warm-up (stream track), scored by
its maximum block score. The calibrated threshold is the maximum over the clean
regimes, so each clean regime is held to the target separately. OES32 is also
reported at its fixed v0.1 reference threshold (0.50), which reproduces v0.1 exactly.

The scorecard JSON/CSV are deterministic (their hashes match across reruns); the CPU
timings are machine-dependent and are written to separate files.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import core
from ._version import __version__
from .core import (
    PROJECT,
    PURPOSE_CALIBRATION,
    PURPOSE_FIT,
    Config,
    Expectation,
    Regime,
    _require_finite,
    _require_int,
    _round,
    atomic_write,
    csv_bytes,
    evaluate_batch,
    generate_batch,
    json_bytes,
    summarize,
    wilson_interval,
)
from .detectors import Detector, TemporalDetector, create_detector, get_detector_class, load_plugins
from .streams import (
    CLEAN_STREAM_REGIMES,
    PURPOSE_STREAM_CALIBRATION,
    PURPOSE_STREAM_EVAL,
    PURPOSE_STREAM_FIT,
    SHIFT_MAGNITUDE,
    STREAM_EXPECTATION,
    StreamConfig,
    StreamRegime,
    generate_streams,
)

DEFAULT_DETECTORS: tuple[str, ...] = ("oes32", "zscore", "ewma", "cusum", "oes32+ewma")
TRACKS: tuple[str, ...] = ("frame", "stream")
CLEAN_FRAME_REGIMES: tuple[Regime, ...] = (Regime.STABLE, Regime.NOISY)
REFERENCE_LABEL = "oes32@0.50"
#: Streams generated and scored per chunk (bounds memory).
STREAM_CHUNK = 32

SCORECARD_FIELDS: tuple[str, ...] = (
    "track",
    "detector",
    "regime",
    "calibration",
    "threshold",
    "n",
    "fp_rate",
    "fp_ci95_high",
    "early_alarm_rate",
    "recall",
    "exact_match_rate",
    "mean_iou",
    "mean_detected_blocks",
    "latency_mean",
    "latency_median",
    "latency_p90",
    "latency_max",
)


@dataclass(frozen=True)
class CompareConfig:
    """Settings for :func:`run_compare`."""

    config: Config = field(default_factory=Config)
    stream: StreamConfig = field(default_factory=StreamConfig)
    trials: int = 1000
    fit_trials: int = 1000
    calibration_trials: int = 1000
    streams: int = 1000
    calibration_streams: int = 1000
    fit_streams: int = 200
    target_fp: float = 0.01
    detectors: tuple[str, ...] = DEFAULT_DETECTORS
    tracks: tuple[str, ...] = TRACKS

    def __post_init__(self) -> None:
        for name in ("trials", "fit_trials", "calibration_trials", "streams", "calibration_streams", "fit_streams"):
            object.__setattr__(self, name, _require_int(name, getattr(self, name), 1))
        target = _require_finite("target_fp", self.target_fp, 0.0)
        if target >= 1.0:
            raise ValueError(f"target_fp must be in [0, 1); received {target!r}.")
        object.__setattr__(self, "target_fp", target)
        detectors = tuple(self.detectors)
        tracks = tuple(self.tracks)
        if not detectors or len(set(detectors)) != len(detectors):
            raise ValueError("detectors must be a non-empty list without duplicates.")
        for name in detectors:
            get_detector_class(name)
        if not tracks or len(set(tracks)) != len(tracks) or not set(tracks) <= set(TRACKS):
            raise ValueError(f"tracks must be a non-empty subset of {TRACKS} without duplicates.")
        object.__setattr__(self, "detectors", detectors)
        object.__setattr__(self, "tracks", tracks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "trials": self.trials,
            "fit_trials_per_clean_regime": self.fit_trials,
            "calibration_trials_per_clean_regime": self.calibration_trials,
            "streams": self.streams,
            "calibration_streams_per_clean_regime": self.calibration_streams,
            "fit_streams_per_clean_regime": self.fit_streams,
            "stream": self.stream.to_dict(),
            "stream_shift_magnitude": SHIFT_MAGNITUDE,
            "target_fp": self.target_fp,
            "detectors": list(self.detectors),
            "tracks": list(self.tracks),
        }


def calibrate_threshold(max_scores: Mapping[str, Any], target_fp: float) -> dict[str, Any]:
    """Equal-footing threshold from per-unit maximum scores on clean calibration data.

    For each regime: the smallest threshold reached by at most ``floor(target_fp * n)``
    units (just above the ``(k+1)``-th largest maximum). Returns the maximum over
    regimes plus per-regime details, including the calibration-set FP rate achieved.
    """
    target_fp = _require_finite("target_fp", target_fp, 0.0)
    if target_fp >= 1.0 or not max_scores:
        raise ValueError("target_fp must be in [0, 1) and max_scores must be non-empty.")
    per_regime: dict[str, dict[str, Any]] = {}
    for regime, values in max_scores.items():
        ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
        if ordered.size == 0:
            raise ValueError(f"No calibration scores for {regime!r}.")
        allowed = min(int(math.floor(target_fp * ordered.size + 1e-9)), ordered.size - 1)
        per_regime[regime] = {
            "n": int(ordered.size),
            "allowed_exceedances": allowed,
            "threshold": float(np.nextafter(ordered[allowed], np.inf)),
        }
    threshold = max(entry["threshold"] for entry in per_regime.values())
    for regime, values in max_scores.items():
        per_regime[regime]["calibration_fp_rate"] = _round(float(np.mean(np.asarray(values) >= threshold)))
    return {"threshold": threshold, "target_fp": target_fp, "per_regime": per_regime}


def stream_outcomes(
    scores: np.ndarray, truth: np.ndarray, onsets: np.ndarray, threshold: float, warmup: int
) -> dict[str, np.ndarray]:
    """Per-stream outcomes from ``(n, steps, blocks)`` scores.

    Alarms before ``warmup`` are ignored for every detector. ``latency`` is the first
    step at/after onset with an alarm on an expected block, minus onset (``-1`` if none).
    ``exact``/``iou`` compare the blocks alarmed at the first alarm step at/after onset
    with the expected blocks.
    """
    alarm = scores >= threshold
    alarm[:, :warmup] = False
    n, steps, _ = alarm.shape
    t = np.arange(steps)
    post = t[None, :] >= onsets[:, None]
    pre = ~post & (t[None, :] >= warmup)
    frame_alarm = alarm.any(axis=2)
    true_alarm = (alarm & truth[:, None, :]).any(axis=2) & post
    hit = true_alarm.any(axis=1)
    latency = np.where(hit, np.argmax(true_alarm, axis=1) - onsets, -1)
    post_alarm = frame_alarm & post
    has_post = post_alarm.any(axis=1)
    at_first = alarm[np.arange(n), np.argmax(post_alarm, axis=1)]
    inter = (at_first & truth).sum(axis=1)
    union = (at_first | truth).sum(axis=1)
    return {
        "any_alarm": frame_alarm.any(axis=1),
        "pre_alarm": (frame_alarm & pre).any(axis=1),
        "hit": hit,
        "latency": latency,
        "exact": has_post & (at_first == truth).all(axis=1),
        "iou": np.where(has_post, inter / np.maximum(union, 1), 0.0),
    }


def _row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(SCORECARD_FIELDS)
    row.update(values)
    return row


def _frame_row(detector: str, calibration: str, threshold: float, summary: Mapping[str, Any]) -> dict[str, Any]:
    regime = Regime(summary["regime"])
    clean = summary["expectation"] == Expectation.NONE.value
    recall = {Regime.LOCALIZED_BURST: summary["event_detection_rate"], Regime.GLOBAL_SHOCK: summary["detection_rate"]}
    return _row(
        track="frame",
        detector=detector,
        regime=regime.value,
        calibration=calibration,
        threshold=threshold,
        n=summary["trials"],
        fp_rate=summary["false_positive_rate"],
        fp_ci95_high=summary["detection_rate_ci95_high"] if clean else None,
        recall=recall.get(regime),
        exact_match_rate=summary["exact_match_rate"],
        mean_iou=summary["mean_overlap"],
        mean_detected_blocks=summary["mean_detected_blocks"],
    )


def _stream_row(
    detector: str, regime: StreamRegime, threshold: float, outcomes: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    n = int(outcomes["any_alarm"].size)
    base = {"track": "stream", "detector": detector, "regime": regime.value, "calibration": "calibrated",
            "threshold": threshold, "n": n}
    if STREAM_EXPECTATION[regime] is Expectation.NONE:
        alarms = int(outcomes["any_alarm"].sum())
        return _row(**base, fp_rate=_round(alarms / n), fp_ci95_high=_round(wilson_interval(alarms, n)[1]))
    latency = outcomes["latency"][outcomes["hit"]].astype(np.float64)
    stats = {}
    if latency.size:
        stats = {
            "latency_mean": _round(latency.mean()),
            "latency_median": _round(np.median(latency)),
            "latency_p90": _round(np.percentile(latency, 90)),
            "latency_max": _round(latency.max()),
        }
    return _row(
        **base,
        early_alarm_rate=_round(outcomes["pre_alarm"].mean()),
        recall=_round(outcomes["hit"].mean()),
        exact_match_rate=_round(outcomes["exact"].mean()),
        mean_iou=_round(outcomes["iou"].mean()),
        **stats,
    )


def _new_detector(name: str, cc: CompareConfig) -> Detector:
    cls = get_detector_class(name)
    kwargs = {"warmup": cc.stream.warmup} if issubclass(cls, TemporalDetector) else {}
    return create_detector(name, cc.config, **kwargs)


def fit_detectors(cc: CompareConfig) -> dict[str, Detector]:
    """Create and fit ``cc.detectors`` on the shared clean fit data (public for ``stress``)."""
    return _fit_detectors(cc)


def _fit_detectors(cc: CompareConfig) -> dict[str, Detector]:
    detectors = {name: _new_detector(name, cc) for name in cc.detectors}
    needs_frames = any(d.requires_fit and not d.temporal for d in detectors.values())
    needs_streams = any(d.requires_fit and d.temporal for d in detectors.values())
    if needs_frames:
        frames = np.concatenate(
            [generate_batch(r, range(cc.fit_trials), cc.config, PURPOSE_FIT)[0] for r in CLEAN_FRAME_REGIMES]
        )
    if needs_streams:
        streams = np.concatenate(
            [generate_streams(r, range(cc.fit_streams), cc.config, cc.stream, PURPOSE_STREAM_FIT)[0]
             for r in CLEAN_STREAM_REGIMES]
        )
    for detector in detectors.values():
        if detector.requires_fit:
            detector.fit(streams if detector.temporal else frames)
    return detectors


class _Timer:
    """Accumulates CPU time (``time.process_time``) and units scored per key."""

    def __init__(self) -> None:
        self.seconds: dict[tuple[str, str], float] = {}
        self.units: dict[tuple[str, str], int] = {}

    def score(self, key: tuple[str, str], detector: Detector, data: np.ndarray, units: int) -> np.ndarray:
        started = time.process_time()
        scores = detector.score(data)
        self.seconds[key] = self.seconds.get(key, 0.0) + time.process_time() - started
        self.units[key] = self.units.get(key, 0) + units
        return scores

    def report(self) -> list[dict[str, Any]]:
        return [
            {"track": track, "detector": name, "frames": self.units[(track, name)],
             "cpu_seconds": self.seconds[(track, name)],
             "cpu_us_per_frame": 1e6 * self.seconds[(track, name)] / self.units[(track, name)]}
            for track, name in sorted(self.seconds)
        ]


def frame_calibration(cc: CompareConfig, detectors: Mapping[str, Detector]) -> tuple[dict[str, float], list]:
    """Calibrate every frame (non-temporal) detector on held-out clean frames.

    Returns ``(thresholds, calibration records)``; shared by ``compare`` and ``stress``.
    """
    calibration_frames = {
        r.value: generate_batch(r, range(cc.calibration_trials), cc.config, PURPOSE_CALIBRATION)[0]
        for r in CLEAN_FRAME_REGIMES
    }
    thresholds: dict[str, float] = {}
    calibrations: list[dict[str, Any]] = []
    for name, detector in detectors.items():
        if detector.temporal:
            continue
        maxima = {r: detector.score(frames).max(axis=1) for r, frames in calibration_frames.items()}
        calibration = calibrate_threshold(maxima, cc.target_fp)
        calibrations.append({"track": "frame", "detector": name, "unit": "frame", **calibration})
        thresholds[name] = calibration["threshold"]
    return thresholds, calibrations


def _frame_track(cc: CompareConfig, detectors: Mapping[str, Detector], timer: _Timer) -> tuple[list, list]:
    thresholds, calibrations = frame_calibration(cc, detectors)
    evaluation = {r: generate_batch(r, range(cc.trials), cc.config) for r in Regime}
    indices = np.arange(cc.trials)
    rows: list[dict[str, Any]] = []
    for name, calibrated in thresholds.items():
        detector = detectors[name]
        scored = {r: timer.score(("frame", name), detector, x, len(x)) for r, (x, _) in evaluation.items()}
        settings = [(name, "calibrated", calibrated)]
        if name == "oes32":
            settings.append((REFERENCE_LABEL, "fixed v0.1 reference", core.Config().threshold))
        for label, how, threshold in settings:
            for regime, (_, truth) in evaluation.items():
                batch = evaluate_batch(regime, indices, scored[regime], truth, cc.config, threshold=threshold)
                rows.append(_frame_row(label, how, threshold, summarize(batch)))
    return rows, calibrations


def _stream_chunks(regime: StreamRegime, count: int, cc: CompareConfig, purpose: int):
    for start in range(0, count, STREAM_CHUNK):
        yield generate_streams(regime, range(start, min(start + STREAM_CHUNK, count)), cc.config, cc.stream, purpose)


def stream_calibration(cc: CompareConfig, detectors: Mapping[str, Detector]) -> tuple[dict[str, float], list]:
    """Calibrate every detector on held-out clean streams (unit = whole stream after warm-up).

    Returns ``(thresholds, calibration records)``; shared by ``compare`` and ``stress``.
    """
    warmup = cc.stream.warmup
    maxima: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in detectors}
    for regime in CLEAN_STREAM_REGIMES:
        for x, _, _ in _stream_chunks(regime, cc.calibration_streams, cc, PURPOSE_STREAM_CALIBRATION):
            for name, detector in detectors.items():
                scores = detector.score(x)[:, warmup:]
                maxima[name].setdefault(regime.value, []).append(scores.max(axis=(1, 2)))
    calibrations = []
    thresholds = {}
    for name in detectors:
        calibration = calibrate_threshold({r: np.concatenate(v) for r, v in maxima[name].items()}, cc.target_fp)
        calibrations.append({"track": "stream", "detector": name, "unit": "stream (all steps after warm-up)",
                             **calibration})
        thresholds[name] = calibration["threshold"]
    return thresholds, calibrations


def _stream_track(cc: CompareConfig, detectors: Mapping[str, Detector], timer: _Timer) -> tuple[list, list]:
    warmup = cc.stream.warmup
    thresholds, calibrations = stream_calibration(cc, detectors)
    rows = []
    for regime in StreamRegime:
        outcomes: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in detectors}
        for x, truth, onsets in _stream_chunks(regime, cc.streams, cc, PURPOSE_STREAM_EVAL):
            for name, detector in detectors.items():
                scores = timer.score(("stream", name), detector, x, x.shape[0] * x.shape[1])
                outcomes[name].append(stream_outcomes(scores, truth, onsets, thresholds[name], warmup))
        for name in detectors:
            merged = {key: np.concatenate([o[key] for o in outcomes[name]]) for key in outcomes[name][0]}
            rows.append(_stream_row(name, regime, thresholds[name], merged))
    return rows, calibrations


def run_compare(cc: CompareConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the comparison. Returns ``(scorecard, timing)``; only the scorecard is deterministic."""
    timer = _Timer()
    detectors = _fit_detectors(cc)
    rows: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    if "frame" in cc.tracks:
        frame_rows, frame_cal = _frame_track(cc, detectors, timer)
        rows += frame_rows
        calibrations += frame_cal
    if "stream" in cc.tracks:
        stream_rows, stream_cal = _stream_track(cc, detectors, timer)
        rows += stream_rows
        calibrations += stream_cal
    config = cc.config.to_dict()
    config["threshold"] = None
    scorecard = {
        "project": PROJECT,
        "version": __version__,
        "kind": "scorecard",
        "config": config,
        "compare": cc.to_dict(),
        "calibration_method": (
            "per clean regime (stable, noisy): smallest threshold reached by at most floor(target_fp*n) held-out "
            "calibration units (unit = frame, or whole stream after warm-up, scored by its maximum block score); "
            "threshold = maximum over clean regimes"
        ),
        "detectors": {
            name: {"class": type(d).__name__, "rule": d.rule, "temporal": d.temporal, "params": d.params()}
            for name, d in detectors.items()
        },
        "calibration": calibrations,
        "rows": rows,
    }
    return scorecard, timer.report()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def scorecard_markdown(scorecard: Mapping[str, Any], timing: Sequence[Mapping[str, Any]]) -> str:
    """Markdown report (includes machine-dependent CPU timings, so it is not hashed)."""
    cc = scorecard["compare"]
    lines = [
        f"# {scorecard['project']} scorecard (v{scorecard['version']})",
        "",
        "Synthetic data only; these numbers describe software behaviour on the benchmark's own regimes, "
        "not real-world performance.",
        "",
        f"Seed {scorecard['config']['seed']}; frame track {cc['trials']} trials/regime; stream track "
        f"{cc['streams']} streams/regime of {cc['stream']['steps']} steps (warm-up {cc['stream']['warmup']}, onset "
        f"{cc['stream']['onset_min']}–{cc['stream']['onset_max']}); target FP {cc['target_fp']:.3f}.",
        "",
        "## Calibration",
        "",
        scorecard["calibration_method"] + ".",
        "",
        "| track | detector | threshold | calibration FP (per clean regime) |",
        "|---|---|---|---|",
    ]
    for c in scorecard["calibration"]:
        per = ", ".join(f"{r}: {v['calibration_fp_rate']:.3f}" for r, v in c["per_regime"].items())
        lines.append(f"| {c['track']} | {c['detector']} | {c['threshold']:.6g} | {per} |")
    header = ("| detector | regime | FP | early alarm | recall | exact | IoU | latency mean / median / p90 "
              "(steps) |")
    for track in ("frame", "stream"):
        rows = [r for r in scorecard["rows"] if r["track"] == track]
        if not rows:
            continue
        lines += ["", f"## {track.capitalize()} track", "", header, "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            latency = "–" if r["latency_mean"] is None else (
                f"{r['latency_mean']:.2f} / {r['latency_median']:.1f} / {r['latency_p90']:.1f}")
            lines.append(
                f"| {r['detector']} | {r['regime']} | {_cell(r['fp_rate'])} | {_cell(r['early_alarm_rate'])} | "
                f"{_cell(r['recall'])} | {_cell(r['exact_match_rate'])} | {_cell(r['mean_iou'])} | {latency} |"
            )
    lines += ["", "## CPU time (machine-dependent)", "", "| track | detector | frames scored | CPU µs per frame |",
              "|---|---|---|---|"]
    for t in timing:
        lines.append(f"| {t['track']} | {t['detector']} | {t['frames']} | {t['cpu_us_per_frame']:.2f} |")
    lines += ["", "Frame = one 512-channel frame (frame track) or one time step (stream track); measured with "
              "`time.process_time`, so values vary by machine and load.", ""]
    return "\n".join(lines)


def write_compare_outputs(
    scorecard: Mapping[str, Any],
    timing: Sequence[Mapping[str, Any]],
    out_dir: Path,
    prefix: str,
    metadata: Mapping[str, Any],
    fields: Sequence[str] = SCORECARD_FIELDS,
    markdown: Any = None,
) -> dict[str, Path]:
    """Write the deterministic scorecard (JSON, CSV, manifest) and the non-deterministic extras.

    ``fields`` and ``markdown`` (a ``(scorecard, timing) -> str`` function) let the stress
    suite reuse this writer; they default to the compare scorecard's.
    """
    markdown = scorecard_markdown if markdown is None else markdown
    out_dir = Path(out_dir)
    paths = {
        "scorecard_json": out_dir / f"{prefix}_scorecard.json",
        "scorecard_csv": out_dir / f"{prefix}_scorecard.csv",
        "manifest": out_dir / f"{prefix}_manifest.json",
        "scorecard_md": out_dir / f"{prefix}_scorecard.md",
        "timing": out_dir / f"{prefix}_timing.json",
        "metadata": out_dir / f"{prefix}_run_metadata.json",
    }
    json_hash = atomic_write(paths["scorecard_json"], json_bytes(scorecard))
    csv_hash = atomic_write(paths["scorecard_csv"], csv_bytes(scorecard["rows"], fields))
    manifest = {
        "project": PROJECT,
        "version": __version__,
        "files": {
            paths["scorecard_json"].name: {"sha256": json_hash},
            paths["scorecard_csv"].name: {"sha256": csv_hash},
        },
    }
    manifest_hash = atomic_write(paths["manifest"], json_bytes(manifest))
    atomic_write(paths["scorecard_md"], markdown(scorecard, timing).encode("utf-8"))
    atomic_write(paths["timing"], json_bytes({"timer": "time.process_time", "machine_dependent": True,
                                              "entries": list(timing)}))
    atomic_write(paths["metadata"], json_bytes({**metadata, "manifest_sha256": manifest_hash}))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _name_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def add_compare_parser(sub: Any) -> argparse.ArgumentParser:
    """Register the ``compare`` subcommand on an argparse subparsers object."""
    parser = sub.add_parser("compare", help="compare detectors and write a scorecard (JSON, CSV, Markdown)")
    core._add_common(parser, "oes_resilience_compare")
    parser.add_argument("--fit-trials", type=core._positive_int, default=1000, help="clean fit frames per regime")
    parser.add_argument("--calibration-trials", type=core._positive_int, default=1000,
                        help="held-out clean calibration frames per regime")
    parser.add_argument("--streams", type=core._positive_int, default=1000, help="evaluation streams per regime")
    parser.add_argument("--calibration-streams", type=core._positive_int, default=1000,
                        help="held-out clean calibration streams per regime")
    parser.add_argument("--fit-streams", type=core._positive_int, default=200,
                        help="clean fit streams per regime for temporal detectors that need fitting (default 200)")
    parser.add_argument("--steps", type=int, default=64, help="steps per stream (default 64)")
    parser.add_argument("--warmup", type=int, default=16, help="event-free warm-up steps (default 16)")
    parser.add_argument("--target-fp", type=float, default=0.01, help="calibration target FP rate (default 0.01)")
    parser.add_argument("--detectors", type=_name_list, default=DEFAULT_DETECTORS,
                        help=f"comma-separated detector names (default {','.join(DEFAULT_DETECTORS)})")
    parser.add_argument("--tracks", type=_name_list, default=TRACKS, help="frame,stream (default both)")
    parser.add_argument("--plugins", action="store_true", help="load detectors from installed entry points")
    parser.add_argument("--verify", action="store_true",
                        help="recompute the scorecard and exit 4 unless its hash matches")
    return parser


def cmd_compare(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Handler for ``compare``."""
    if args.plugins:
        load_plugins()
    cc = CompareConfig(
        config=core._config_from_args(args),
        stream=StreamConfig(steps=args.steps, warmup=args.warmup),
        trials=args.trials,
        fit_trials=args.fit_trials,
        calibration_trials=args.calibration_trials,
        streams=args.streams,
        calibration_streams=args.calibration_streams,
        fit_streams=args.fit_streams,
        target_fp=args.target_fp,
        detectors=args.detectors,
        tracks=args.tracks,
    )
    started = time.perf_counter()
    scorecard, timing = run_compare(cc)
    paths = write_compare_outputs(scorecard, timing, args.out_dir, args.prefix,
                                  core.run_metadata(argv, time.perf_counter() - started))
    code = core.EXIT_OK
    if args.verify:
        first = core.sha256_file(paths["scorecard_json"])
        second = core.hashlib.sha256(json_bytes(run_compare(cc)[0])).hexdigest()
        match = first == second
        print(f"reproducibility check: {'hash match' if match else 'HASH MISMATCH'} ({first[:16]}…)")
        code = core.EXIT_OK if match else core.EXIT_NOT_REPRODUCIBLE
    if not args.quiet:
        print(scorecard_markdown(scorecard, timing))
        for key, path in paths.items():
            print(f"{key}: {path}")
    return code
