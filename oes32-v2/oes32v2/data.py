"""Data access, chronological split and validation-anomaly injection for OES-32-v2.

Reuses the frozen v1 loaders in oes_resilience.smap_msl (unchanged) for labels and
series, so v1 and v2 read identical bytes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from oes_resilience import smap_msl as v1  # noqa: E402

GAP = 128                 # >= maximum window length in the search space
VAL_FRACTION = 0.30
MIN_SEGMENT = 128
INJECTION_BASE_SEED = 20261013
INJECTION_SPACING = 400   # one injected event per 400 validation samples (at least 1)
INJECTION_TYPES = ("spike", "level_shift", "drift", "variance_burst")
DATASETS = v1.DATASETS


@dataclass
class Split:
    n_train: int
    fit_end: int          # fit = [0, fit_end)
    val_start: int        # val = [val_start, n_train)
    has_val: bool


def split_for(n: int) -> Split:
    val_len = int(np.floor(VAL_FRACTION * n))
    fit_end = n - val_len - GAP
    ok = fit_end >= MIN_SEGMENT and val_len >= MIN_SEGMENT
    return Split(n, fit_end if ok else n, n - val_len if ok else n, ok)


def channels(data_dir: str | Path) -> list[v1.Channel]:
    chans = v1.load_labels(Path(data_dir) / "labeled_anomalies.csv")
    return sorted(chans, key=lambda c: (DATASETS.index(c.spacecraft), c.chan_id))


def load(data_dir: str | Path, channel: v1.Channel) -> tuple[np.ndarray, np.ndarray]:
    return v1.load_series(data_dir, channel)


def robust_scale_for_injection(fit: np.ndarray) -> float:
    mad = 1.4826 * float(np.median(np.abs(fit - np.median(fit))))
    if mad > 1e-9:
        return mad
    sd = float(np.std(fit))
    return sd if sd > 1e-9 else 1.0


def inject(val: np.ndarray, fit: np.ndarray, channel_index: int) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    """Deterministic synthetic anomalies in a copy of the validation segment.

    Returns (injected series, inclusive event windows in segment coordinates, types).
    """
    rng = np.random.default_rng(INJECTION_BASE_SEED + channel_index)
    x = np.array(val, dtype=np.float64, copy=True)
    n = x.size
    k = max(1, n // INJECTION_SPACING)
    slot = n // k
    s = robust_scale_for_injection(fit)
    events, kinds = [], []
    for i in range(k):
        kind = INJECTION_TYPES[int(rng.integers(len(INJECTION_TYPES)))]
        lo_len, hi_len = {"spike": (1, 3), "level_shift": (50, 150), "drift": (100, 200), "variance_burst": (50, 150)}[kind]
        length = int(min(rng.integers(lo_len, hi_len + 1), max(1, slot // 2)))
        start = i * slot + int(rng.integers(slot // 4, max(slot // 4 + 1, slot - length - slot // 8)))
        end = min(start + length - 1, n - 1)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "spike":
            x[start:end + 1] += sign * rng.uniform(4, 8) * s
        elif kind == "level_shift":
            x[start:end + 1] += sign * rng.uniform(2, 4) * s
        elif kind == "drift":
            x[start:end + 1] += sign * rng.uniform(3, 6) * s * np.linspace(0, 1, end - start + 1)
        else:
            x[start:end + 1] += rng.normal(0, rng.uniform(2, 4) * s, end - start + 1)
        events.append((start, end))
        kinds.append(kind)
    return x, events, kinds


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    v1_manifest = REPO_ROOT / "reports" / "smap_msl_data.sha256"
    files = v1.read_manifest(v1_manifest)
    verified = v1.verify_data(data_dir, v1_manifest)
    rows = []
    for c in channels(data_dir):
        tr = np.load(data_dir / "train" / f"{c.chan_id}.npy", allow_pickle=False)
        te = np.load(data_dir / "test" / f"{c.chan_id}.npy", allow_pickle=False)
        sp = split_for(tr.shape[0])
        rows.append({
            "spacecraft": c.spacecraft, "chan_id": c.chan_id,
            "train_shape": list(tr.shape), "test_shape": list(te.shape),
            "fit": [0, sp.fit_end], "gap": [sp.fit_end, sp.val_start] if sp.has_val else None,
            "val": [sp.val_start, sp.n_train] if sp.has_val else None, "has_validation": sp.has_val,
            "test_events_inclusive": [list(e) for e in c.events], "test_event_classes": list(c.classes),
            "label_rows": c.label_rows,
        })
    return {
        "schema": "oes32-v2/dataset-manifest/1",
        "dataset": "NASA SMAP/MSL telemetry anomaly dataset (Hundman et al. 2018, KDD '18, doi:10.1145/3219819.3219845; telemanom)",
        "source_urls": list(v1.DEFAULT_URLS),
        "labels": "labeled_anomalies.csv from the same archive (bytes listed below)",
        "not_redistributed": True,
        "reused_v1_manifest": {"path": "reports/smap_msl_data.sha256", "sha256": sha256_file(v1_manifest),
                               "files_verified_now": verified["files"]},
        "files_sha256": files,
        "signal": "column 0 (telemetry value), float64",
        "time_ordering": "array row order; train precedes test for every channel (telemanom convention)",
        "label_mapping": "rows sharing (spacecraft, chan_id) pooled; overlapping/adjacent inclusive windows merged (v1 merge_windows)",
        "split_rule": {"val_fraction": VAL_FRACTION, "gap": GAP, "min_segment": MIN_SEGMENT},
        "excluded_from_evaluation": "channels without a labelled row (T-10, as in v1)",
        "channels": rows,
        "counts": {d: {"channels": sum(r["spacecraft"] == d for r in rows),
                       "with_validation": sum(r["spacecraft"] == d and r["has_validation"] for r in rows),
                       "test_events": sum(len(r["test_events_inclusive"]) for r in rows if r["spacecraft"] == d)}
                   for d in DATASETS},
    }


if __name__ == "__main__":
    out = V2_ROOT / "data" / "dataset_manifest.json"
    out.write_text(json.dumps(build_manifest(sys.argv[1]), indent=1) + "\n", encoding="utf-8")
    print(out)
