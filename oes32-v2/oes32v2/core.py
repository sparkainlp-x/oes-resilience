"""OES-32-v2 scores, baselines, alert post-processing and event metrics (offline benchmark mode)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .data import v1

CHUNK = 65536


# ---------------------------------------------------------------- scaling
def robust_stats(fit: np.ndarray) -> tuple[float, float]:
    med = float(np.median(fit))
    return med, 1.4826 * float(np.median(np.abs(fit - med)))


def scale(raw: np.ndarray, med: float, mad: float, eps: float, clip: Any) -> np.ndarray:
    x = (np.asarray(raw, dtype=np.float64) - med) / max(mad, eps)
    if clip not in (None, "none"):
        x = np.clip(x, -float(clip), float(clip))
    return x


# ---------------------------------------------------------------- window scores
def _padded_view(a: np.ndarray, w: int) -> np.ndarray:
    pad = np.concatenate([np.full(w - 1, a[0]), a])
    return np.lib.stride_tricks.sliding_window_view(pad, w)


def window_agg(a: np.ndarray, w: int, agg: str, k: int = 1) -> np.ndarray:
    """Trailing-window aggregation of a (n,) array, padded with a[0] (causal)."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    if agg == "mean":
        pad = np.concatenate([np.full(w - 1, a[0]), a])
        c = np.concatenate([[0.0], np.cumsum(pad)])
        return (c[w:] - c[:-w])[:n] / w
    view = _padded_view(a, w)
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        v = view[s:s + CHUNK]
        if agg == "max":
            out[s:s + CHUNK] = v.max(axis=1)
        elif agg == "median":
            out[s:s + CHUNK] = np.median(v, axis=1)
        elif agg == "topk_mean":
            kk = min(k, w)
            out[s:s + CHUNK] = np.partition(v, w - kk, axis=1)[:, w - kk:].mean(axis=1)
        else:
            raise ValueError(agg)
    return out


def smooth(s: np.ndarray, m: Any) -> np.ndarray:
    if m in (None, "none"):
        return s
    return window_agg(s, int(m), "mean")


def oes_v2_score(x: np.ndarray, cfg: dict) -> np.ndarray:
    w = cfg["window"]
    return smooth(window_agg(np.abs(x), w["length"], w["aggregation"], w.get("topk", 1)), w["smoothing"])


# ---------------------------------------------------------------- baselines
def rolling_mad_score(x: np.ndarray, w: int = 64) -> np.ndarray:
    view = _padded_view(np.asarray(x, dtype=np.float64), w)
    out = np.empty(x.size)
    for s in range(0, x.size, CHUNK):
        v = view[s:s + CHUNK]
        med = np.median(v, axis=1)
        mad = 1.4826 * np.median(np.abs(v - med[:, None]), axis=1)
        out[s:s + CHUNK] = np.abs(v[:, -1] - med) / np.maximum(mad, 1.0)
    return out


def lag_windows(x: np.ndarray, L: int = 16) -> np.ndarray:
    return _padded_view(np.clip(np.asarray(x, dtype=np.float64), -10, 10), L)


@dataclass
class PCAModel:
    mean: np.ndarray
    comps: np.ndarray

    @classmethod
    def fit(cls, x_fit: np.ndarray, L: int = 16) -> "PCAModel":
        X = lag_windows(x_fit, L)
        mu = X.mean(axis=0)
        _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
        var = sv ** 2
        tot = var.sum()
        if tot <= 0:
            k = 1
        else:
            k = int(np.searchsorted(np.cumsum(var) / tot, 0.90) + 1)
        return cls(mu, vt[: min(k, 8)])

    def score(self, x: np.ndarray, L: int = 16) -> np.ndarray:
        X = lag_windows(x, L) - self.mean
        R = X - (X @ self.comps.T) @ self.comps
        return np.sqrt((R ** 2).sum(axis=1))


def iforest_fit_score(x_fit: np.ndarray, segments: Sequence[np.ndarray], seed: int) -> list[np.ndarray]:
    from sklearn.ensemble import IsolationForest
    X = lag_windows(x_fit)
    if X.shape[0] > 5000:
        X = X[np.linspace(0, X.shape[0] - 1, 5000).astype(int)]
    model = IsolationForest(n_estimators=100, max_samples=min(256, X.shape[0]), random_state=seed).fit(X)
    return [-model.score_samples(lag_windows(s)) for s in segments]


def v1_calibrate(scores: np.ndarray, target: float = 0.01) -> float:
    return float(v1.calibrate_threshold({"train": scores}, target)["threshold"])


# ---------------------------------------------------------------- thresholds
def threshold_from(clean_scores: np.ndarray, rule: dict, pooled: np.ndarray | None = None) -> float:
    kind, val = rule["kind"], float(rule["value"])
    if kind == "per_channel_quantile":
        return float(np.quantile(clean_scores, val))
    if kind == "global_quantile":
        return float(np.quantile(pooled, val))
    if kind == "robust_multiplier":
        med = float(np.median(clean_scores))
        return med + val * 1.4826 * float(np.median(np.abs(clean_scores - med)))
    raise ValueError(kind)


# ---------------------------------------------------------------- alerts
def runs(mask: np.ndarray) -> np.ndarray:
    p = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.flatnonzero(np.diff(p))
    return np.stack([d[::2], d[1::2] - 1], axis=1) if d.size else np.zeros((0, 2), dtype=np.int64)


def alerts(score: np.ndarray, T: float, post: dict | None) -> np.ndarray:
    """Predicted events (inclusive [start, end] rows) after hysteresis -> merge -> min duration -> cooldown."""
    if post is None:
        return runs(score > T)
    r = float(post["hysteresis_low_high_ratio"])
    low = score > r * T
    R = runs(low)
    if R.size:
        hi = np.concatenate([[0], np.cumsum(score > T)])
        R = R[(hi[R[:, 1] + 1] - hi[R[:, 0]]) > 0]
    g = int(post["max_merge_gap"])
    if R.shape[0] > 1 and g > 0:
        gaps = R[1:, 0] - R[:-1, 1] - 1
        new = np.concatenate([[True], gaps > g])
        idx = np.flatnonzero(new)
        ends = np.maximum.reduceat(R[:, 1], idx)
        R = np.stack([R[idx, 0], ends], axis=1)
    d = int(post["min_alert_duration"])
    if d > 1 and R.size:
        R = R[(R[:, 1] - R[:, 0] + 1) >= d]
    c = int(post["cooldown"])
    if c > 0 and R.shape[0] > 1:
        keep, last = [0], R[0, 1]
        for i in range(1, R.shape[0]):
            if R[i, 0] - last > c:
                keep.append(i)
                last = R[i, 1]
        R = R[keep]
    return R


def v1_alarm_runs(score: np.ndarray, T: float) -> np.ndarray:
    return runs(score >= T)


# ---------------------------------------------------------------- metrics
def channel_eval(pred: np.ndarray, events: Sequence[tuple[int, int]], n: int) -> dict:
    truth = np.zeros(n, dtype=bool)
    for s, e in events:
        truth[s:e + 1] = True
    alert = np.zeros(n, dtype=bool)
    for s, e in pred:
        alert[s:e + 1] = True
    tp_mask_p = np.concatenate([[0], np.cumsum(truth)])
    pred_match = (tp_mask_p[pred[:, 1] + 1] - tp_mask_p[pred[:, 0]]) > 0 if pred.size else np.zeros(0, bool)
    starts, ends = (pred[:, 0], pred[:, 1]) if pred.size else (np.zeros(0, int), np.zeros(0, int))
    tp = 0
    dups, delays = [], []
    for s, e in events:
        m = (starts <= e) & (ends >= s)
        cnt = int(m.sum())
        if cnt:
            tp += 1
            dups.append(cnt - 1)
            first = int(max(starts[m].min(), s))
            delays.append(first - s)
    fp = int((~pred_match).sum())
    fn = len(events) - tp
    ptp = int((alert & truth).sum())
    pa = alert.copy()
    for (s, e) in events:
        if alert[s:e + 1].any():
            pa[s:e + 1] = True
    return {
        "n": n, "events": len(events), "tp": tp, "fp": fp, "fn": fn,
        "pred_events": int(pred.shape[0]), "dups": dups, "delays": delays,
        "durations": (pred[:, 1] - pred[:, 0] + 1).tolist() if pred.size else [],
        "point_tp": ptp, "point_fp": int((alert & ~truth).sum()), "point_fn": int((~alert & truth).sum()),
        "pa_tp": int((pa & truth).sum()), "pa_fp": int((pa & ~truth).sum()), "pa_fn": int((~pa & truth).sum()),
    }


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    return v1.prf(tp, fp, fn)


def aggregate(rows: Sequence[dict]) -> dict:
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    p, rc, f = prf(tp, fp, fn)
    per = [prf(r["tp"], r["fp"], r["fn"])[2] for r in rows if r["events"] > 0]
    pp, pr, pf = prf(sum(r["point_tp"] for r in rows), sum(r["point_fp"] for r in rows), sum(r["point_fn"] for r in rows))
    ap, ar, af = prf(sum(r["pa_tp"] for r in rows), sum(r["pa_fp"] for r in rows), sum(r["pa_fn"] for r in rows))
    delays = [d for r in rows for d in r["delays"]]
    dups = [d for r in rows for d in r["dups"]]
    durs = [d for r in rows for d in r["durations"]]
    q = lambda a, x: float(np.percentile(a, x)) if a else None  # noqa: E731
    return {
        "channels": len(rows), "events": sum(r["events"] for r in rows), "tp": tp, "fp": fp, "fn": fn,
        "event_precision": p, "event_recall": rc, "event_f1": f,
        "macro_channel_f1": float(np.mean(per)) if per else 0.0,
        "point_precision": pp, "point_recall": pr, "point_f1": pf,
        "point_adjusted_f1_secondary": af,
        "duplicates_per_matched_event": float(np.mean(dups)) if dups else 0.0,
        "delay_median": q(delays, 50), "delay_p25": q(delays, 25), "delay_p75": q(delays, 75), "delay_p90": q(delays, 90),
        "pred_duration_median": q(durs, 50), "pred_duration_p90": q(durs, 90), "pred_events": len(durs),
    }


def bootstrap_f1(rows_by_method: dict[str, list[dict]], seed: int = 42, B: int = 10000) -> dict:
    """Channel-level percentile bootstrap of pooled event F1 for each method (same resampled channels)."""
    names = list(rows_by_method)
    n = len(rows_by_method[names[0]])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    out = {}
    for m in names:
        tp = np.array([r["tp"] for r in rows_by_method[m]], float)[idx].sum(1)
        fp = np.array([r["fp"] for r in rows_by_method[m]], float)[idx].sum(1)
        fn = np.array([r["fn"] for r in rows_by_method[m]], float)[idx].sum(1)
        den = 2 * tp + fp + fn
        out[m] = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return out
