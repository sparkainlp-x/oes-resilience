"""Per-channel pipelines for every method on a target segment (validation-injected or test).

Fitting uses the fit segment; thresholds use the clean calibration segment; nothing is
fitted or calibrated on the target segment.
"""
from __future__ import annotations

import numpy as np

from . import core
from .data import DATASETS, v1

IF_SEEDS = (0, 1, 2, 3, 4)


def v1_pred(v1_train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Frozen v1: standardise on v1's train, OES32 window 32, v1 calibration at 1 % on train scores, alarm >= T."""
    (z_tr, z_t), _, _ = v1.standardize(v1_train, target)
    s_tr = v1.method_scores("oes32", z_tr, {"window": 32})
    s_t = v1.method_scores("oes32", z_t, {"window": 32})
    T = core.v1_calibrate(s_tr, 0.01)
    return core.v1_alarm_runs(s_t, T), s_t, T


def _xs(fit, calib, target, eps=1e-9):
    med, mad = core.robust_stats(fit)
    return [core.scale(a, med, mad, eps, "none") for a in (fit, calib, target)]


def baseline_pred(name: str, fit, calib, target, seed: int = 0):
    xf, xc, xt = _xs(fit, calib, target)
    if name == "rolling_mad":
        sc, st = core.rolling_mad_score(xc), core.rolling_mad_score(xt)
        T = core.v1_calibrate(sc, 0.01)
    elif name == "train_percentile":
        st = np.abs(xt)
        T = float(np.percentile(np.abs(xf), 99.5))
    elif name == "pca_residual":
        m = core.PCAModel.fit(xf)
        sc, st = m.score(xc), m.score(xt)
        T = core.v1_calibrate(sc, 0.01)
    elif name == "isolation_forest":
        sc, st = core.iforest_fit_score(xf, [xc, xt], seed)
        T = core.v1_calibrate(sc, 0.01)
    else:
        raise ValueError(name)
    return core.alerts(st, T, None), st, T


def v2_global_stats(fits_by_sc: dict[str, list[np.ndarray]]) -> dict:
    return {d: core.robust_stats(np.concatenate(fits_by_sc[d])) for d in DATASETS}


def v2_scores(cfg: dict, fit, calib, target, stats: tuple[float, float]):
    n = cfg["normalization"]
    xf, xc, xt = (core.scale(a, *stats, n["epsilon"], n["clip"]) for a in (fit, calib, target))
    bc, bt = core.oes_v2_score(xc, cfg), core.oes_v2_score(xt, cfg)
    comp = cfg["complementary"]
    if comp["feature"] == "none":
        return bc, bt
    if comp["feature"] == "rolling_mad":
        cc, ct = core.rolling_mad_score(xc), core.rolling_mad_score(xt)
    else:
        m = core.PCAModel.fit(xf)
        cc, ct = m.score(xc), m.score(xt)
    na, nb = max(float(np.quantile(bc, 0.99)), 1e-12), max(float(np.quantile(cc, 0.99)), 1e-12)
    a_c, a_t, b_c, b_t = bc / na, bt / na, cc / nb, ct / nb
    how = comp["combination"]
    if how == "normalized_mean":
        return (a_c + b_c) / 2, (a_t + b_t) / 2
    if how == "max":
        return np.maximum(a_c, b_c), np.maximum(a_t, b_t)
    w = float(comp["weight"])
    return w * a_c + (1 - w) * b_c, w * a_t + (1 - w) * b_t


def v2_pred(cfg: dict, fit, calib, target, stats, pooled_calib: np.ndarray | None = None):
    sc, st = v2_scores(cfg, fit, calib, target, stats)
    T = core.threshold_from(sc, cfg["threshold"], pooled_calib)
    return core.alerts(st, T, cfg["postprocess"]), st, T
