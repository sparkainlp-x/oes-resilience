# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Online Peak-Over-Threshold (POT) calibration for one-dimensional anomaly scores.

The estimator uses a Generalized Pareto tail approximation with method-of-moments
parameters, avoiding a mandatory SciPy dependency. Flagged observations are excluded
from the rolling calibration window. The initial and rolling windows fall back to an
empirical upper quantile until enough tail exceedances are available.

This component is intentionally separate from the benchmark's frozen replay and
preregistered SMAP/MSL threshold policies. It cannot distinguish a sustained anomaly
from a legitimate regime change using scores alone; callers should only call
:meth:`reset` after an independently validated regime-change signal.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from .core import _require_finite, _require_int

_MIN_SCALE = 1e-9


class POTThreshold:
    """Streaming EVT threshold with anomaly masking.

    Args:
        risk_level: Desired upper-tail exceedance probability in ``(0, 1)``.
        init_quantile: POT base-threshold quantile in ``(0, 1)``. It must leave a
            larger empirical tail fraction than ``risk_level``.
        window_size: Maximum number of accepted clean scores retained for online
            recalibration.
        min_exceedances: Minimum exceedance count needed for the GPD moment fit;
            otherwise the empirical ``1-risk_level`` quantile is used.

    ``fit_calibration`` accepts at least 20 finite one-dimensional clean scores.
    ``update`` evaluates a score against the threshold that existed before the
    observation, returns ``(is_anomaly, threshold_used)``, and only adds non-anomalies
    to the rolling history. The comparison is strict (``score > threshold``), so
    threshold ties are not treated as anomalies.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_quantile: float = 0.98,
        window_size: int = 1000,
        min_exceedances: int = 10,
    ) -> None:
        self.risk_level = _require_finite("risk_level", risk_level, 0.0)
        if not 0.0 < self.risk_level < 1.0:
            raise ValueError("risk_level must be in (0, 1).")
        self.init_quantile = _require_finite("init_quantile", init_quantile, 0.0)
        if not 0.0 < self.init_quantile < 1.0:
            raise ValueError("init_quantile must be in (0, 1).")
        if self.risk_level >= 1.0 - self.init_quantile:
            raise ValueError("risk_level must be smaller than the POT exceedance fraction (1 - init_quantile).")
        self.window_size = _require_int("window_size", window_size, 20)
        self.min_exceedances = _require_int("min_exceedances", min_exceedances, 1)
        self.history: deque[float] = deque(maxlen=self.window_size)
        self.threshold: float | None = None
        self.base_threshold: float | None = None
        self.shape: float | None = None
        self.scale: float | None = None
        self.exceedance_count = 0
        self.fitted = False

    @staticmethod
    def _scores(values: Any) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("calibration scores must be a non-empty one-dimensional array.")
        if not np.isfinite(scores).all():
            raise ValueError("calibration scores must contain only finite values.")
        return scores

    def fit_calibration(self, scores: Any) -> POTThreshold:
        """Fit the initial tail model on clean calibration scores; returns ``self``."""
        values = self._scores(scores)
        if values.size < 20:
            raise ValueError("at least 20 clean calibration scores are required.")
        self.history.clear()
        self.history.extend(values[-self.window_size :].tolist())
        self._refit(values)
        self.fitted = True
        return self

    def reset(self, clean_scores: Any) -> POTThreshold:
        """Reinitialize after an independently confirmed regime change.

        Reset scores must be known-clean observations from the new regime. This method
        is deliberately explicit: automatically treating an arbitrary sustained event
        as a benign regime change would leak anomaly values into the threshold model.
        """
        return self.fit_calibration(clean_scores)

    def _refit(self, values: np.ndarray) -> None:
        u = float(np.quantile(values, self.init_quantile))
        exceedances = values[values > u] - u
        self.base_threshold = u
        self.exceedance_count = int(exceedances.size)

        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust_scale = max(1.4826 * mad, _MIN_SCALE)
        self.shape = 0.0
        self.scale = robust_scale

        # Sparse tails are better handled by a finite-sample empirical quantile than
        # by extrapolating a poorly identified GPD.
        if exceedances.size < self.min_exceedances:
            self.threshold = float(np.quantile(values, 1.0 - self.risk_level))
            return

        mean = float(np.mean(exceedances))
        variance = float(np.var(exceedances))
        if not math.isfinite(mean) or mean <= _MIN_SCALE or not math.isfinite(variance):
            self.threshold = float(np.quantile(values, 1.0 - self.risk_level))
            return

        # GPD moments: xi = 0.5 * (1 - mean^2 / variance), beta = mean * (1 - xi).
        # Clip xi to the finite-variance region for a stable method-of-moments fit.
        if variance <= _MIN_SCALE * _MIN_SCALE:
            shape = 0.0
        else:
            shape = 0.5 * (1.0 - (mean * mean / variance))
        shape = float(np.clip(shape, -0.45, 0.45))
        scale = mean * (1.0 - shape)
        if not math.isfinite(scale) or scale <= _MIN_SCALE:
            self.threshold = float(np.quantile(values, 1.0 - self.risk_level))
            return

        tail_rate = exceedances.size / values.size
        if tail_rate <= self.risk_level:
            self.threshold = float(np.quantile(values, 1.0 - self.risk_level))
            return
        ratio = tail_rate / self.risk_level
        log_ratio = math.log(ratio)
        if abs(shape) < 1e-6:
            excess_quantile = scale * log_ratio
        else:
            excess_quantile = scale * math.expm1(shape * log_ratio) / shape
        candidate = u + max(0.0, excess_quantile)
        if shape < 0:
            candidate = min(candidate, u - scale / shape)
        if not math.isfinite(candidate):
            candidate = float(np.quantile(values, 1.0 - self.risk_level))
        self.shape = shape
        self.scale = scale
        self.threshold = max(u, float(candidate))

    def update(self, current_score: float) -> tuple[bool, float]:
        """Score one observation, then update the model if it is not flagged."""
        if not self.fitted or self.threshold is None:
            raise RuntimeError("POTThreshold must be fitted with clean calibration scores first.")
        score = _require_finite("current_score", current_score)
        used_threshold = self.threshold
        is_anomaly = score > used_threshold
        if not is_anomaly:
            self.history.append(score)
            self._refit(np.asarray(self.history, dtype=np.float64))
        return is_anomaly, used_threshold

    def params(self) -> dict[str, Any]:
        """Return JSON-compatible configuration and current tail estimates."""
        return {
            "risk_level": self.risk_level,
            "init_quantile": self.init_quantile,
            "window_size": self.window_size,
            "min_exceedances": self.min_exceedances,
            "base_threshold": self.base_threshold,
            "threshold": self.threshold,
            "shape": self.shape,
            "scale": self.scale,
            "exceedance_count": self.exceedance_count,
            "history_size": len(self.history),
        }
