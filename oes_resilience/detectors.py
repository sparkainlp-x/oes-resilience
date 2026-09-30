# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Detector plugin API, registry and the built-in detectors.

A detector maps telemetry to one non-negative score per 32-channel block (per frame, or
per time step for streams); a block is detected when ``score >= threshold``. See
``docs/detector-api.md`` for the full contract and a plugin example.

Built-in detectors:

* ``oes32``: the v0.1 reference score ``0.45 max|x| + 0.35 RMS + 0.20 mean|x|``.
* ``zscore``: robust z-score of each block's RMS against a median/MAD baseline fitted
  on clean frames.
* ``ewma`` and ``cusum``: temporal change detectors on each block's mean, standardised
  against the stream's own warm-up steps.
* ``iforest``: Isolation Forest on per-block features. Optional; needs scikit-learn
  (``pip install oes-resilience[iforest]``). The core stays NumPy-only.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

import numpy as np

from .core import (
    Config,
    ScoreWeights,
    _require_finite,
    _require_int,
    classify_counts,
    score_signals,
    validate_threshold,
)

#: Smallest scale used when standardising, to avoid division by zero on flat data.
MIN_SCALE = 1e-9
#: Entry-point group scanned by :func:`load_plugins`.
ENTRY_POINT_GROUP = "oes_resilience.detectors"

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


@dataclass(frozen=True)
class DetectionResult:
    """Common output of :meth:`Detector.detect`.

    ``scores`` and ``detected`` have the input's leading shape plus a final ``blocks``
    axis: ``(blocks,)`` for one frame, ``(n, blocks)`` for frames or one stream's steps,
    ``(n, steps, blocks)`` for a batch of streams. ``status`` drops the block axis.
    """

    detector: str
    threshold: float
    scores: np.ndarray
    detected: np.ndarray
    status: np.ndarray
    explanation: dict[str, Any] = field(default_factory=dict)

    @property
    def detected_blocks(self) -> Any:
        """Detected block indices: a list for one frame, nested lists otherwise."""
        if self.detected.ndim == 1:
            return np.flatnonzero(self.detected).tolist()
        return [DetectionResult._rows(row) for row in self.detected]

    @staticmethod
    def _rows(mask: np.ndarray) -> Any:
        if mask.ndim == 1:
            return np.flatnonzero(mask).tolist()
        return [DetectionResult._rows(row) for row in mask]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "detector": self.detector,
            "threshold": self.threshold,
            "scores": self.scores.tolist(),
            "detected_blocks": self.detected_blocks,
            "status": self.status.tolist(),
            "explanation": self.explanation,
        }


class Detector:
    """Base class for block-level detectors.

    Subclasses set ``name`` (registry key), ``rule`` (one-line description),
    ``default_threshold`` and optionally ``temporal``/``requires_fit``, and implement
    :meth:`_score_frames` (frame detectors) or :meth:`_score_streams` (temporal ones).
    Scores must be finite and non-negative; higher means more anomalous.
    """

    name: ClassVar[str] = ""
    rule: ClassVar[str] = ""
    default_threshold: ClassVar[float] = 0.5
    temporal: ClassVar[bool] = False
    requires_fit: ClassVar[bool] = False

    def __init__(self, config: Config | None = None, threshold: float | None = None) -> None:
        self.config = config if config is not None else Config()
        if not isinstance(self.config, Config):
            raise TypeError("config must be a Config instance.")
        self.threshold = validate_threshold(self.default_threshold if threshold is None else threshold)
        self.fitted = not self.requires_fit

    # -- public API ---------------------------------------------------------

    def fit(self, telemetry: Any) -> Detector:
        """Learn any baseline from clean telemetry; returns ``self``.

        Frame detectors take ``(n, channels)`` frames; temporal detectors take one
        ``(steps, channels)`` stream or ``(n, steps, channels)`` streams.
        """
        self._fit(self._as_array(telemetry, for_fit=True))
        self.fitted = True
        return self

    def score(self, telemetry: Any) -> np.ndarray:
        """Per-block scores with the input's leading shape (see :class:`DetectionResult`)."""
        if not self.fitted:
            raise RuntimeError(f"Detector {self.name!r} must be fitted before scoring; call fit() first.")
        array = self._as_array(telemetry)
        if self.temporal:
            batch = array if array.ndim == 3 else array[None]
            scores = self._score_streams(batch)
            scores = scores if array.ndim == 3 else scores[0]
        else:
            flat = array.reshape(-1, self.config.channels)
            scores = self._score_frames(flat).reshape(*array.shape[:-1], self.config.blocks)
        scores = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError(f"Detector {self.name!r} produced non-finite or negative scores.")
        return scores

    def detect(self, telemetry: Any, threshold: float | None = None) -> DetectionResult:
        """Score, threshold and classify telemetry."""
        threshold = self.threshold if threshold is None else validate_threshold(threshold)
        scores = self.score(telemetry)
        detected = scores >= threshold
        status = classify_counts(detected.sum(axis=-1), self.config)
        return DetectionResult(
            detector=self.name,
            threshold=threshold,
            scores=scores,
            detected=detected,
            status=np.asarray(status),
            explanation=self.explain(scores, detected, threshold),
        )

    def params(self) -> dict[str, Any]:
        """Detector parameters (JSON-ready); override to add your own."""
        return {}

    def explain(self, scores: np.ndarray, detected: np.ndarray, threshold: float) -> dict[str, Any]:
        """Human-readable explanation of a result; subclasses may extend it."""
        last = scores.reshape(-1, scores.shape[-1])[-1]
        top = int(np.argmax(last))
        return {
            "detector": self.name,
            "rule": self.rule,
            "threshold": threshold,
            "params": self.params(),
            "temporal": self.temporal,
            "detected_count": int(detected.sum()),
            "top_block_last_frame": top,
            "top_score_last_frame": float(last[top]),
        }

    # -- hooks ----------------------------------------------------------------

    def _fit(self, telemetry: np.ndarray) -> None:
        """Override to learn from clean telemetry (default: nothing to learn)."""

    def _score_frames(self, frames: np.ndarray) -> np.ndarray:
        """``(n, channels) -> (n, blocks)``; required for frame detectors."""
        raise NotImplementedError

    def _score_streams(self, streams: np.ndarray) -> np.ndarray:
        """``(n, steps, channels) -> (n, steps, blocks)``; required for temporal detectors."""
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------

    def _as_array(self, telemetry: Any, for_fit: bool = False) -> np.ndarray:
        array = np.asarray(telemetry, dtype=np.float64)
        allowed = (2, 3) if self.temporal else ((2,) if for_fit else (1, 2, 3))
        if array.ndim not in allowed or array.shape[-1] != self.config.channels:
            raise ValueError(
                f"{self.name}: expected {allowed}-D telemetry with {self.config.channels} channels "
                f"in the last axis; received shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError(f"{self.name}: telemetry is empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{self.name}: telemetry contains NaN or infinite values.")
        return array

    def _blocks(self, frames: np.ndarray) -> np.ndarray:
        """Reshape ``(..., channels)`` to ``(..., blocks, block_size)``."""
        return frames.reshape(*frames.shape[:-1], self.config.blocks, self.config.block_size)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Detector]] = {}
D = TypeVar("D", bound=type[Detector])


def register_detector(cls: D | None = None, *, replace: bool = False) -> Any:
    """Register a :class:`Detector` subclass under ``cls.name`` (usable as a decorator)."""

    def decorator(klass: D) -> D:
        if not (isinstance(klass, type) and issubclass(klass, Detector)):
            raise TypeError("Only Detector subclasses can be registered.")
        if not isinstance(klass.name, str) or not _NAME_PATTERN.match(klass.name):
            raise ValueError(f"Invalid detector name {klass.name!r}; use lowercase letters, digits, '_' or '-'.")
        if klass.name in _REGISTRY and _REGISTRY[klass.name] is not klass and not replace:
            raise ValueError(f"A detector named {klass.name!r} is already registered.")
        _REGISTRY[klass.name] = klass
        return klass

    return decorator if cls is None else decorator(cls)


def unregister_detector(name: str) -> None:
    """Remove a detector from the registry (mainly for tests and plugins)."""
    _REGISTRY.pop(name, None)


def available_detectors() -> list[str]:
    """Sorted names of registered detectors."""
    return sorted(_REGISTRY)


def get_detector_class(name: str) -> type[Detector]:
    """Look up a registered detector class by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown detector {name!r}; available: {', '.join(available_detectors())}.") from None


def create_detector(name: str, config: Config | None = None, **kwargs: Any) -> Detector:
    """Instantiate a registered detector."""
    return get_detector_class(name)(config=config, **kwargs)


def load_plugins(entry_points: Callable[..., Any] | None = None) -> list[str]:
    """Register detectors advertised under the ``oes_resilience.detectors`` entry-point group.

    A plugin package declares, in its ``pyproject.toml``::

        [project.entry-points."oes_resilience.detectors"]
        mydetector = "my_package:MyDetector"

    Broken plugins are skipped with a warning. Returns the names that were registered.
    """
    if entry_points is None:
        from importlib.metadata import entry_points as _entry_points

        entry_points = _entry_points
    loaded = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            register_detector(entry.load())
            loaded.append(entry.name)
        except Exception as exc:  # noqa: BLE001 - a plugin must not break the benchmark
            warnings.warn(f"Skipping detector plugin {entry.name!r}: {exc}", RuntimeWarning, stacklevel=2)
    return loaded


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------


@register_detector
class OES32Detector(Detector):
    """The v0.1 reference detector (identical scores to :func:`core.score_signals`)."""

    name = "oes32"
    rule = "score = w_max*max|x| + w_rms*RMS(x) + w_mean*mean|x| per 32-channel block; detected if score >= threshold"
    default_threshold = 0.50

    def _score_frames(self, frames: np.ndarray) -> np.ndarray:
        return score_signals(frames, self.config)

    def params(self) -> dict[str, Any]:
        w: ScoreWeights = self.config.score_weights
        return {"weights": {"maximum": w.maximum, "rms": w.rms, "mean_absolute": w.mean_absolute}}

    def explain(self, scores: np.ndarray, detected: np.ndarray, threshold: float) -> dict[str, Any]:
        result = super().explain(scores, detected, threshold)
        result["components"] = ["max|x|", "RMS", "mean|x|"]
        return result


def _block_statistic(blocks: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "rms":
        return np.sqrt(np.mean(blocks * blocks, axis=-1))
    if statistic == "mean":
        return blocks.mean(axis=-1)
    return np.abs(blocks).mean(axis=-1)  # "mean_abs"


@register_detector
class RobustZScoreDetector(Detector):
    """``|stat - median| / (1.4826 * MAD)`` per block, baseline fitted on clean frames."""

    name = "zscore"
    rule = "score = |s - median_b| / (1.4826 * MAD_b), s = block statistic, baseline per block from clean frames"
    default_threshold = 3.5
    requires_fit = True
    statistics: ClassVar[tuple[str, ...]] = ("rms", "mean", "mean_abs")

    def __init__(self, config: Config | None = None, threshold: float | None = None, statistic: str = "rms") -> None:
        super().__init__(config, threshold)
        if statistic not in self.statistics:
            raise ValueError(f"statistic must be one of {self.statistics}; received {statistic!r}.")
        self.statistic = statistic
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _fit(self, frames: np.ndarray) -> None:
        stats = _block_statistic(self._blocks(frames), self.statistic)
        self.median = np.median(stats, axis=0)
        mad = np.median(np.abs(stats - self.median), axis=0)
        self.scale = np.maximum(1.4826 * mad, MIN_SCALE)

    def _score_frames(self, frames: np.ndarray) -> np.ndarray:
        stats = _block_statistic(self._blocks(frames), self.statistic)
        return np.abs(stats - self.median) / self.scale

    def params(self) -> dict[str, Any]:
        result: dict[str, Any] = {"statistic": self.statistic}
        if self.median is not None:
            result["median_mean_over_blocks"] = float(np.mean(self.median))
            result["scale_mean_over_blocks"] = float(np.mean(self.scale))
        return result


class TemporalDetector(Detector):
    """Base for detectors that standardise each block's mean against the warm-up steps.

    For each stream, block means ``m[t, b]`` over the first ``warmup`` steps give a
    per-block mean ``mu_b`` and a pooled standard deviation ``sigma`` (all blocks);
    ``z[t, b] = (m[t, b] - mu_b) / sigma``. Scores during warm-up are 0.
    """

    temporal = True

    def __init__(self, config: Config | None = None, threshold: float | None = None, warmup: int = 16) -> None:
        super().__init__(config, threshold)
        self.warmup = _require_int("warmup", warmup, 2)

    def standardize(self, streams: np.ndarray) -> np.ndarray:
        """``(n, steps, channels) -> (n, steps, blocks)`` standardised block means."""
        steps = streams.shape[1]
        if steps <= self.warmup:
            raise ValueError(f"{self.name}: streams need more than warmup={self.warmup} steps; got {steps}.")
        means = self._blocks(streams).mean(axis=-1)
        base = means[:, : self.warmup]
        mu = base.mean(axis=1)
        dof = self.warmup * self.config.blocks - self.config.blocks
        sigma = np.sqrt(((base - mu[:, None]) ** 2).sum(axis=(1, 2)) / dof)
        z = (means - mu[:, None]) / np.maximum(sigma, MIN_SCALE)[:, None, None]
        z[:, : self.warmup] = 0.0
        return z

    def params(self) -> dict[str, Any]:
        return {"warmup": self.warmup, "statistic": "block mean, standardised on warm-up"}


@register_detector
class EWMADetector(TemporalDetector):
    """Two-sided EWMA of the standardised block mean."""

    name = "ewma"
    rule = "e_t = lam*z_t + (1-lam)*e_(t-1) from the end of warm-up; score = |e_t| / sqrt(lam/(2-lam))"
    default_threshold = 3.0

    def __init__(
        self, config: Config | None = None, threshold: float | None = None, warmup: int = 16, lam: float = 0.2
    ) -> None:
        super().__init__(config, threshold, warmup)
        self.lam = _require_finite("lam", lam)
        if not 0.0 < self.lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1]; received {lam!r}.")

    def _score_streams(self, streams: np.ndarray) -> np.ndarray:
        z = self.standardize(streams)
        scores = np.zeros_like(z)
        state = np.zeros((z.shape[0], z.shape[2]))
        norm = np.sqrt(self.lam / (2.0 - self.lam))
        for t in range(self.warmup, z.shape[1]):
            state = self.lam * z[:, t] + (1.0 - self.lam) * state
            scores[:, t] = np.abs(state) / norm
        return scores

    def params(self) -> dict[str, Any]:
        return {**super().params(), "lam": self.lam}


@register_detector
class CUSUMDetector(TemporalDetector):
    """Two-sided tabular CUSUM of the standardised block mean."""

    name = "cusum"
    rule = "S+ = max(0, S+ + z - k), S- = max(0, S- - z - k) from the end of warm-up; score = max(S+, S-)"
    default_threshold = 5.0

    def __init__(
        self, config: Config | None = None, threshold: float | None = None, warmup: int = 16, k: float = 0.5
    ) -> None:
        super().__init__(config, threshold, warmup)
        self.k = _require_finite("k", k, 0.0)

    def _score_streams(self, streams: np.ndarray) -> np.ndarray:
        z = self.standardize(streams)
        scores = np.zeros_like(z)
        upper = np.zeros((z.shape[0], z.shape[2]))
        lower = np.zeros_like(upper)
        for t in range(self.warmup, z.shape[1]):
            upper = np.maximum(0.0, upper + z[:, t] - self.k)
            lower = np.maximum(0.0, lower - z[:, t] - self.k)
            scores[:, t] = np.maximum(upper, lower)
        return scores

    def params(self) -> dict[str, Any]:
        return {**super().params(), "k": self.k}


def sklearn_available() -> bool:
    """Whether the optional scikit-learn dependency can be imported."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


@register_detector
class IsolationForestDetector(Detector):
    """Isolation Forest on per-block features (optional; requires scikit-learn).

    Features per block: mean, standard deviation, max|x| and RMS. One forest is fitted
    on all blocks of the clean fit frames; score = ``-score_samples`` (in (0, 1]).
    """

    name = "iforest"
    rule = "score = -IsolationForest.score_samples([mean, std, max|x|, RMS]) per block; forest fitted on clean frames"
    default_threshold = 0.6
    requires_fit = True

    def __init__(
        self, config: Config | None = None, threshold: float | None = None, n_estimators: int = 100,
        max_samples: int = 256, random_state: int | None = None,
    ) -> None:
        super().__init__(config, threshold)
        self.n_estimators = _require_int("n_estimators", n_estimators, 1)
        self.max_samples = _require_int("max_samples", max_samples, 2)
        self.random_state = self.config.seed if random_state is None else _require_int("random_state", random_state, 0)
        self.model: Any = None

    def _features(self, frames: np.ndarray) -> np.ndarray:
        blocks = self._blocks(frames)
        features = np.stack(
            [blocks.mean(-1), blocks.std(-1), np.abs(blocks).max(-1), np.sqrt((blocks * blocks).mean(-1))], axis=-1
        )
        return features.reshape(-1, 4)

    def _fit(self, frames: np.ndarray) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise ImportError(
                "The 'iforest' detector needs scikit-learn: pip install 'oes-resilience[iforest]'."
            ) from exc
        self.model = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples, random_state=self.random_state
        ).fit(self._features(frames))

    def _score_frames(self, frames: np.ndarray) -> np.ndarray:
        scores = -self.model.score_samples(self._features(frames))
        return scores.reshape(frames.shape[0], self.config.blocks)

    def params(self) -> dict[str, Any]:
        return {"n_estimators": self.n_estimators, "max_samples": self.max_samples, "random_state": self.random_state}
