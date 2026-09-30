# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""OES-Resilience core: the v0.1 single-frame benchmark harness and the CLI.

The v0.1 harness contains the benchmark and one transparent reference
detector, "OES32": synthetic 512-channel signals are generated under four
regimes, split into sixteen 32-channel blocks, and every block is scored with

    score = 0.45 * max|x| + 0.35 * RMS(x) + 0.20 * mean|x|

A block is flagged when ``score >= threshold`` (default 0.50).

Important:
    This implementation evaluates synthetic signals only. It is not a
    physical sensor, quantum processor, navigation (GPS) system, or medical
    device, and its results make no claim about any of those. The
    "expectation checks" below are software-behaviour checks on synthetic
    data, nothing more.

Reproducibility:
    Each trial draws from its own generator seeded with
    ``np.random.SeedSequence(seed, spawn_key=(regime_key, trial_index))``.
    This is equivalent to ``SeedSequence(seed).spawn(...)[regime_key]
    .spawn(...)[trial_index]``, so a trial's signal depends only on
    ``(seed, regime, trial_index)`` -- not on the number of trials, the order
    of regimes, or the threshold.

Outputs (per run):
    <prefix>_results.json   deterministic results (byte-for-byte reproducible)
    <prefix>_summary.csv    deterministic per-regime summary
    <prefix>_manifest.json  deterministic SHA-256 hashes of the two files above
    <prefix>_run_metadata.json  non-deterministic run info (time, platform, ...)

Only NumPy is required. ``python -m oes_resilience --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
import platform
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__

PROJECT = "OES-Resilience"

# Exit codes.
EXIT_OK = 0
EXIT_FAILURE = 1  # test failure or unexpected runtime error
EXIT_USAGE = 2  # invalid arguments / configuration (same code argparse uses)
EXIT_EXPECTATIONS = 3  # --strict and a documented expectation was not met
EXIT_NOT_REPRODUCIBLE = 4  # compare --verify: a rerun produced a different hash

#: Default location of the test suite (a source checkout: <repo>/tests).
DEFAULT_TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
#: Decimal places used when floats are written to the deterministic outputs.
FLOAT_DECIMALS = 12
#: Trials generated and scored per chunk (bounds memory to chunk * channels).
CHUNK_TRIALS = 4096


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------


class Regime(str, Enum):
    """Synthetic signal regimes."""

    STABLE = "stable"
    NOISY = "noisy"
    LOCALIZED_BURST = "localized_burst"
    GLOBAL_SHOCK = "global_shock"


#: Fixed per-regime keys for seed derivation. Explicit (not enum order) so
#: that adding or reordering regimes never changes an existing regime's seeds.
REGIME_KEYS: Mapping[Regime, int] = {
    Regime.STABLE: 0,
    Regime.NOISY: 1,
    Regime.LOCALIZED_BURST: 2,
    Regime.GLOBAL_SHOCK: 3,
}

#: Baseline distribution parameters (mean, standard deviation).
BASE_NOISE = (0.00, 0.05)
NOISY_NOISE = (0.25, 0.10)
BURST_ADDITION = (0.80, 0.15)
SHOCK_NOISE = (2.00, 0.50)


class Expectation(str, Enum):
    """Which blocks a regime is expected to activate."""

    NONE = "none"  # stable, noisy
    EVENT_BLOCKS = "event_blocks"  # localized_burst
    ALL_BLOCKS = "all_blocks"  # global_shock ("most or all" is acceptable)


REGIME_EXPECTATION: Mapping[Regime, Expectation] = {
    Regime.STABLE: Expectation.NONE,
    Regime.NOISY: Expectation.NONE,
    Regime.LOCALIZED_BURST: Expectation.EVENT_BLOCKS,
    Regime.GLOBAL_SHOCK: Expectation.ALL_BLOCKS,
}


class Status(str, Enum):
    """Per-signal triage status, derived from the number of detected blocks."""

    STABLE = "STABLE"
    LOCALIZED_ALERT = "LOCALIZED_ALERT"
    DISTRIBUTED_ALERT = "DISTRIBUTED_ALERT"
    GLOBAL_SHOCK = "GLOBAL_SHOCK"


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _require_int(name: str, value: Any, minimum: int) -> int:
    """Return ``value`` as ``int`` if it is an integer (not bool) >= minimum."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer; received {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}.")
    return int(value)


def _require_finite(name: str, value: Any, minimum: float | None = None) -> float:
    """Return ``value`` as ``float`` if it is a finite real (not bool)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number; received {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; received {value!r}.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value!r}.")
    return result


def validate_threshold(threshold: Any) -> float:
    """Validate a detection threshold: finite and non-negative."""
    return _require_finite("threshold", threshold, minimum=0.0)


def validate_trials(trials: Any) -> int:
    """Validate a trials-per-regime count: a positive integer."""
    return _require_int("trials_per_regime", trials, minimum=1)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreWeights:
    """Weights of the block score; finite, non-negative, summing to 1."""

    maximum: float = 0.45
    rms: float = 0.35
    mean_absolute: float = 0.20

    def __post_init__(self) -> None:
        for name in ("maximum", "rms", "mean_absolute"):
            object.__setattr__(
                self, name, _require_finite(f"weight {name}", getattr(self, name), 0.0)
            )
        total = self.maximum + self.rms + self.mean_absolute
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Score weights must sum to 1.0; they sum to {total!r}.")


@dataclass(frozen=True)
class Config:
    """Benchmark configuration. All fields are validated on construction.

    ``global_fraction`` is the fraction of blocks (rounded up) that must be
    detected for the ``GLOBAL_SHOCK`` status. The default 1.0 keeps the
    v0.1.0 behaviour (all blocks).
    """

    channels: int = 512
    block_size: int = 32
    threshold: float = 0.50
    seed: int = 42
    global_fraction: float = 1.0
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        channels = _require_int("channels", self.channels, 1)
        block_size = _require_int("block_size", self.block_size, 1)
        if channels % block_size != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by block_size ({block_size})."
            )
        seed = _require_int("seed", self.seed, 0)
        threshold = validate_threshold(self.threshold)
        fraction = _require_finite("global_fraction", self.global_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"global_fraction must be in (0, 1]; received {fraction!r}.")
        if not isinstance(self.score_weights, ScoreWeights):
            raise TypeError("score_weights must be a ScoreWeights instance.")
        for name, value in (
            ("channels", channels),
            ("block_size", block_size),
            ("seed", seed),
            ("threshold", threshold),
            ("global_fraction", fraction),
        ):
            object.__setattr__(self, name, value)

    @property
    def blocks(self) -> int:
        """Number of blocks per signal."""
        return self.channels // self.block_size

    @property
    def global_min_blocks(self) -> int:
        """Minimum detected-block count for the ``GLOBAL_SHOCK`` status."""
        return max(1, math.ceil(self.global_fraction * self.blocks - 1e-9))

    def with_threshold(self, threshold: float) -> Config:
        """Return a copy with a different threshold."""
        return Config(
            channels=self.channels,
            block_size=self.block_size,
            threshold=threshold,
            seed=self.seed,
            global_fraction=self.global_fraction,
            score_weights=self.score_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation (includes derived fields)."""
        result = asdict(self)
        result["blocks"] = self.blocks
        result["global_min_blocks"] = self.global_min_blocks
        return result


# --------------------------------------------------------------------------
# Events and block mapping
# --------------------------------------------------------------------------


def block_range(start: int, stop: int, block_size: int) -> range:
    """Indices of the blocks overlapped by channels ``[start, stop)``.

    This is the single source of truth for channel-to-block mapping.
    """
    block_size = _require_int("block_size", block_size, 1)
    if not 0 <= start < stop:
        raise ValueError(f"Invalid channel interval [{start}, {stop}).")
    return range(start // block_size, (stop - 1) // block_size + 1)


@dataclass(frozen=True)
class Event:
    """An injected anomaly occupying channels ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_int("event start", self.start, 0)
        _require_int("event stop", self.stop, 1)
        if self.stop <= self.start:
            raise ValueError("Event stop must be greater than start.")

    @property
    def width(self) -> int:
        """Number of channels in the event."""
        return self.stop - self.start

    def blocks(self, block_size: int) -> tuple[int, ...]:
        """Blocks overlapped by the event for the given block size."""
        return tuple(block_range(self.start, self.stop, block_size))


# --------------------------------------------------------------------------
# Signal generation
# --------------------------------------------------------------------------


#: Purpose keys for seed streams that must never overlap the v0.1 evaluation
#: trials (which use the 2-tuple spawn key ``(regime_key, trial_index)``).
PURPOSE_FIT = 100
PURPOSE_CALIBRATION = 101


def trial_seed_sequence(
    seed: int, regime: Regime | str, trial_index: int, purpose: int | None = None
) -> np.random.SeedSequence:
    """Keyed seed for one trial; stable regardless of trial count or order.

    ``purpose=None`` gives the v0.1 evaluation stream ``(regime_key, i)``;
    any other purpose key gives the disjoint stream ``(purpose, regime_key, i)``
    (used for detector fitting and threshold calibration).
    """
    regime = Regime(regime)
    key: tuple[int, ...] = (REGIME_KEYS[regime], _require_int("trial_index", trial_index, 0))
    if purpose is not None:
        key = (_require_int("purpose", purpose, 0), *key)
    return np.random.SeedSequence(entropy=_require_int("seed", seed, 0), spawn_key=key)


def trial_rng(
    seed: int, regime: Regime | str, trial_index: int, purpose: int | None = None
) -> np.random.Generator:
    """Independent generator for one ``(seed, regime, trial_index[, purpose])``."""
    return np.random.default_rng(trial_seed_sequence(seed, regime, trial_index, purpose))


def generate_signal(
    regime: Regime | str, rng: np.random.Generator, config: Config
) -> tuple[np.ndarray, Event | None]:
    """Draw one synthetic signal (and its event, for ``localized_burst``).

    Draw order matches v0.1.0, so the same generator state yields the same
    signal as the original prototype.
    """
    regime = Regime(regime)
    n = config.channels
    if regime is Regime.STABLE:
        return rng.normal(*BASE_NOISE, n), None
    if regime is Regime.NOISY:
        return rng.normal(*NOISY_NOISE, n), None
    if regime is Regime.GLOBAL_SHOCK:
        return rng.normal(*SHOCK_NOISE, n), None
    # Regime.LOCALIZED_BURST (Regime() already rejected unknown values).
    signal = rng.normal(*BASE_NOISE, n)
    block = int(rng.integers(0, config.blocks))
    start = block * config.block_size
    stop = start + config.block_size
    signal[start:stop] += rng.normal(*BURST_ADDITION, config.block_size)
    return signal, Event(start, stop)


def generate_batch(
    regime: Regime | str, trial_indices: Sequence[int] | range, config: Config, purpose: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate signals for the given trial indices.

    Returns ``(signals, truth)`` where ``signals`` has shape
    ``(n, channels)`` and ``truth`` is a boolean ``(n, blocks)`` mask of the
    blocks each trial is expected to activate. ``purpose`` selects the seed
    stream (see :func:`trial_seed_sequence`).
    """
    regime = Regime(regime)
    n = len(trial_indices)
    signals = np.empty((n, config.channels), dtype=np.float64)
    truth = np.zeros((n, config.blocks), dtype=bool)
    expectation = REGIME_EXPECTATION[regime]
    if expectation is Expectation.ALL_BLOCKS:
        truth[:] = True
    for row, trial_index in enumerate(trial_indices):
        rng = trial_rng(config.seed, regime, trial_index, purpose)
        signal, event = generate_signal(regime, rng, config)
        signals[row] = signal
        if event is not None:
            truth[row, list(event.blocks(config.block_size))] = True
    return signals, truth


# --------------------------------------------------------------------------
# Scoring and detection
# --------------------------------------------------------------------------


def validate_signal(signal: Any, config: Config) -> np.ndarray:
    """Return a finite 1-D float64 array with ``config.channels`` entries."""
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if array.shape[0] != config.channels:
        raise ValueError(f"Expected {config.channels} channels; received {array.shape[0]}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Signal contains NaN or infinite values.")
    return array


def score_signals(signals: Any, config: Config) -> np.ndarray:
    """Vectorised block scores.

    ``signals`` may be ``(channels,)`` or ``(n, channels)``; the result is
    ``(blocks,)`` or ``(n, blocks)`` respectively.
    """
    array = np.asarray(signals, dtype=np.float64)
    if array.ndim == 1:
        return score_signals(validate_signal(array, config)[None, :], config)[0]
    if array.ndim != 2 or array.shape[1] != config.channels:
        raise ValueError(f"Expected shape (n, {config.channels}); received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Signals contain NaN or infinite values.")
    blocks = array.reshape(array.shape[0], config.blocks, config.block_size)
    magnitude = np.abs(blocks)
    maximum = magnitude.max(axis=2)
    rms = np.sqrt(np.mean(blocks * blocks, axis=2))
    mean_absolute = magnitude.mean(axis=2)
    w = config.score_weights
    return w.maximum * maximum + w.rms * rms + w.mean_absolute * mean_absolute


def detect(scores: Any, threshold: float) -> np.ndarray:
    """Boolean detection mask ``scores >= threshold`` (any shape)."""
    threshold = validate_threshold(threshold)
    array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("scores contain NaN or infinite values.")
    return array >= threshold


def classify_counts(counts: Any, config: Config) -> np.ndarray:
    """Map detected-block counts to :class:`Status` names (vectorised).

    Rules, in priority order: 0 -> STABLE; 1 -> LOCALIZED_ALERT;
    >= ``config.global_min_blocks`` -> GLOBAL_SHOCK; otherwise
    DISTRIBUTED_ALERT.
    """
    counts = np.asarray(counts)
    return np.select(
        [counts == 0, counts == 1, counts >= config.global_min_blocks],
        [Status.STABLE.value, Status.LOCALIZED_ALERT.value, Status.GLOBAL_SHOCK.value],
        default=Status.DISTRIBUTED_ALERT.value,
    )


def classify_status(detected: Iterable[int], config: Config) -> str:
    """Status for one list of detected block indices."""
    return str(classify_counts(len(list(detected)), config))


@dataclass(frozen=True)
class SignalAssessment:
    """Triage of a single (external or synthetic) signal."""

    scores: tuple[float, ...]
    detected_blocks: tuple[int, ...]
    status: str


def assess_signal(signal: Any, config: Config) -> SignalAssessment:
    """Score, threshold and classify one signal."""
    scores = score_signals(signal, config)
    detected = tuple(int(i) for i in np.flatnonzero(detect(scores, config.threshold)))
    return SignalAssessment(
        scores=tuple(float(s) for s in scores),
        detected_blocks=detected,
        status=classify_status(detected, config),
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Per-row distance (in blocks) from each block to the nearest True block.

    ``inf`` where the row has no True entry. O(n * blocks) memory.
    """
    n, width = mask.shape
    forward = np.empty((n, width))
    backward = np.empty((n, width))
    last = np.full(n, -np.inf)
    for b in range(width):
        last = np.where(mask[:, b], b, last)
        forward[:, b] = b - last
    nxt = np.full(n, np.inf)
    for b in range(width - 1, -1, -1):
        nxt = np.where(mask[:, b], b, nxt)
        backward[:, b] = nxt - b
    return np.minimum(forward, backward)


@dataclass
class BatchEvaluation:
    """Per-trial metric arrays for one regime at one threshold."""

    regime: Regime
    threshold: float
    trial_indices: np.ndarray  # (n,)
    scores: np.ndarray  # (n, blocks)
    truth: np.ndarray  # (n, blocks) bool
    detected: np.ndarray  # (n, blocks) bool
    counts: np.ndarray  # (n,)
    exact_match: np.ndarray  # (n,) bool
    overlap: np.ndarray  # (n,) IoU; 1.0 when both sets are empty
    event_detected: np.ndarray  # (n,) bool: any true block detected
    extra_blocks: np.ndarray  # (n,) int: detected blocks outside the truth
    localization_error: np.ndarray  # (n,) float, nan when undefined
    status: np.ndarray  # (n,) str


def evaluate_batch(
    regime: Regime | str,
    trial_indices: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    config: Config,
    threshold: float | None = None,
) -> BatchEvaluation:
    """Compute per-trial metrics from precomputed scores and truth masks."""
    regime = Regime(regime)
    threshold = config.threshold if threshold is None else validate_threshold(threshold)
    detected = detect(scores, threshold)
    truth = np.asarray(truth, dtype=bool)
    if detected.shape != truth.shape:
        raise ValueError("scores and truth must have the same shape.")
    hits = (detected & truth).sum(axis=1)
    union = (detected | truth).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        overlap = np.where(union == 0, 1.0, hits / np.maximum(union, 1))
    error = np.where(detected, _distance_to_mask(truth), np.inf).min(axis=1)
    error = np.where(np.isfinite(error), error, np.nan)
    counts = detected.sum(axis=1)
    return BatchEvaluation(
        regime=regime,
        threshold=threshold,
        trial_indices=np.asarray(trial_indices),
        scores=scores,
        truth=truth,
        detected=detected,
        counts=counts,
        exact_match=(detected == truth).all(axis=1),
        overlap=overlap,
        event_detected=hits > 0,
        extra_blocks=(detected & ~truth).sum(axis=1),
        localization_error=error,
        status=classify_counts(counts, config),
    )


def _round(value: float | None) -> float | None:
    """Round for deterministic serialisation; ``None`` passes through."""
    if value is None:
        return None
    return round(float(value), FLOAT_DECIMALS)


def _mean_or_none(values: np.ndarray) -> float | None:
    """Mean of a possibly empty array, ``None`` (not NaN) when empty."""
    values = np.asarray(values, dtype=np.float64)
    return None if values.size == 0 else _round(values.mean())


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return low, high


SUMMARY_FIELDS: tuple[str, ...] = (
    "regime",
    "threshold",
    "trials",
    "expectation",
    "detection_rate",
    "detection_rate_ci95_low",
    "detection_rate_ci95_high",
    "false_positive_rate",
    "event_detection_rate",
    "exact_match_rate",
    "mean_overlap",
    "mean_localization_error",
    "localization_error_trials",
    "extra_block_rate",
    "mean_detected_blocks",
    "mean_block_coverage",
    "majority_blocks_rate",
    "all_blocks_rate",
    "zero_detection_rate",
    "mean_maximum_score",
    "min_maximum_score",
    "max_maximum_score",
    "status_stable",
    "status_localized_alert",
    "status_distributed_alert",
    "status_global_shock",
)


def summarize(evaluation: BatchEvaluation) -> dict[str, Any]:
    """Per-regime summary; metrics that do not apply are ``None``.

    * ``detection_rate``: trials with at least one detected block.
    * ``false_positive_rate``: same, but only for regimes expecting none.
    * ``event_detection_rate`` / ``mean_localization_error``: burst only.
    * ``exact_match_rate``: detected set equals the expected set (empty for
      stable/noisy, the event block for bursts, all blocks for shocks).
    * ``mean_overlap``: IoU of detected vs expected blocks (burst, shock).
    """
    e = evaluation
    n = int(e.counts.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty batch.")
    blocks = e.detected.shape[1]
    expectation = REGIME_EXPECTATION[e.regime]
    any_detection = e.counts > 0
    detections = int(any_detection.sum())
    low, high = wilson_interval(detections, n)
    max_scores = e.scores.max(axis=1)
    status_counts = {status: int((e.status == status.value).sum()) for status in Status}
    is_none = expectation is Expectation.NONE
    is_event = expectation is Expectation.EVENT_BLOCKS
    finite_error = e.localization_error[~np.isnan(e.localization_error)]
    return {
        "regime": e.regime.value,
        "threshold": _round(e.threshold),
        "trials": n,
        "expectation": expectation.value,
        "detection_rate": _round(detections / n),
        "detection_rate_ci95_low": _round(low),
        "detection_rate_ci95_high": _round(high),
        "false_positive_rate": _round(detections / n) if is_none else None,
        "event_detection_rate": _mean_or_none(e.event_detected) if is_event else None,
        "exact_match_rate": _mean_or_none(e.exact_match),
        "mean_overlap": None if is_none else _mean_or_none(e.overlap),
        "mean_localization_error": _mean_or_none(finite_error) if is_event else None,
        "localization_error_trials": int(finite_error.size) if is_event else None,
        "extra_block_rate": _mean_or_none(e.extra_blocks > 0),
        "mean_detected_blocks": _mean_or_none(e.counts),
        "mean_block_coverage": _mean_or_none(e.counts / blocks),
        "majority_blocks_rate": _mean_or_none(e.counts * 2 > blocks),
        "all_blocks_rate": _mean_or_none(e.counts == blocks),
        "zero_detection_rate": _round(1 - detections / n),
        "mean_maximum_score": _mean_or_none(max_scores),
        "min_maximum_score": _round(max_scores.min()),
        "max_maximum_score": _round(max_scores.max()),
        "status_stable": status_counts[Status.STABLE],
        "status_localized_alert": status_counts[Status.LOCALIZED_ALERT],
        "status_distributed_alert": status_counts[Status.DISTRIBUTED_ALERT],
        "status_global_shock": status_counts[Status.GLOBAL_SHOCK],
    }


# --------------------------------------------------------------------------
# Documented expectations (software-behaviour checks only)
# --------------------------------------------------------------------------

#: Operationalisation of the documented reference behaviour. "Normally no
#: detections" is taken as a false-positive rate <= 1%.
MAX_NO_EVENT_FP_RATE = 0.01
MIN_BURST_DETECTION_RATE = 0.99
MIN_BURST_EXACT_MATCH_RATE = 0.95
MIN_SHOCK_MAJORITY_RATE = 0.99


def check_expectations(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare summaries with the documented reference behaviour."""
    checks: list[dict[str, Any]] = []

    def add(regime: str, name: str, value: float, op: str, bound: float) -> None:
        passed = value <= bound if op == "<=" else value >= bound
        checks.append(
            {"regime": regime, "check": name, "value": value, "op": op, "bound": bound, "passed": bool(passed)}
        )

    for s in summaries:
        regime = Regime(s["regime"])
        if REGIME_EXPECTATION[regime] is Expectation.NONE:
            add(s["regime"], "false_positive_rate", s["false_positive_rate"], "<=", MAX_NO_EVENT_FP_RATE)
        elif regime is Regime.LOCALIZED_BURST:
            add(s["regime"], "event_detection_rate", s["event_detection_rate"], ">=", MIN_BURST_DETECTION_RATE)
            add(s["regime"], "exact_match_rate", s["exact_match_rate"], ">=", MIN_BURST_EXACT_MATCH_RATE)
        else:
            add(s["regime"], "majority_blocks_rate", s["majority_blocks_rate"], ">=", MIN_SHOCK_MAJORITY_RATE)
    return checks


# --------------------------------------------------------------------------
# Benchmark and sweep
# --------------------------------------------------------------------------


def score_trials(
    regime: Regime | str, trial_indices: Sequence[int] | range, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and score specific trials; returns ``(scores, truth)``."""
    signals, truth = generate_batch(regime, trial_indices, config)
    return score_signals(signals, config), truth


def score_regime(regime: Regime | str, trials: int, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Score trials ``0..trials-1`` in chunks of :data:`CHUNK_TRIALS`."""
    trials = validate_trials(trials)
    scores = np.empty((trials, config.blocks))
    truth = np.empty((trials, config.blocks), dtype=bool)
    for start in range(0, trials, CHUNK_TRIALS):
        stop = min(start + CHUNK_TRIALS, trials)
        scores[start:stop], truth[start:stop] = score_trials(regime, range(start, stop), config)
    return scores, truth


def trial_records(e: BatchEvaluation, config: Config) -> list[dict[str, Any]]:
    """JSON-ready per-trial records."""
    expectation = REGIME_EXPECTATION[e.regime]
    is_none = expectation is Expectation.NONE
    is_event = expectation is Expectation.EVENT_BLOCKS
    records = []
    rounded = np.round(e.scores, FLOAT_DECIMALS)
    for row, trial_index in enumerate(e.trial_indices.tolist()):
        truth_blocks = np.flatnonzero(e.truth[row]).tolist()
        event = None
        if is_event:
            event = Event(truth_blocks[0] * config.block_size, (truth_blocks[-1] + 1) * config.block_size)
        error = e.localization_error[row]
        records.append(
            {
                "regime": e.regime.value,
                "trial_index": trial_index,
                "seed_key": [config.seed, REGIME_KEYS[e.regime], trial_index],
                "scores": rounded[row].tolist(),
                "detected_blocks": np.flatnonzero(e.detected[row]).tolist(),
                "expected_blocks": truth_blocks,
                "event_start": None if event is None else event.start,
                "event_stop": None if event is None else event.stop,
                "status": str(e.status[row]),
                "maximum_score": _round(e.scores[row].max()),
                "detected_count": int(e.counts[row]),
                "exact_match": bool(e.exact_match[row]),
                "overlap": None if is_none else _round(e.overlap[row]),
                "event_detected": bool(e.event_detected[row]) if is_event else None,
                "localization_error": None if (not is_event or np.isnan(error)) else int(error),
                "false_positive": bool(e.counts[row] > 0) if is_none else None,
            }
        )
    return records


def run_trial(regime: Regime | str, trial_index: int, config: Config) -> dict[str, Any]:
    """Run and evaluate a single trial (identical to its row in a benchmark)."""
    scores, truth = score_trials(regime, [trial_index], config)
    evaluation = evaluate_batch(regime, np.array([trial_index]), scores, truth, config)
    return trial_records(evaluation, config)[0]


def run_benchmark(config: Config, trials_per_regime: int = 1000, include_trials: bool = True) -> dict[str, Any]:
    """Run all regimes at ``config.threshold``. The result is deterministic."""
    trials_per_regime = validate_trials(trials_per_regime)
    summaries: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    indices = np.arange(trials_per_regime)
    for regime in Regime:
        scores, truth = score_regime(regime, trials_per_regime, config)
        evaluation = evaluate_batch(regime, indices, scores, truth, config)
        summaries.append(summarize(evaluation))
        if include_trials:
            trials.extend(trial_records(evaluation, config))
    result: dict[str, Any] = {
        "project": PROJECT,
        "version": __version__,
        "kind": "benchmark",
        "config": config.to_dict(),
        "trials_per_regime": trials_per_regime,
        "summaries": summaries,
        "expectation_checks": check_expectations(summaries),
    }
    if include_trials:
        result["trials"] = trials
    return result


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Inclusive, drift-free grid ``start, start+step, ..., stop``."""
    start = validate_threshold(start)
    stop = validate_threshold(stop)
    step = _require_finite("step", step)
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        raise ValueError("stop must be >= start.")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 10) for i in range(count)]


def threshold_sweep(thresholds: Iterable[float], trials_per_regime: int, config: Config) -> dict[str, Any]:
    """Evaluate many thresholds on one set of scores per regime.

    Signals are generated and scored once; each threshold only re-applies
    ``scores >= threshold``. Rows are identical to separate
    :func:`run_benchmark` summaries at each threshold.
    """
    values = [validate_threshold(t) for t in thresholds]
    if not values:
        raise ValueError("At least one threshold is required.")
    trials_per_regime = validate_trials(trials_per_regime)
    indices = np.arange(trials_per_regime)
    scored = {regime: score_regime(regime, trials_per_regime, config) for regime in Regime}
    rows: list[dict[str, Any]] = []
    passing: list[float] = []
    for threshold in values:
        summaries = [
            summarize(evaluate_batch(regime, indices, *scored[regime], config, threshold=threshold))
            for regime in Regime
        ]
        checks = check_expectations(summaries)
        if all(c["passed"] for c in checks):
            passing.append(threshold)
        rows.extend(summaries)
    base = config.to_dict()
    base["threshold"] = None
    return {
        "project": PROJECT,
        "version": __version__,
        "kind": "sweep",
        "config": base,
        "trials_per_regime": trials_per_regime,
        "thresholds": values,
        "thresholds_meeting_expectations": passing,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Deterministic, atomic output
# --------------------------------------------------------------------------


def json_bytes(data: Any) -> bytes:
    """Canonical JSON (sorted keys, no NaN/Infinity, trailing newline)."""
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    """CSV with an explicit header and ``\\n`` line endings; None -> empty."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    allowed = set(fieldnames)
    for row in rows:
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"Unexpected CSV fields: {sorted(unknown)}")
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _current_umask() -> int:
    """Process umask (read by setting and immediately restoring it)."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def atomic_write(path: Path, payload: bytes) -> str:
    """Write ``payload`` atomically (temp file + fsync + rename); return SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.chmod(tmp, 0o666 & ~_current_umask())  # mkstemp creates 0600 files
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    results: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    prefix: str,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write results, summary CSV, deterministic manifest and run metadata."""
    out_dir = Path(out_dir)
    paths = {
        "results": out_dir / f"{prefix}_results.json",
        "summary": out_dir / f"{prefix}_summary.csv",
        "manifest": out_dir / f"{prefix}_manifest.json",
        "metadata": out_dir / f"{prefix}_run_metadata.json",
    }
    results_hash = atomic_write(paths["results"], json_bytes(results))
    summary_hash = atomic_write(paths["summary"], csv_bytes(summary_rows, SUMMARY_FIELDS))
    manifest = {
        "project": PROJECT,
        "version": __version__,
        "files": {
            paths["results"].name: {"sha256": results_hash},
            paths["summary"].name: {"sha256": summary_hash},
        },
    }
    manifest_hash = atomic_write(paths["manifest"], json_bytes(manifest))
    atomic_write(paths["metadata"], json_bytes({**metadata, "manifest_sha256": manifest_hash}))
    return paths


def run_metadata(argv: Sequence[str], elapsed: float) -> dict[str, Any]:
    """Non-deterministic run information, kept out of the results file."""
    return {
        "project": PROJECT,
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "elapsed_seconds": elapsed,
        "argv": list(argv),
    }


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        return validate_trials(int(text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc


def _threshold_arg(text: str) -> float:
    try:
        return validate_threshold(float(text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative number, got {text!r}") from exc


def _threshold_list(text: str) -> list[float]:
    return [_threshold_arg(part) for part in text.split(",") if part.strip()]


def _add_common(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument("--trials", type=_positive_int, default=1000, help="trials per regime (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="base seed, >= 0 (default 42)")
    parser.add_argument("--channels", type=int, default=512, help="channels per signal (default 512)")
    parser.add_argument("--block-size", type=int, default=32, help="channels per block (default 32)")
    parser.add_argument(
        "--global-fraction", type=float, default=1.0,
        help="fraction of blocks needed for GLOBAL_SHOCK status (default 1.0 = all)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--prefix", default=prefix, help=f"output file prefix (default {prefix})")
    parser.add_argument("--quiet", action="store_true", help="suppress the console table")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with ``run``, ``sweep`` and ``test`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="oes-resilience",
        description=f"{PROJECT} v{__version__}: reproducible synthetic benchmark for multichannel "
        "telemetry anomaly detection (OES32 reference detector).",
        epilog="Exit codes: 0 ok, 1 test failure/runtime error, 2 invalid arguments, "
        "3 --strict and an expectation check failed, 4 compare --verify hash mismatch.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--test", action="store_true", help="alias for the 'test' subcommand")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the benchmark at one threshold")
    _add_common(run, "oes_resilience")
    run.add_argument("--threshold", type=_threshold_arg, default=0.50, help="detection threshold (default 0.50)")
    run.add_argument("--no-trials", action="store_true", help="omit per-trial records from the results file")
    run.add_argument("--strict", action="store_true", help="exit 3 if an expectation check fails")

    sweep = sub.add_parser("sweep", help="evaluate many thresholds on one set of scores")
    _add_common(sweep, "oes_resilience_sweep")
    sweep.add_argument("--start", type=_threshold_arg, default=0.30, help="first threshold (default 0.30)")
    sweep.add_argument("--stop", type=_threshold_arg, default=1.00, help="last threshold (default 1.00)")
    sweep.add_argument("--step", type=float, default=0.05, help="threshold step (default 0.05)")
    sweep.add_argument("--thresholds", type=_threshold_list, help="explicit comma-separated thresholds")

    from .scorecard import add_compare_parser  # lazy: scorecard imports this module

    add_compare_parser(sub)

    test = sub.add_parser("test", help="run the unit-test suite (unittest discovery)")
    test.add_argument(
        "--tests-dir", type=Path, default=DEFAULT_TESTS_DIR,
        help="directory containing test_*.py (default: ./tests next to this file)",
    )
    test.add_argument("-v", "--verbose", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace, threshold: float = 0.50) -> Config:
    return Config(
        channels=args.channels,
        block_size=args.block_size,
        threshold=threshold,
        seed=args.seed,
        global_fraction=args.global_fraction,
    )


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def format_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    """Plain-text table of the key per-regime metrics."""
    columns = ("regime", "threshold", "false_positive_rate", "detection_rate", "exact_match_rate",
               "mean_detected_blocks", "max_maximum_score")
    headers = ("regime", "thr", "FP rate", "detect", "exact", "mean blocks", "max score")
    table = [headers] + [tuple(_fmt(s[c]) for c in columns) for s in summaries]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in table)


def run_tests(tests_dir: Path, verbose: bool = False) -> int:
    """Discover and run ``test_*.py`` under ``tests_dir``; return an exit code."""
    import unittest

    tests_dir = Path(tests_dir)
    if not tests_dir.is_dir():
        print(f"error: tests directory not found: {tests_dir}", file=sys.stderr)
        return EXIT_USAGE
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(tests_dir))
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    return EXIT_OK if result.wasSuccessful() else EXIT_FAILURE


def _cmd_run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    config = _config_from_args(args, args.threshold)
    started = time.perf_counter()
    results = run_benchmark(config, args.trials, include_trials=not args.no_trials)
    paths = write_outputs(results, results["summaries"], args.out_dir, args.prefix,
                          run_metadata(argv, time.perf_counter() - started))
    failed = [c for c in results["expectation_checks"] if not c["passed"]]
    if not args.quiet:
        print(format_table(results["summaries"]))
        for check in results["expectation_checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"[{mark}] {check['regime']}: {check['check']} = {check['value']:.4f} {check['op']} {check['bound']}")
        for key, path in paths.items():
            print(f"{key}: {path}")
    return EXIT_EXPECTATIONS if (args.strict and failed) else EXIT_OK


def _cmd_sweep(args: argparse.Namespace, argv: Sequence[str]) -> int:
    config = _config_from_args(args)
    thresholds = args.thresholds if args.thresholds is not None else threshold_grid(args.start, args.stop, args.step)
    started = time.perf_counter()
    results = threshold_sweep(thresholds, args.trials, config)
    paths = write_outputs(results, results["rows"], args.out_dir, args.prefix,
                          run_metadata(argv, time.perf_counter() - started))
    if not args.quiet:
        print(format_table(results["rows"]))
        print(f"thresholds meeting all expectation checks: {results['thresholds_meeting_expectations']}")
        for key, path in paths.items():
            print(f"{key}: {path}")
    return EXIT_OK


COMMANDS = frozenset({"run", "sweep", "compare", "test"})
TOP_LEVEL_FLAGS = frozenset({"-h", "--help", "--version", "--test"})


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] not in COMMANDS | TOP_LEVEL_FLAGS:
        argv = ["run", *argv]  # v0.1.0 compatibility: no subcommand means "run"
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help/--version (0) or usage error (2)
        return int(exc.code or 0)
    if args.command is None:  # only reachable via the legacy --test flag
        return run_tests(DEFAULT_TESTS_DIR)
    try:
        if args.command == "test":
            return run_tests(args.tests_dir, args.verbose)
        if args.command == "run":
            return _cmd_run(args, argv)
        if args.command == "compare":
            from .scorecard import cmd_compare

            return cmd_compare(args, argv)
        return _cmd_sweep(args, argv)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
