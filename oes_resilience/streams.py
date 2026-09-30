# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Optional multi-step synthetic streams (T steps x channels) for temporal detectors.

The single-frame v0.1 regimes in :mod:`oes_resilience.core` are unchanged; streams are
an additional track. Every stream starts with ``warmup`` event-free steps. In event
regimes the anomaly starts at a recorded ``onset`` step drawn uniformly from
``[onset_min, onset_max]`` and persists to the end of the stream.

Regimes (per-channel values; the burst/shock parameters are the v0.1 ones):

* ``stream_stable``: N(0, 0.05) at every step.
* ``stream_noisy``:  N(0.25, 0.10) at every step.
* ``stream_burst``:  stable, plus N(0.80, 0.15) on one random block from ``onset``.
* ``stream_shift``:  stable, plus a constant +0.02 on one random block from ``onset``.
  This small persistent shift (0.4 standard deviations per channel) is included on
  purpose as a case that single-frame detectors are not designed to catch.
* ``stream_shock``:  stable until ``onset``, then N(2.0, 0.5) on every channel.

Synthetic data only; no physical or real-world claim is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .core import (
    BASE_NOISE,
    BURST_ADDITION,
    NOISY_NOISE,
    SHOCK_NOISE,
    Config,
    Expectation,
    _require_int,
)

#: Seed purpose keys for streams (disjoint from the frame streams in ``core``).
PURPOSE_STREAM_EVAL = 200
PURPOSE_STREAM_CALIBRATION = 201
PURPOSE_STREAM_FIT = 202

#: Constant added to the shifted block in ``stream_shift``.
SHIFT_MAGNITUDE = 0.02


class StreamRegime(str, Enum):
    """Multi-step synthetic regimes."""

    STABLE = "stream_stable"
    NOISY = "stream_noisy"
    BURST = "stream_burst"
    SHIFT = "stream_shift"
    SHOCK = "stream_shock"


STREAM_REGIME_KEYS: Mapping[StreamRegime, int] = {
    StreamRegime.STABLE: 0,
    StreamRegime.NOISY: 1,
    StreamRegime.BURST: 2,
    StreamRegime.SHOCK: 3,
    StreamRegime.SHIFT: 4,
}

STREAM_EXPECTATION: Mapping[StreamRegime, Expectation] = {
    StreamRegime.STABLE: Expectation.NONE,
    StreamRegime.NOISY: Expectation.NONE,
    StreamRegime.BURST: Expectation.EVENT_BLOCKS,
    StreamRegime.SHIFT: Expectation.EVENT_BLOCKS,
    StreamRegime.SHOCK: Expectation.ALL_BLOCKS,
}

CLEAN_STREAM_REGIMES: tuple[StreamRegime, ...] = (StreamRegime.STABLE, StreamRegime.NOISY)


@dataclass(frozen=True)
class StreamConfig:
    """Stream length, warm-up and onset window.

    ``onset_min``/``onset_max`` default to ``warmup + m`` and ``steps - 2m`` with
    ``m = (steps - warmup) // 6`` (64 steps, warm-up 16 -> onset in [24, 48]).
    """

    steps: int = 64
    warmup: int = 16
    onset_min: int | None = None
    onset_max: int | None = None

    def __post_init__(self) -> None:
        steps = _require_int("steps", self.steps, 3)
        warmup = _require_int("warmup", self.warmup, 2)
        if warmup >= steps - 1:
            raise ValueError(f"warmup ({warmup}) must be at most steps - 2 ({steps - 2}).")
        margin = (steps - warmup) // 6
        onset_min = warmup + margin if self.onset_min is None else _require_int("onset_min", self.onset_min, 0)
        onset_max = steps - 2 * margin if self.onset_max is None else _require_int("onset_max", self.onset_max, 0)
        onset_max = min(onset_max, steps - 1)
        if not warmup <= onset_min <= onset_max < steps:
            raise ValueError(
                f"Need warmup <= onset_min <= onset_max < steps; got {warmup}, {onset_min}, {onset_max}, {steps}."
            )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "warmup", warmup)
        object.__setattr__(self, "onset_min", onset_min)
        object.__setattr__(self, "onset_max", onset_max)

    def to_dict(self) -> dict[str, int]:
        """JSON-ready representation."""
        return {"steps": self.steps, "warmup": self.warmup, "onset_min": self.onset_min, "onset_max": self.onset_max}


def stream_rng(seed: int, regime: StreamRegime | str, index: int, purpose: int) -> np.random.Generator:
    """Generator for one stream, keyed by ``(purpose, regime_key, index)``."""
    regime = StreamRegime(regime)
    key = (_require_int("purpose", purpose, 0), STREAM_REGIME_KEYS[regime], _require_int("index", index, 0))
    return np.random.default_rng(np.random.SeedSequence(entropy=_require_int("seed", seed, 0), spawn_key=key))


def generate_stream(
    regime: StreamRegime | str, rng: np.random.Generator, config: Config, stream: StreamConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw one stream. Returns ``(x, truth, onset)``.

    ``x`` has shape ``(steps, channels)``; ``truth`` is the boolean ``(blocks,)`` mask
    of blocks expected to activate after onset; ``onset`` is ``-1`` for clean regimes.
    """
    regime = StreamRegime(regime)
    shape = (stream.steps, config.channels)
    truth = np.zeros(config.blocks, dtype=bool)
    if regime is StreamRegime.NOISY:
        return rng.normal(*NOISY_NOISE, shape), truth, -1
    x = rng.normal(*BASE_NOISE, shape)
    if regime is StreamRegime.STABLE:
        return x, truth, -1
    if regime is StreamRegime.SHOCK:
        onset = int(rng.integers(stream.onset_min, stream.onset_max + 1))
        x[onset:] = rng.normal(*SHOCK_NOISE, (stream.steps - onset, config.channels))
        truth[:] = True
        return x, truth, onset
    block = int(rng.integers(0, config.blocks))
    onset = int(rng.integers(stream.onset_min, stream.onset_max + 1))
    columns = slice(block * config.block_size, (block + 1) * config.block_size)
    if regime is StreamRegime.BURST:
        x[onset:, columns] += rng.normal(*BURST_ADDITION, (stream.steps - onset, config.block_size))
    else:  # StreamRegime.SHIFT
        x[onset:, columns] += SHIFT_MAGNITUDE
    truth[block] = True
    return x, truth, onset


def generate_streams(
    regime: StreamRegime | str,
    indices: Sequence[int] | range,
    config: Config,
    stream: StreamConfig,
    purpose: int = PURPOSE_STREAM_EVAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate streams for the given indices: ``(x (n,T,C), truth (n,B), onsets (n,))``."""
    n = len(indices)
    x = np.empty((n, stream.steps, config.channels))
    truth = np.zeros((n, config.blocks), dtype=bool)
    onsets = np.full(n, -1, dtype=np.int64)
    for row, index in enumerate(indices):
        x[row], truth[row], onsets[row] = generate_stream(
            regime, stream_rng(config.seed, regime, index, purpose), config, stream
        )
    return x, truth, onsets
