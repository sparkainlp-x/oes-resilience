# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Stress suite: detector robustness under seeded synthetic perturbations.

Every scenario is generated from its own keyed seed stream (``(300, track, scenario,
index)``, disjoint from the v0.1 trials and the compare streams) and carries exact
ground truth: which blocks hold the event and when it starts. Two tracks:

* **frame**: one 512-channel frame per trial (event present from the start, onset 0).
  Only frame (non-temporal) detectors apply.
* **stream**: multi-step streams with warm-up and a random onset, as in ``compare``.

Thresholds are *not* re-tuned for the stress: each detector keeps the threshold
calibrated on clean stable + noisy data exactly as in ``compare`` (same fit and
calibration seeds), so the suite measures how a detector deployed at ~1% FP behaves
when conditions change.

Robustness is reported relative to each detector's own clean baseline on the same
track: background scenarios against ``clean`` (FP inflation = FP - clean FP) and
event scenarios against ``clean_burst`` (recall retention = recall / clean recall).
``drift`` and ``baseline_step`` have no clean counterpart and are reported in
absolute terms only. ``change`` is ``worse``/``better`` only when the 95% Wilson
intervals of the stressed and reference rates do not overlap (a conservative test),
otherwise ``n.s.``.

Missing data (``dropout*``) is NaN in the telemetry. Detectors that do not declare
``supports_missing`` are reported as unsupported rather than fed zero-filled data.

Synthetic data only; no physical or real-world claim is made.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import core
from ._version import __version__
from .adaptive_threshold import POTThreshold
from .core import (
    BASE_NOISE,
    BURST_ADDITION,
    PROJECT,
    Config,
    _require_finite,
    _require_int,
    _round,
    block_range,
    json_bytes,
    wilson_interval,
)
from .detectors import Detector, load_plugins
from .scorecard import (
    STREAM_CHUNK,
    CompareConfig,
    _cell,
    _name_list,
    _Timer,
    fit_detectors,
    frame_calibration,
    stream_calibration,
    stream_outcomes,
    write_compare_outputs,
)
from .streams import StreamConfig

#: Seed purpose key for the stress suite.
PURPOSE_STRESS = 300
TRACK_KEYS: Mapping[str, int] = {"frame": 0, "stream": 1}
DEFAULT_STRESS_DETECTORS: tuple[str, ...] = (
    "oes32", "oes32-robust", "zscore", "ewma", "cusum", "cusum-cp", "oes32+ewma"
)


@dataclass(frozen=True)
class StressParams:
    """Perturbation parameters (all synthetic, in the benchmark's own units)."""

    t_df: float = 3.0
    burst_mean: float = BURST_ADDITION[0]
    dropout_fraction: float = 0.10
    saturated_fraction: float = 0.02
    clip_level: float = 0.6
    narrow_width: int = 8
    drift_slope: float = 0.001
    step_size: float = 0.10
    correlated_blocks: int = 3
    correlated_amplitude: tuple[float, float] = (0.6, 0.1)
    global_offset: float = 0.25

    def __post_init__(self) -> None:
        if _require_finite("t_df", self.t_df) <= 0:
            raise ValueError(f"t_df must be > 0; received {self.t_df!r}.")
        for name in ("dropout_fraction", "saturated_fraction"):
            value = _require_finite(name, getattr(self, name), 0.0)
            if value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1); received {value!r}.")
        if _require_finite("clip_level", self.clip_level) <= 0:
            raise ValueError(f"clip_level must be > 0; received {self.clip_level!r}.")
        _require_int("narrow_width", self.narrow_width, 1)
        _require_int("correlated_blocks", self.correlated_blocks, 1)
        for name in ("burst_mean", "drift_slope", "step_size", "global_offset"):
            _require_finite(name, getattr(self, name))
        mean, std = (float(v) for v in self.correlated_amplitude)
        _require_finite("correlated amplitude mean", mean)
        _require_finite("correlated amplitude std", std, 0.0)
        object.__setattr__(self, "correlated_amplitude", (mean, std))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "t_df": self.t_df, "burst_mean": self.burst_mean, "dropout_fraction": self.dropout_fraction,
            "saturated_fraction": self.saturated_fraction, "clip_level": self.clip_level,
            "narrow_width": self.narrow_width, "drift_slope": self.drift_slope, "step_size": self.step_size,
            "correlated_blocks": self.correlated_blocks, "correlated_amplitude": list(self.correlated_amplitude),
            "global_offset": self.global_offset,
        }

    def t_scale(self) -> float:
        """Student-t scale: matches the clean std (0.05) when the variance exists (df > 2)."""
        sigma = BASE_NOISE[1]
        return sigma * math.sqrt((self.t_df - 2.0) / self.t_df) if self.t_df > 2 else sigma


@dataclass(frozen=True)
class Scenario:
    """One stress scenario. ``event`` is None for background-only scenarios."""

    name: str
    key: int
    description: str
    event: str | None = None
    background: str = "gaussian"
    dropout: bool = False
    saturated: bool = False
    clip: bool = False
    offset: bool = False
    tracks: tuple[str, ...] = ("frame", "stream")
    reference: str | None = None

    @property
    def kind(self) -> str:
        """``background`` (FP metrics) or ``event`` (recall metrics)."""
        return "background" if self.event is None else "event"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("clean", 0, "N(0, 0.05) background (FP reference)"),
    Scenario("clean_burst", 1, "clean + N(burst_mean, 0.15) on one block from onset (recall reference)",
             event="burst"),
    Scenario("heavy_tail", 2, "Student-t background (df = t_df, same std as clean when df > 2)",
             background="student_t", reference="clean"),
    Scenario("heavy_tail_burst", 3, "heavy-tailed background + burst", event="burst", background="student_t",
             reference="clean_burst"),
    Scenario("dropout", 4, "dropout_fraction of channels missing (NaN) for the whole stream", dropout=True,
             reference="clean"),
    Scenario("dropout_burst", 5, "missing channels + burst", event="burst", dropout=True, reference="clean_burst"),
    Scenario("saturated", 6, "saturated_fraction of channels stuck at +clip_level", saturated=True,
             reference="clean"),
    Scenario("clipped_burst", 7, "burst with every channel clipped to [-clip_level, +clip_level]", event="burst",
             clip=True, reference="clean_burst"),
    Scenario("narrow_burst", 8, "burst on narrow_width channels inside one block", event="narrow",
             reference="clean_burst"),
    Scenario("cross_block_burst", 9, "block-wide burst starting mid-block, so it spans two blocks",
             event="cross", reference="clean_burst"),
    Scenario("global_shift", 10, "all channels offset by +global_offset", offset=True, reference="clean"),
    Scenario("global_shift_burst", 11, "global offset + burst (truth = the burst block)", event="burst",
             offset=True, reference="clean_burst"),
    Scenario("correlated_burst", 12, "common per-step amplitude on correlated_blocks adjacent blocks",
             event="correlated", reference="clean_burst"),
    Scenario("drift", 13, "linear drift of drift_slope per step on one block from onset", event="drift",
             tracks=("stream",)),
    Scenario("baseline_step", 14, "abrupt +step_size on every channel from onset (truth = all blocks)",
             event="step", tracks=("stream",)),
    Scenario("regime_steps", 15, "four clean baseline regimes with alternating step offsets",
             background="regime_steps", tracks=("stream",), reference="clean"),
)
SCENARIO_BY_NAME: Mapping[str, Scenario] = {s.name: s for s in SCENARIOS}
SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIO_BY_NAME)


def get_scenario(name: str | Scenario) -> Scenario:
    """Look up a scenario by name."""
    if isinstance(name, Scenario):
        return name
    try:
        return SCENARIO_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown stress scenario {name!r}; available: {', '.join(SCENARIO_NAMES)}.") from None


def stress_rng(seed: int, track: str, scenario: str | Scenario, index: int) -> np.random.Generator:
    """Generator for one stress trial, keyed by ``(300, track, scenario, index)``."""
    key = (PURPOSE_STRESS, TRACK_KEYS[track], get_scenario(scenario).key, _require_int("index", index, 0))
    return np.random.default_rng(np.random.SeedSequence(entropy=_require_int("seed", seed, 0), spawn_key=key))


def generate_stress(
    scenario: str | Scenario,
    rng: np.random.Generator,
    config: Config,
    steps: int,
    onset_range: tuple[int, int],
    params: StressParams,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw one stressed sample ``(x (steps, channels), truth (blocks,), onset)``.

    ``onset`` is -1 for background scenarios. The frame track uses ``steps = 1`` and
    ``onset_range = (0, 0)``. Draw order: background, channel faults, event.
    """
    s = get_scenario(scenario)
    channels, bs, blocks = config.channels, config.block_size, config.blocks
    shape = (steps, channels)
    if s.background == "regime_steps":
        x = rng.normal(*BASE_NOISE, shape)
        levels = (0.0, 0.20, -0.15, 0.10)
        for indices, level in zip(np.array_split(np.arange(steps), len(levels)), levels, strict=True):
            x[indices] += level
    elif s.background == "student_t":
        x = rng.standard_t(params.t_df, shape) * params.t_scale()
    else:
        x = rng.normal(*BASE_NOISE, shape)
    if s.offset:
        x += params.global_offset
    truth = np.zeros(blocks, dtype=bool)
    onset = -1
    if s.event is not None:
        onset = int(rng.integers(onset_range[0], onset_range[1] + 1))
        length = steps - onset
        if s.event == "step":
            x[onset:] += params.step_size
            truth[:] = True
        elif s.event == "correlated":
            k = params.correlated_blocks
            if k > blocks:
                raise ValueError(f"correlated_blocks ({k}) exceeds the number of blocks ({blocks}).")
            first = int(rng.integers(0, blocks - k + 1))
            x[onset:, first * bs:(first + k) * bs] += rng.normal(*params.correlated_amplitude, (length, 1))
            truth[first:first + k] = True
        else:
            if s.event == "cross":
                if blocks < 2 or bs < 2:
                    raise ValueError("cross_block_burst needs at least 2 blocks of at least 2 channels.")
                start = int(rng.integers(0, blocks - 1)) * bs + bs // 2
                width = bs
            elif s.event == "narrow":
                width = params.narrow_width
                if width > bs:
                    raise ValueError(f"narrow_width ({width}) exceeds block_size ({bs}).")
                start = int(rng.integers(0, blocks)) * bs + int(rng.integers(0, bs - width + 1))
            else:
                start, width = int(rng.integers(0, blocks)) * bs, bs
            columns = slice(start, start + width)
            if s.event == "drift":
                x[onset:, columns] += params.drift_slope * np.arange(1, length + 1)[:, None]
            else:
                x[onset:, columns] += rng.normal(params.burst_mean, BURST_ADDITION[1], (length, width))
            truth[list(block_range(start, start + width, bs))] = True
    if s.saturated:
        count = int(round(params.saturated_fraction * channels))
        x[:, rng.choice(channels, size=count, replace=False)] = params.clip_level
    if s.clip:
        np.clip(x, -params.clip_level, params.clip_level, out=x)
    if s.dropout:
        count = int(round(params.dropout_fraction * channels))
        x[:, rng.choice(channels, size=count, replace=False)] = np.nan
    return x, truth, onset


def generate_stress_batch(
    scenario: str | Scenario, track: str, indices: Sequence[int] | range, config: Config, stream: StreamConfig,
    params: StressParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch ``(x (n, steps, C), truth (n, B), onsets (n,))``; the frame track has ``steps = 1``."""
    if track not in TRACK_KEYS:
        raise ValueError(f"Unknown track {track!r}.")
    steps, onset_range = (1, (0, 0)) if track == "frame" else (stream.steps, (stream.onset_min, stream.onset_max))
    n = len(indices)
    x = np.empty((n, steps, config.channels))
    truth = np.zeros((n, config.blocks), dtype=bool)
    onsets = np.full(n, -1, dtype=np.int64)
    for row, index in enumerate(indices):
        x[row], truth[row], onsets[row] = generate_stress(
            scenario, stress_rng(config.seed, track, scenario, index), config, steps, onset_range, params
        )
    return x, truth, onsets


@dataclass(frozen=True)
class StressConfig:
    """Settings for :func:`run_stress`. Sizes, detectors and tracks come from ``compare``."""

    compare: CompareConfig = field(default_factory=lambda: CompareConfig(detectors=DEFAULT_STRESS_DETECTORS))
    params: StressParams = field(default_factory=StressParams)
    scenarios: tuple[str, ...] = SCENARIO_NAMES

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios or len(set(scenarios)) != len(scenarios):
            raise ValueError("scenarios must be a non-empty list without duplicates.")
        for name in scenarios:
            get_scenario(name)
        # References are always evaluated so that robustness can be computed.
        needed = {SCENARIO_BY_NAME[n].reference for n in scenarios} - {None} - set(scenarios)
        ordered = tuple(n for n in SCENARIO_NAMES if n in set(scenarios) | needed)
        object.__setattr__(self, "scenarios", ordered)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {"compare": self.compare.to_dict(), "params": self.params.to_dict(), "scenarios": list(self.scenarios)}


STRESS_FIELDS: tuple[str, ...] = (
    "track", "detector", "scenario", "kind", "reference", "supported", "threshold", "n",
    "fp_rate", "fp_ci95_low", "fp_ci95_high", "ref_fp_rate", "fp_inflation",
    "early_alarm_rate", "recall", "recall_ci95_low", "recall_ci95_high", "ref_recall", "recall_retention",
    "exact_match_rate", "mean_iou", "latency_median", "ref_latency_median", "latency_delta", "change",
)


def _rate(successes: int, n: int) -> tuple[float, float, float]:
    low, high = wilson_interval(successes, n)
    return _round(successes / n), _round(low), _round(high)


def _metrics(outcomes: Mapping[str, np.ndarray], kind: str) -> dict[str, Any]:
    n = int(outcomes["any_alarm"].size)
    if kind == "background":
        fp, low, high = _rate(int(outcomes["any_alarm"].sum()), n)
        return {"n": n, "fp_rate": fp, "fp_ci95_low": low, "fp_ci95_high": high}
    recall, low, high = _rate(int(outcomes["hit"].sum()), n)
    latency = outcomes["latency"][outcomes["hit"]]
    return {
        "n": n, "recall": recall, "recall_ci95_low": low, "recall_ci95_high": high,
        "early_alarm_rate": _round(outcomes["pre_alarm"].mean()),
        "exact_match_rate": _round(outcomes["exact"].mean()), "mean_iou": _round(outcomes["iou"].mean()),
        "latency_median": _round(float(np.median(latency))) if latency.size else None,
    }


def robustness(row: Mapping[str, Any], ref: Mapping[str, Any] | None) -> dict[str, Any]:
    """Robustness fields of ``row`` relative to its clean reference row (or none)."""
    out: dict[str, Any] = {"change": None}
    if ref is None or not row.get("supported") or not ref.get("supported"):
        return out
    if row["kind"] == "background":
        out["ref_fp_rate"] = ref["fp_rate"]
        out["fp_inflation"] = _round(row["fp_rate"] - ref["fp_rate"])
        worse, better = row["fp_ci95_low"] > ref["fp_ci95_high"], row["fp_ci95_high"] < ref["fp_ci95_low"]
    else:
        out["ref_recall"] = ref["recall"]
        out["recall_retention"] = _round(row["recall"] / ref["recall"]) if ref["recall"] else None
        out["ref_latency_median"] = ref["latency_median"]
        if row["latency_median"] is not None and ref["latency_median"] is not None:
            out["latency_delta"] = _round(row["latency_median"] - ref["latency_median"])
        worse = row["recall_ci95_high"] < ref["recall_ci95_low"]
        better = row["recall_ci95_low"] > ref["recall_ci95_high"]
    out["change"] = "worse" if worse else "better" if better else "n.s."
    return out


def _stress_chunks(scenario: str, track: str, count: int, sc: StressConfig):
    cc = sc.compare
    for start in range(0, count, STREAM_CHUNK):
        yield generate_stress_batch(scenario, track, range(start, min(start + STREAM_CHUNK, count)), cc.config,
                                    cc.stream, sc.params)


def _evaluate_track(
    track: str, sc: StressConfig, detectors: Mapping[str, Detector], thresholds: Mapping[str, float], timer: _Timer
) -> list[dict[str, Any]]:
    cc = sc.compare
    count, warmup = (cc.trials, 0) if track == "frame" else (cc.streams, cc.stream.warmup)
    rows: list[dict[str, Any]] = []
    for name in sc.scenarios:
        scenario = SCENARIO_BY_NAME[name]
        if track not in scenario.tracks:
            continue
        active = {d: detectors[d] for d in thresholds if not (scenario.dropout and not detectors[d].supports_missing)}
        outcomes: dict[str, list[dict[str, np.ndarray]]] = {d: [] for d in active}
        for x, truth, onsets in _stress_chunks(name, track, count, sc):
            for d, detector in active.items():
                data = x[:, 0] if track == "frame" else x
                scores = timer.score((track, d), detector, data, x.shape[0] * x.shape[1])
                if track == "frame":
                    scores = scores[:, None, :]
                outcomes[d].append(stream_outcomes(scores, truth, onsets, thresholds[d], warmup))
        for d in thresholds:
            row = dict.fromkeys(STRESS_FIELDS)
            row.update(track=track, detector=d, scenario=name, kind=scenario.kind, reference=scenario.reference,
                       threshold=thresholds[d], supported=d in active)
            if d in active:
                merged = {k: np.concatenate([o[k] for o in outcomes[d]]) for k in outcomes[d][0]}
                row.update(_metrics(merged, scenario.kind))
            rows.append(row)
    index = {(r["track"], r["detector"], r["scenario"]): r for r in rows}
    for row in rows:
        ref = index.get((row["track"], row["detector"], row["reference"]))
        row.update(robustness(row, ref))
    return rows


def _adaptive_threshold_convergence(sc: StressConfig, detector: Detector) -> dict[str, Any] | None:
    """Measure POT recovery around the known boundaries of the seeded regime-step trace.

    The change-point detector and POT threshold run online without oracle resets. Ground
    truth is used only to define evaluation windows. The held-out clean reference uses
    post-warmup windows matching the regime-segment length. A transition is recovered
    once three consecutive non-anomalous scores leave the threshold within 10% of that
    reference threshold, before the next known transition.
    """
    cc = sc.compare
    if "regime_steps" not in sc.scenarios or "stream" not in cc.tracks:
        return None

    segments = [part for part in np.array_split(np.arange(cc.stream.steps), 4) if part.size]
    boundaries = [int(part[0]) for part in segments[1:] if int(part[0]) >= cc.stream.warmup]
    if not boundaries:
        return {
            "detector": "cusum-cp",
            "status": "no_regime_transition_after_warmup",
            "transition_count": 0,
            "converged_transitions": 0,
            "convergence_rate": None,
            "median_convergence_steps": None,
            "p90_convergence_steps": None,
            "reference_threshold": None,
            "calibration_window_steps": None,
        }

    calibration_window = min(len(part) for part in segments[1:] if int(part[0]) >= cc.stream.warmup)
    clean_scores = []
    clean_count = max(cc.calibration_streams, 20)
    for x, _, _ in _stress_chunks("clean", "stream", clean_count, sc):
        clean_scores.append(
            detector.score(x).max(axis=2)[:, cc.stream.warmup:cc.stream.warmup + calibration_window].ravel()
        )
    calibration = np.concatenate(clean_scores)
    risk_level = min(cc.target_fp, 0.05)
    base = POTThreshold(
        risk_level=risk_level, init_quantile=0.90, window_size=500, min_exceedances=8
    ).fit_calibration(calibration)
    reference_threshold = float(base.threshold)

    latencies: list[int] = []
    transition_count = 0
    tolerance = 0.10 * max(abs(reference_threshold), 1e-12)
    for x, _, _ in _stress_chunks("regime_steps", "stream", cc.streams, sc):
        scores = detector.score(x).max(axis=2)
        for stream_scores in scores:
            pot = deepcopy(base)
            flagged = np.zeros(cc.stream.steps, dtype=bool)
            thresholds = np.full(cc.stream.steps, reference_threshold, dtype=float)
            for step in range(cc.stream.warmup, cc.stream.steps):
                flagged[step], _ = pot.update(float(stream_scores[step]))
                thresholds[step] = float(pot.threshold)

            for idx, boundary in enumerate(boundaries):
                end = boundaries[idx + 1] if idx + 1 < len(boundaries) else cc.stream.steps
                transition_count += 1
                consecutive = 0
                for step in range(boundary, end):
                    within_band = abs(thresholds[step] - reference_threshold) <= tolerance
                    consecutive = consecutive + 1 if within_band and not flagged[step] else 0
                    if consecutive == 3:
                        latencies.append(step - boundary + 1)
                        break

    return {
        "detector": "cusum-cp",
        "status": "measured",
        "scenario": "regime_steps",
        "risk_level": risk_level,
        "calibration_window_steps": calibration_window,
        "reference_threshold": _round(reference_threshold),
        "tolerance_fraction": 0.10,
        "required_consecutive_clean_updates": 3,
        "transition_count": transition_count,
        "converged_transitions": len(latencies),
        "convergence_rate": _round(len(latencies) / transition_count) if transition_count else None,
        "median_convergence_steps": _round(float(np.median(latencies))) if latencies else None,
        "p90_convergence_steps": _round(float(np.percentile(latencies, 90))) if latencies else None,
        "evaluation_policy": "ground-truth boundaries define windows only; no POT oracle resets",
    }


def run_stress(sc: StressConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the stress suite. Returns ``(scorecard, timing)``; only the scorecard is deterministic."""
    cc = sc.compare
    timer = _Timer()
    detectors = fit_detectors(cc)
    rows: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for track, calibrate in (("frame", frame_calibration), ("stream", stream_calibration)):
        if track not in cc.tracks:
            continue
        thresholds, records = calibrate(cc, detectors)
        calibrations += records
        rows += _evaluate_track(track, sc, detectors, thresholds, timer)
    config = cc.config.to_dict()
    config["threshold"] = None
    adaptive_threshold = _adaptive_threshold_convergence(sc, detectors["cusum-cp"]) if "cusum-cp" in detectors else None
    scorecard = {
        "project": PROJECT,
        "version": __version__,
        "kind": "stress_scorecard",
        "config": config,
        "stress": sc.to_dict(),
        "scenarios": {
            name: {"description": SCENARIO_BY_NAME[name].description, "kind": SCENARIO_BY_NAME[name].kind,
                   "tracks": list(SCENARIO_BY_NAME[name].tracks), "reference": SCENARIO_BY_NAME[name].reference}
            for name in sc.scenarios
        },
        "method": (
            "thresholds calibrated on clean stable+noisy data exactly as in compare (target FP per clean regime), "
            "not re-tuned per scenario; background rows: FP = share of units with any alarm (after warm-up), "
            "fp_inflation = FP - clean FP; event rows: recall = share with an alarm on a true block at/after onset, "
            "recall_retention = recall / clean_burst recall; 95% Wilson intervals; change = worse/better only if "
            "the stressed and reference intervals do not overlap"
        ),
        "detectors": {
            name: {"class": type(d).__name__, "rule": d.rule, "temporal": d.temporal,
                   "supports_missing": d.supports_missing, "params": d.params()}
            for name, d in detectors.items()
        },
        "calibration": calibrations,
        "adaptive_threshold": adaptive_threshold,
        "rows": rows,
    }
    return scorecard, timer.report()


def stress_markdown(scorecard: Mapping[str, Any], timing: Sequence[Mapping[str, Any]]) -> str:
    """Markdown report (includes machine-dependent CPU timings, so it is not hashed)."""
    cc = scorecard["stress"]["compare"]
    lines = [
        f"# {scorecard['project']} stress scorecard (v{scorecard['version']})",
        "",
        "Synthetic data only; these numbers describe software behaviour on the benchmark's own perturbations, "
        "not real-world performance.",
        "",
        f"Seed {scorecard['config']['seed']}; frame track {cc['trials']} trials/scenario; stream track "
        f"{cc['streams']} streams/scenario of {cc['stream']['steps']} steps (warm-up {cc['stream']['warmup']}); "
        f"target FP {cc['target_fp']:.3f}. Parameters: "
        + ", ".join(f"{k}={v}" for k, v in scorecard["stress"]["params"].items()) + ".",
        "",
        "Method: " + scorecard["method"] + ".",
        "",
        "## Scenarios",
        "",
        "| scenario | kind | tracks | reference | description |",
        "|---|---|---|---|---|",
    ]
    for name, s in scorecard["scenarios"].items():
        lines.append(f"| {name} | {s['kind']} | {', '.join(s['tracks'])} | {s['reference'] or '–'} | "
                     f"{s['description']} |")
    adaptive = scorecard.get("adaptive_threshold")
    if adaptive is not None:
        convergence_rate = adaptive["convergence_rate"]
        median_steps = adaptive["median_convergence_steps"]
        p90_steps = adaptive["p90_convergence_steps"]
        lines += [
            "",
            "## Online POT threshold convergence",
            "",
            "Ground-truth regime boundaries define measurement windows only; "
            "the POT state is not reset at transitions. Clean calibration windows "
            "match post-reset segment length. Recovery requires three "
            "consecutive unflagged scores with the threshold within 10% of the "
            "held-out clean reference threshold.",
            "",
            f"Converged {adaptive['converged_transitions']} / {adaptive['transition_count']} transitions "
            f"({convergence_rate if convergence_rate is not None else 'n/a'}); median "
            f"{median_steps if median_steps is not None else 'n/a'} steps, "
            f"p90 {p90_steps if p90_steps is not None else 'n/a'} steps. "
            f"Detector: `{adaptive['detector']}`; POT risk level {adaptive.get('risk_level', 'n/a')}; "
            f"reference threshold {adaptive['reference_threshold']}.",
        ]
    for track in ("frame", "stream"):
        rows = [r for r in scorecard["rows"] if r["track"] == track]
        if not rows:
            continue
        lines += ["", f"## {track.capitalize()} track: background scenarios (false positives)", "",
                  "| detector | scenario | FP [95% CI] | clean FP | FP inflation | change |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            if r["kind"] != "background":
                continue
            if not r["supported"]:
                lines.append(f"| {r['detector']} | {r['scenario']} | unsupported (missing data) | – | – | – |")
                continue
            lines.append(f"| {r['detector']} | {r['scenario']} | {r['fp_rate']:.3f} [{r['fp_ci95_low']:.3f}, "
                         f"{r['fp_ci95_high']:.3f}] | {_cell(r['ref_fp_rate'])} | {_cell(r['fp_inflation'])} | "
                         f"{r['change'] or '–'} |")
        lines += ["", f"## {track.capitalize()} track: event scenarios (recall)", "",
                  "| detector | scenario | recall [95% CI] | clean recall | retention | exact | early alarm | "
                  "median latency (Δ) | change |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["kind"] != "event":
                continue
            if not r["supported"]:
                lines.append(f"| {r['detector']} | {r['scenario']} | unsupported (missing data) | – | – | – | – | – "
                             "| – |")
                continue
            latency = _cell(r["latency_median"], 1)
            if r["latency_delta"] is not None:
                latency += f" ({r['latency_delta']:+.1f})"
            lines.append(f"| {r['detector']} | {r['scenario']} | {r['recall']:.3f} [{r['recall_ci95_low']:.3f}, "
                         f"{r['recall_ci95_high']:.3f}] | {_cell(r['ref_recall'])} | {_cell(r['recall_retention'])} "
                         f"| {_cell(r['exact_match_rate'])} | {_cell(r['early_alarm_rate'])} | {latency} | "
                         f"{r['change'] or '–'} |")
    lines += ["", "## CPU time (machine-dependent)", "", "| track | detector | frames scored | CPU µs per frame |",
              "|---|---|---|---|"]
    for t in timing:
        lines.append(f"| {t['track']} | {t['detector']} | {t['frames']} | {t['cpu_us_per_frame']:.2f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_stress_parser(sub: Any) -> argparse.ArgumentParser:
    """Register the ``stress`` subcommand on an argparse subparsers object."""
    parser = sub.add_parser("stress", help="run the stress suite and write a robustness scorecard")
    core._add_common(parser, "oes_resilience_stress")
    d = StressParams()
    parser.add_argument("--streams", type=core._positive_int, default=1000, help="stress streams per scenario")
    parser.add_argument("--fit-trials", type=core._positive_int, default=1000, help="clean fit frames per regime")
    parser.add_argument("--fit-streams", type=core._positive_int, default=200, help="clean fit streams per regime")
    parser.add_argument("--calibration-trials", type=core._positive_int, default=1000,
                        help="held-out clean calibration frames per regime")
    parser.add_argument("--calibration-streams", type=core._positive_int, default=1000,
                        help="held-out clean calibration streams per regime")
    parser.add_argument("--steps", type=int, default=64, help="steps per stream (default 64)")
    parser.add_argument("--warmup", type=int, default=16, help="event-free warm-up steps (default 16)")
    parser.add_argument("--target-fp", type=float, default=0.01, help="calibration target FP rate (default 0.01)")
    parser.add_argument("--detectors", type=_name_list, default=DEFAULT_STRESS_DETECTORS,
                        help=f"comma-separated detector names (default {','.join(DEFAULT_STRESS_DETECTORS)})")
    parser.add_argument("--tracks", type=_name_list, default=("frame", "stream"), help="frame,stream (default both)")
    parser.add_argument("--scenarios", type=_name_list, default=SCENARIO_NAMES,
                        help="comma-separated scenarios (default all); references are added automatically")
    parser.add_argument("--t-df", type=float, default=d.t_df, help="Student-t degrees of freedom (default 3)")
    parser.add_argument("--burst-mean", type=float, default=d.burst_mean,
                        help="mean burst addition per channel (default 0.8, the v0.1 value)")
    parser.add_argument("--dropout-fraction", type=float, default=d.dropout_fraction,
                        help="fraction of channels missing in dropout scenarios (default 0.10)")
    parser.add_argument("--saturated-fraction", type=float, default=d.saturated_fraction,
                        help="fraction of channels stuck at +clip-level (default 0.02)")
    parser.add_argument("--clip-level", type=float, default=d.clip_level, help="clip/saturation level (default 0.6)")
    parser.add_argument("--narrow-width", type=int, default=d.narrow_width, help="narrow burst width (default 8)")
    parser.add_argument("--drift-slope", type=float, default=d.drift_slope, help="drift per step (default 0.001)")
    parser.add_argument("--step-size", type=float, default=d.step_size, help="baseline step size (default 0.10)")
    parser.add_argument("--correlated-blocks", type=int, default=d.correlated_blocks,
                        help="adjacent blocks in correlated_burst (default 3)")
    parser.add_argument("--correlated-mean", type=float, default=d.correlated_amplitude[0],
                        help="mean common amplitude in correlated_burst (default 0.6; std 0.1)")
    parser.add_argument("--global-offset", type=float, default=d.global_offset, help="global offset (default 0.25)")
    parser.add_argument("--plugins", action="store_true", help="load detectors from installed entry points")
    parser.add_argument("--verify", action="store_true",
                        help="recompute the stress scorecard and exit 4 unless its hash matches")
    return parser


def cmd_stress(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Handler for ``stress``."""
    if args.plugins:
        load_plugins()
    d = StressParams()
    sc = StressConfig(
        compare=CompareConfig(
            config=core._config_from_args(args), stream=StreamConfig(steps=args.steps, warmup=args.warmup),
            trials=args.trials, fit_trials=args.fit_trials, calibration_trials=args.calibration_trials,
            streams=args.streams, calibration_streams=args.calibration_streams, fit_streams=args.fit_streams,
            target_fp=args.target_fp, detectors=args.detectors, tracks=args.tracks,
        ),
        params=StressParams(
            t_df=args.t_df, burst_mean=args.burst_mean, dropout_fraction=args.dropout_fraction,
            saturated_fraction=args.saturated_fraction,
            clip_level=args.clip_level, narrow_width=args.narrow_width, drift_slope=args.drift_slope,
            step_size=args.step_size, correlated_blocks=args.correlated_blocks,
            correlated_amplitude=(args.correlated_mean, d.correlated_amplitude[1]), global_offset=args.global_offset,
        ),
        scenarios=args.scenarios,
    )
    started = time.perf_counter()
    scorecard, timing = run_stress(sc)
    paths = write_compare_outputs(scorecard, timing, args.out_dir, args.prefix,
                                  core.run_metadata(argv, time.perf_counter() - started),
                                  fields=STRESS_FIELDS, markdown=stress_markdown)
    code = core.EXIT_OK
    if args.verify:
        first = core.sha256_file(paths["scorecard_json"])
        second = core.hashlib.sha256(json_bytes(run_stress(sc)[0])).hexdigest()
        match = first == second
        print(f"reproducibility check: {'hash match' if match else 'HASH MISMATCH'} ({first[:16]}…)")
        code = core.EXIT_OK if match else core.EXIT_NOT_REPRODUCIBLE
    if not args.quiet:
        print(stress_markdown(scorecard, timing))
        for key, path in paths.items():
            print(f"{key}: {path}")
    return code
