"""
OES-512 Telemetry Triage
Version: 0.1.0

Synthetic block-level anomaly-detection benchmark.

Important:
    This implementation evaluates synthetic signals only. It is not a
    physical sensor, quantum processor, navigation system, or medical device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = "OES-512 Telemetry Triage"
VERSION = "0.1.0"


class Regime(str, Enum):
    STABLE = "stable"
    NOISY = "noisy"
    LOCALIZED_BURST = "localized_burst"
    GLOBAL_SHOCK = "global_shock"


@dataclass(frozen=True)
class ScoreWeights:
    maximum: float = 0.45
    rms: float = 0.35
    mean_absolute: float = 0.20

    def __post_init__(self) -> None:
        values = (self.maximum, self.rms, self.mean_absolute)
        if any(value < 0 for value in values):
            raise ValueError("Score weights cannot be negative.")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("Score weights must sum to 1.0.")


@dataclass(frozen=True)
class Config:
    channels: int = 512
    block_size: int = 32
    threshold: float = 0.50
    seed: int = 42
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.channels % self.block_size != 0:
            raise ValueError("channels must be divisible by block_size.")
        if self.threshold < 0:
            raise ValueError("threshold cannot be negative.")

    @property
    def blocks(self) -> int:
        return self.channels // self.block_size

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["score_weights"] = asdict(self.score_weights)
        result["blocks"] = self.blocks
        return result


@dataclass(frozen=True)
class Event:
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def blocks(self) -> tuple[int, ...]:
        return tuple(range(self.start // 32, (self.stop - 1) // 32 + 1))


@dataclass
class TrialResult:
    regime: str
    seed: int
    threshold: float
    scores: list[float]
    detected_blocks: list[int]
    event_start: int | None
    event_stop: int | None
    event_block: int | None
    status: str
    maximum_score: float
    detected_count: int
    event_detected: bool | None
    exact_match: bool | None
    overlap: float | None
    localization_error: int | None
    false_positive: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_signal(signal, config):
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1: raise ValueError("Signal must be one-dimensional.")
    if len(array) != config.channels: raise ValueError(f"Expected {config.channels} channels; received {len(array)}.")
    if not np.all(np.isfinite(array)): raise ValueError("Signal contains NaN or infinite values.")
    return array


def generate_signal(regime, rng, config):
    regime = Regime(regime)
    if regime is Regime.STABLE:
        return rng.normal(0.00, 0.05, config.channels), None
    if regime is Regime.NOISY:
        return rng.normal(0.25, 0.10, config.channels), None
    if regime is Regime.GLOBAL_SHOCK:
        return rng.normal(2.00, 0.50, config.channels), None
    if regime is Regime.LOCALIZED_BURST:
        signal = rng.normal(0.00, 0.05, config.channels)
        block = int(rng.integers(0, config.blocks))
        start = block * config.block_size
        stop = start + config.block_size
        signal[start:stop] += rng.normal(0.80, 0.15, config.block_size)
        return signal, Event(start, stop)
    raise ValueError(f"Unsupported regime: {regime}")


def split_blocks(signal, config):
    signal = validate_signal(signal, config)
    return signal.reshape(config.blocks, config.block_size)


def score_block(block, weights):
    block = np.asarray(block, dtype=np.float64)
    maximum = float(np.max(np.abs(block)))
    rms = float(np.sqrt(np.mean(block**2)))
    mean_absolute = float(np.mean(np.abs(block)))
    return weights.maximum*maximum + weights.rms*rms + weights.mean_absolute*mean_absolute


def score_signal(signal, config):
    blocks = split_blocks(signal, config)
    return np.array([score_block(b, config.score_weights) for b in blocks], dtype=np.float64)


def detected_blocks(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1: raise ValueError("scores must be one-dimensional.")
    if not np.all(np.isfinite(scores)): raise ValueError("scores contain NaN or infinite values.")
    return np.flatnonzero(scores >= threshold).astype(int).tolist()


def classify_status(detected, total_blocks):
    count = len(list(detected))
    if count == 0: return "STABLE"
    if count == 1: return "LOCALIZED_ALERT"
    if count < total_blocks: return "DISTRIBUTED_ALERT"
    return "GLOBAL_SHOCK"


def interval_iou(predicted_blocks, event, block_size):
    if event is None: return None
    true_blocks = set(range(event.start // block_size, (event.stop - 1) // block_size + 1))
    if not predicted_blocks and not true_blocks: return 1.0
    union = predicted_blocks | true_blocks
    if not union: return 0.0
    return len(predicted_blocks & true_blocks) / len(union)


def localization_error(predicted_blocks, event, block_size):
    if event is None or not predicted_blocks: return None
    true_block = event.start // block_size
    return min(abs(block - true_block) for block in predicted_blocks)


def evaluate_trial(regime, seed, signal, event, config):
    scores = score_signal(signal, config)
    detected = detected_blocks(scores, config.threshold)
    detected_set = set(detected)
    status = classify_status(detected, config.blocks)
    is_event_regime = Regime(regime) is Regime.LOCALIZED_BURST
    if is_event_regime:
        if event is None: raise ValueError("Localized burst must have an event.")
        true_blocks = set(range(event.start // config.block_size, (event.stop - 1) // config.block_size + 1))
        event_detected = bool(detected_set & true_blocks)
        exact_match = detected_set == true_blocks
        overlap = interval_iou(detected_set, event, config.block_size)
        error = localization_error(detected, event, config.block_size)
        false_positive = None
        event_block = event.start // config.block_size
    else:
        event_detected = None; exact_match = None; overlap = None; error = None
        false_positive = bool(detected); event_block = None
    return TrialResult(regime=Regime(regime).value, seed=seed, threshold=config.threshold, scores=scores.round(10).tolist(), detected_blocks=detected, event_start=None if event is None else event.start, event_stop=None if event is None else event.stop, event_block=event_block, status=status, maximum_score=float(np.max(scores)), detected_count=len(detected), event_detected=event_detected, exact_match=exact_match, overlap=overlap, localization_error=error, false_positive=false_positive)


def run_trial(regime, seed, config):
    rng = np.random.default_rng(seed)
    signal, event = generate_signal(regime, rng, config)
    return evaluate_trial(regime=regime, seed=seed, signal=signal, event=event, config=config)


def summarize_trials(trials):
    if not trials: raise ValueError("Cannot summarize an empty trial list.")
    regime = trials[0].regime
    counts = [t.detected_count for t in trials]
    maximum_scores = [t.maximum_score for t in trials]
    summary = {"regime": regime, "trials": len(trials), "mean_detected_blocks": float(np.mean(counts)), "mean_maximum_score": float(np.mean(maximum_scores)), "zero_detection_rate": float(np.mean([c == 0 for c in counts])), "event_detection_rate": None, "exact_match_rate": None, "false_positive_rate": None, "mean_overlap": None, "mean_localization_error": None}
    event_trials = [t for t in trials if t.event_detected is not None]
    no_event_trials = [t for t in trials if t.false_positive is not None]
    if event_trials:
        summary["event_detection_rate"] = float(np.mean([t.event_detected for t in event_trials]))
        summary["exact_match_rate"] = float(np.mean([t.exact_match for t in event_trials]))
        summary["mean_overlap"] = float(np.mean([t.overlap for t in event_trials]))
        summary["mean_localization_error"] = float(np.mean([t.localization_error for t in event_trials if t.localization_error is not None]))
    if no_event_trials:
        summary["false_positive_rate"] = float(np.mean([t.false_positive for t in no_event_trials]))
    return summary


def run_benchmark(config, trials_per_regime=1000):
    if trials_per_regime <= 0: raise ValueError("trials_per_regime must be positive.")
    started = time.perf_counter()
    trial_results = []; summaries = []
    regimes = list(Regime)
    for regime_index, regime in enumerate(regimes):
        regime_trials = []
        for trial_index in range(trials_per_regime):
            seed = config.seed + regime_index * trials_per_regime + trial_index
            result = run_trial(regime, seed, config)
            trial_results.append(result); regime_trials.append(result)
        summaries.append(summarize_trials(regime_trials))
    elapsed = time.perf_counter() - started
    return {"metadata": {"project": PROJECT, "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "elapsed_seconds": elapsed}, "config": config.to_dict(), "summaries": summaries, "trials": [t.to_dict() for t in trial_results]}


def threshold_sweep(thresholds, trials_per_regime, config):
    results = []
    for threshold in thresholds:
        threshold_config = Config(channels=config.channels, block_size=config.block_size, threshold=float(threshold), seed=config.seed, score_weights=config.score_weights)
        benchmark = run_benchmark(threshold_config, trials_per_regime)
        for summary in benchmark["summaries"]:
            results.append({"threshold": float(threshold), **summary})
    return results


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def write_csv(rows, path):
    if not rows: raise ValueError("Cannot write empty CSV data.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def run_tests():
    config = Config()
    zero = np.zeros(config.channels)
    scores = score_signal(zero, config)
    assert len(scores) == 16
    assert np.allclose(scores, 0.0)
    stable = run_trial(Regime.STABLE, 42, config)
    stable_again = run_trial(Regime.STABLE, 42, config)
    assert stable.to_dict() == stable_again.to_dict()
    burst = run_trial(Regime.LOCALIZED_BURST, 42, config)
    assert burst.event_block is not None
    assert burst.event_start is not None
    assert burst.event_stop is not None
    shock = run_trial(Regime.GLOBAL_SHOCK, 42, config)
    assert shock.detected_count >= 1
    try:
        Config(channels=500, block_size=32)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid dimensions were accepted.")
    print("All tests passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the OES-512 synthetic telemetry benchmark.")
    parser.add_argument("--trials", type=int, default=1000, help="Trials per regime.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Detection threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output", type=Path, default=Path("outputs/oes512_results.json"), help="JSON output path.")
    parser.add_argument("--csv", type=Path, default=Path("outputs/oes512_summary.csv"), help="CSV summary output path.")
    parser.add_argument("--test", action="store_true", help="Run built-in tests and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test:
        run_tests(); return
    config = Config(threshold=args.threshold, seed=args.seed)
    benchmark = run_benchmark(config=config, trials_per_regime=args.trials)
    write_json(benchmark, args.output)
    write_csv(benchmark["summaries"], args.csv)
    manifest = {"project": PROJECT, "version": VERSION, "config": config.to_dict(), "result_file": str(args.output), "summary_file": str(args.csv), "result_sha256": sha256_file(args.output), "summary_sha256": sha256_file(args.csv)}
    manifest_path = args.output.with_name("manifest.json")
    write_json(manifest, manifest_path)
    print(f"Results written to: {args.output}")
    print(f"Summary written to: {args.csv}")
    print(f"Manifest written to: {manifest_path}")
    for summary in benchmark["summaries"]:
        print(f"{summary['regime']}: mean_detected_blocks={summary['mean_detected_blocks']:.3f}, zero_detection_rate={summary['zero_detection_rate']:.3f}")


if __name__ == "__main__":
    main()
