# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Replay evaluation: score timestamped 512-channel replay files against a frozen protocol.

``oes-resilience replay`` validates a JSON Lines replay (one frame per line, strictly
increasing timestamps, labelled event episodes), runs registered detectors on it and
writes a deterministic report of per-regime event recall, false alarms, block
localization and detection latency, plus an optional incumbent column.

Thresholds come only from a **preregistration** file whose exact bytes are hashed into
the report. Two policies:

* ``calibrated`` (default in the examples): every detector is calibrated on a separate,
  event-free calibration replay to the same frame-level false-positive budget
  (``target_fp``, e.g. 0.01), with :func:`oes_resilience.scorecard.calibrate_threshold`.
  The SHA-256 of the calibration file is locked in the preregistration.
* ``fixed``: thresholds are written in the preregistration. With equal numeric
  thresholds for detectors on different scales (e.g. ``oes32`` and ``maxabs`` both at
  0.50) the comparison is **not** calibrated and must not be read as one detector
  beating another; the report says so.

This module only reads local files. It is not a facility connector, an operational
alarm, a safety instrument or a process-control function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import core
from .core import Config, atomic_write, csv_bytes, json_bytes
from .detectors import Detector, available_detectors, create_detector, load_plugins
from .scorecard import calibrate_threshold

CHANNEL_COUNT = 512
BLOCK_SIZE = 32
BLOCK_COUNT = CHANNEL_COUNT // BLOCK_SIZE
FRAME_REQUIRED = frozenset({"timestamp", "channels", "regime", "event_label", "event_id"})
FRAME_OPTIONAL = frozenset({"affected_blocks", "incumbent_flags", "incumbent_alarm"})
SCORE_WEIGHTS = {"peak_abs": 0.45, "rms": 0.35, "mean_abs": 0.20}
FIXED_RULES = {
    "event_hit_rule": "any_alert_frame_during_event",
    "localization_rule": "framewise_micro_precision_recall_iou",
    "latency_rule": "first_alert_frame_minus_first_event_frame_ms",
}
#: v1 (schema_version 1) preregistrations name these two detectors implicitly.
V1_DETECTORS = {"candidate_threshold": "oes32", "baseline_max_abs_threshold": "maxabs"}
DEFAULT_WARMUP = 16
FAIRNESS_NOTE = (
    "fixed thresholds are not calibrated to a common false-positive budget; equal numeric thresholds on "
    "detectors with different score scales do not make a fair comparison, and the results must not be read "
    "as one detector outperforming another"
)
CSV_FIELDS = (
    "detector", "threshold", "threshold_source", "regime", "frame_count", "event_count", "detected_event_count",
    "event_recall", "missed_event_ids", "no_event_frame_count", "false_positive_frame_count",
    "false_positive_frame_rate", "false_positive_frames_per_10min", "false_positive_alarm_episodes_per_10min",
    "localization_precision", "localization_recall", "localization_iou", "mean_latency_ms", "median_latency_ms",
)


class ValidationError(ValueError):
    """An input or protocol file does not match the documented schema."""


@dataclass(frozen=True)
class Frame:
    """One validated replay frame (timestamps normalised to UTC)."""

    timestamp: datetime
    channels: tuple[float, ...]
    regime: str
    event_label: str
    event_id: str | None
    affected_blocks: tuple[int, ...] | None
    incumbent_flags: tuple[bool, ...] | None
    incumbent_alarm: bool | None


# --------------------------------------------------------------------------
# Parsing and validation (fail closed)
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-standard JSON numeric constant is not allowed: {value}")


def _load_json(text: str, where: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except ValidationError:
        raise
    except RecursionError as exc:
        raise ValidationError(f"{where}: JSON is nested too deeply") from exc
    except ValueError as exc:  # JSONDecodeError, oversized integer literals
        raise ValidationError(f"{where}: invalid JSON: {exc}") from exc


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a JSON number (not a boolean)")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValidationError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _timezone_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be an ISO-8601 timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{field} must include a UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationError(f"{field} must be a non-empty string without outer whitespace")
    return value


def _block_list(value: Any, field: str, *, event: bool) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValidationError(f"{field} must be an array of block indices or null")
    blocks: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValidationError(f"{field} entries must be integer block indices")
        if item < 0 or item >= BLOCK_COUNT:
            raise ValidationError(f"{field} entries must be in [0, {BLOCK_COUNT - 1}]")
        if item in blocks:
            raise ValidationError(f"{field} entries must be unique")
        blocks.append(item)
    if event and not blocks:
        raise ValidationError(f"{field} must be non-empty when event blocks are known")
    if not event and blocks:
        raise ValidationError(f"{field} must be empty on a no-event frame")
    return tuple(sorted(blocks))


def _parse_frame(obj: Any, where: str) -> Frame:
    if not isinstance(obj, dict):
        raise ValidationError(f"{where}: each JSONL record must be an object")
    keys = set(obj)
    missing = FRAME_REQUIRED - keys
    unknown = keys - FRAME_REQUIRED - FRAME_OPTIONAL
    if missing:
        raise ValidationError(f"{where}: missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValidationError(f"{where}: unknown fields: {', '.join(sorted(unknown))}")
    timestamp = _timezone_timestamp(obj["timestamp"], f"{where}.timestamp")
    regime = _nonempty_string(obj["regime"], f"{where}.regime")
    event_label = _nonempty_string(obj["event_label"], f"{where}.event_label")
    if event_label == "none":
        if obj["event_id"] is not None:
            raise ValidationError(f"{where}.event_id must be null when event_label is 'none'")
        event_id = None
    else:
        event_id = _nonempty_string(obj["event_id"], f"{where}.event_id")
    raw_channels = obj["channels"]
    if not isinstance(raw_channels, list) or len(raw_channels) != CHANNEL_COUNT:
        raise ValidationError(f"{where}.channels must be an array of exactly {CHANNEL_COUNT} numbers")
    channels = tuple(_finite_number(value, f"{where}.channels[{index}]") for index, value in enumerate(raw_channels))
    affected_blocks = None
    if "affected_blocks" in obj:
        affected_blocks = _block_list(obj["affected_blocks"], f"{where}.affected_blocks", event=event_id is not None)
    incumbent_flags = None
    if "incumbent_flags" in obj:
        raw_flags = obj["incumbent_flags"]
        if not isinstance(raw_flags, list) or len(raw_flags) != BLOCK_COUNT:
            raise ValidationError(f"{where}.incumbent_flags must be a {BLOCK_COUNT}-element boolean array")
        if any(not isinstance(flag, bool) for flag in raw_flags):
            raise ValidationError(f"{where}.incumbent_flags entries must be booleans")
        incumbent_flags = tuple(raw_flags)
    incumbent_alarm = None
    if "incumbent_alarm" in obj:
        if not isinstance(obj["incumbent_alarm"], bool):
            raise ValidationError(f"{where}.incumbent_alarm must be a boolean")
        incumbent_alarm = obj["incumbent_alarm"]
    return Frame(timestamp, channels, regime, event_label, event_id, affected_blocks, incumbent_flags, incumbent_alarm)


def _read_utf8(path: str | Path, what: str) -> tuple[bytes, str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read {what} {source}: {exc}") from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{what} {source} must be UTF-8") from exc


def load_replay(path: str | Path) -> tuple[list[Frame], bytes]:
    """Read and validate a UTF-8 JSONL replay; return the frames and the exact file bytes."""
    source = Path(path)
    raw, text = _read_utf8(source, "replay file")
    # Split on JSON Lines record separators only: str.splitlines() would also split on
    # U+2028/U+2029 and other characters that are legal inside JSON strings.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    frames: list[Frame] = []
    for number, line in enumerate(lines, start=1):
        line = line[:-1] if line.endswith("\r") else line
        if not line.strip():
            raise ValidationError(f"{source}:{number}: blank lines are not allowed")
        frames.append(_parse_frame(_load_json(line, f"{source}:{number}"), f"{source}:{number}"))
    if not frames:
        raise ValidationError(f"replay file {source} contains no frames")
    for index in range(1, len(frames)):
        if frames[index].timestamp <= frames[index - 1].timestamp:
            raise ValidationError(f"timestamps must be strictly increasing (frames {index} and {index + 1})")
    active: str | None = None
    seen: set[str] = set()
    identity: dict[str, tuple[str, str]] = {}
    for index, frame in enumerate(frames, start=1):
        if frame.event_id is None:
            if active is not None:
                seen.add(active)
            active = None
            continue
        key = (frame.regime, frame.event_label)
        if identity.get(frame.event_id, key) != key:
            raise ValidationError(f"event_id {frame.event_id!r} changes regime or event_label at frame {index}")
        identity[frame.event_id] = key
        if frame.event_id != active:
            if frame.event_id in seen:
                raise ValidationError(f"event_id {frame.event_id!r} must occupy one contiguous run")
            if active is not None:
                seen.add(active)
            active = frame.event_id
    rows = [f.incumbent_flags is not None or f.incumbent_alarm is not None for f in frames]
    if any(rows) and not all(rows):
        raise ValidationError("incumbent signals, when present, must cover every replay frame")
    return frames, raw


def _exact_keys(obj: Any, required: set[str], where: str, optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError(f"{where} must be a JSON object")
    missing = required - set(obj)
    unknown = set(obj) - required - optional
    if missing:
        raise ValidationError(f"{where} missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValidationError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    return obj


def _threshold(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number < 0:
        raise ValidationError(f"{field} must be non-negative")
    return number


def _check_score_spec(score: Any) -> None:
    _exact_keys(score, {"channel_count", "block_size", "weights", "comparison"}, "preregistration.score_spec")
    if type(score["channel_count"]) is not int or score["channel_count"] != CHANNEL_COUNT:
        raise ValidationError(f"preregistration score channel_count must equal {CHANNEL_COUNT}")
    if type(score["block_size"]) is not int or score["block_size"] != BLOCK_SIZE:
        raise ValidationError(f"preregistration score block_size must equal {BLOCK_SIZE}")
    _exact_keys(score["weights"], set(SCORE_WEIGHTS), "score weights")
    for name, expected in SCORE_WEIGHTS.items():
        if _finite_number(score["weights"][name], f"score weights.{name}") != expected:
            raise ValidationError("preregistration score weights must match the documented frozen score")
    if score["comparison"] != ">=":
        raise ValidationError("preregistration score comparison must be '>='")


def load_preregistration(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """Read a frozen protocol and return a normalised dict plus the exact file bytes.

    ``schema_version`` 1 (the original replay harness format: ``candidate_threshold`` for
    ``oes32`` and ``baseline_max_abs_threshold`` for ``maxabs``) is read as a ``fixed``
    policy. ``schema_version`` 2 lists detectors and a ``threshold_policy``.
    """
    source = Path(path)
    raw, text = _read_utf8(source, "preregistration file")
    protocol = _load_json(text, str(source))
    if not isinstance(protocol, dict):
        raise ValidationError("preregistration must be a JSON object")
    version = protocol.get("schema_version")
    common = {"schema_version", "protocol_id", "locked_at", "score_spec", *FIXED_RULES}
    if type(version) is int and version == 1:
        _exact_keys(protocol, common | set(V1_DETECTORS), "preregistration")
        thresholds = {name: _threshold(protocol[key], f"preregistration.{key}") for key, name in V1_DETECTORS.items()}
        policy: dict[str, Any] = {"mode": "fixed", "thresholds": thresholds}
        names = list(thresholds)
        warmup = DEFAULT_WARMUP
    elif type(version) is int and version == 2:
        _exact_keys(protocol, common | {"detectors", "threshold_policy"}, "preregistration", frozenset({"warmup"}))
        names = protocol["detectors"]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise ValidationError("preregistration.detectors must be a non-empty array of detector names")
        if len(set(names)) != len(names):
            raise ValidationError("preregistration.detectors must not repeat a name")
        warmup = protocol.get("warmup", DEFAULT_WARMUP)
        if type(warmup) is not int or warmup < 2:
            raise ValidationError("preregistration.warmup must be an integer >= 2")
        policy = _parse_policy(protocol["threshold_policy"], names)
    else:
        raise ValidationError("preregistration.schema_version must equal integer 1 or 2")
    _nonempty_string(protocol["protocol_id"], "preregistration.protocol_id")
    _timezone_timestamp(protocol["locked_at"], "preregistration.locked_at")
    _check_score_spec(protocol["score_spec"])
    for name, expected in FIXED_RULES.items():
        if protocol[name] != expected:
            raise ValidationError(f"preregistration.{name} must be {expected!r}")
    unknown = sorted(set(names) - set(available_detectors()))
    if unknown:
        raise ValidationError(f"unknown detector(s) {unknown}; available: {', '.join(available_detectors())}")
    normalised = {
        "schema_version": version,
        "protocol_id": protocol["protocol_id"],
        "locked_at": protocol["locked_at"],
        "detectors": list(names),
        "warmup": warmup,
        "threshold_policy": policy,
    }
    return normalised, raw


def _parse_policy(policy: Any, names: list[str]) -> dict[str, Any]:
    if not isinstance(policy, dict) or policy.get("mode") not in ("calibrated", "fixed"):
        raise ValidationError("preregistration.threshold_policy.mode must be 'calibrated' or 'fixed'")
    if policy["mode"] == "fixed":
        _exact_keys(policy, {"mode", "thresholds"}, "preregistration.threshold_policy")
        thresholds = _exact_keys(policy["thresholds"], set(names), "preregistration.threshold_policy.thresholds")
        return {"mode": "fixed", "thresholds": {n: _threshold(thresholds[n], f"threshold for {n}") for n in names}}
    _exact_keys(policy, {"mode", "target_fp", "unit", "calibration_sha256"}, "preregistration.threshold_policy")
    target = _finite_number(policy["target_fp"], "preregistration.threshold_policy.target_fp")
    if not 0.0 <= target < 1.0:
        raise ValidationError("preregistration.threshold_policy.target_fp must be in [0, 1)")
    if policy["unit"] != "frame":
        raise ValidationError("preregistration.threshold_policy.unit must be 'frame'")
    digest = policy["calibration_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValidationError("preregistration.threshold_policy.calibration_sha256 must be a lowercase SHA-256 hex")
    return {"mode": "calibrated", "target_fp": target, "unit": "frame", "calibration_sha256": digest}


# --------------------------------------------------------------------------
# Detectors on replays
# --------------------------------------------------------------------------


def _matrix(frames: Sequence[Frame]) -> np.ndarray:
    return np.asarray([frame.channels for frame in frames], dtype=np.float64)


def _build_detector(name: str, warmup: int) -> Detector:
    config = Config(channels=CHANNEL_COUNT, block_size=BLOCK_SIZE)
    kwargs: dict[str, Any] = {}
    detector_class = type(create_detector(name, config))
    if detector_class.temporal:
        kwargs["warmup"] = warmup
    return create_detector(name, config, **kwargs)


def replay_scores(detector: Detector, matrix: np.ndarray) -> np.ndarray:
    """``(frames, 512) -> (frames, 16)`` block scores; temporal detectors see one stream."""
    if detector.temporal and matrix.shape[0] <= detector.warmup:
        raise ValidationError(
            f"{detector.name}: the replay needs more than warmup={detector.warmup} frames; got {matrix.shape[0]}"
        )
    return detector.score(matrix)


def _scored_frames(detector: Detector, count: int) -> np.ndarray:
    """Frames that carry a real score (temporal warm-up frames always score 0)."""
    mask = np.ones(count, dtype=bool)
    if detector.temporal:
        mask[: detector.warmup] = False
    return mask


def calibrate_on_replay(
    detector: Detector, frames: Sequence[Frame], matrix: np.ndarray, target_fp: float
) -> dict[str, Any]:
    """Frame-unit calibration on an event-free replay, with the scorecard's rule.

    Unit = one frame, scored by its maximum block score; per regime, the smallest threshold
    reached by at most ``floor(target_fp * n)`` frames; the maximum over regimes is used.
    """
    scores = replay_scores(detector, matrix).max(axis=1)
    mask = _scored_frames(detector, len(frames))
    maxima: dict[str, list[float]] = {}
    for keep, frame, value in zip(mask, frames, scores, strict=True):
        if keep:
            maxima.setdefault(frame.regime, []).append(float(value))
    return calibrate_threshold(maxima, target_fp)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _exposure_seconds(indices: list[int], frames: Sequence[Frame], step: float | None) -> float:
    """Sum contiguous regime windows, extending each last sample by the median step."""
    if not indices or step is None:
        return 0.0
    groups: list[list[int]] = []
    for index in indices:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return sum((frames[g[-1]].timestamp - frames[g[0]].timestamp).total_seconds() + step for g in groups)


def detector_metrics(
    frames: Sequence[Frame], alarms: Sequence[bool], block_flags: Sequence[Sequence[bool] | None]
) -> dict[str, dict[str, Any]]:
    """Per-regime event, false-alarm, localization and latency outcomes."""
    if len(frames) != len(alarms) or len(frames) != len(block_flags):
        raise ValueError("detector outputs must have one entry per frame")
    step = _median([(frames[i].timestamp - frames[i - 1].timestamp).total_seconds() for i in range(1, len(frames))])
    results: dict[str, dict[str, Any]] = {}
    for regime in dict.fromkeys(frame.regime for frame in frames):
        indices = [i for i, frame in enumerate(frames) if frame.regime == regime]
        events: dict[str, list[int]] = {}
        for i in indices:
            if frames[i].event_id is not None:
                events.setdefault(frames[i].event_id, []).append(i)
        detected: list[str] = []
        missed: list[str] = []
        latency: dict[str, float | None] = {}
        for event_id, members in events.items():
            hit = next((i for i in members if alarms[i]), None)
            if hit is None:
                missed.append(event_id)
                latency[event_id] = None
            else:
                detected.append(event_id)
                latency[event_id] = (frames[hit].timestamp - frames[members[0]].timestamp).total_seconds() * 1000.0
        quiet = [i for i in indices if frames[i].event_id is None]
        false_frames = [i for i in quiet if alarms[i]]
        episodes, previous = 0, None
        for i in quiet:
            if alarms[i] and previous != i - 1:
                episodes += 1
            previous = i if alarms[i] else None
        tp = fp = fn = annotated = 0
        for i in indices:
            truth, predicted = frames[i].affected_blocks, block_flags[i]
            if truth is None or predicted is None:
                continue
            annotated += 1
            truth_set = set(truth)
            predicted_set = {b for b, flag in enumerate(predicted) if flag}
            tp += len(truth_set & predicted_set)
            fp += len(predicted_set - truth_set)
            fn += len(truth_set - predicted_set)
        exposure = _exposure_seconds(indices, frames, step)
        found = [v for v in latency.values() if v is not None]
        results[regime] = {
            "frame_count": len(indices),
            "event_count": len(events),
            "detected_event_count": len(detected),
            "event_recall": _divide(len(detected), len(events)),
            "missed_event_count": len(missed),
            "missed_event_ids": missed,
            "event_frame_count": len(indices) - len(quiet),
            "alarm_frame_count": sum(1 for i in indices if alarms[i]),
            "no_event_frame_count": len(quiet),
            "false_positive_frame_count": len(false_frames),
            "false_positive_frame_rate": _divide(len(false_frames), len(quiet)),
            "false_positive_frames_per_10min": len(false_frames) * 600.0 / exposure if exposure > 0 else None,
            "false_positive_alarm_episode_count": episodes,
            "false_positive_alarm_episodes_per_10min": episodes * 600.0 / exposure if exposure > 0 else None,
            "observed_exposure_seconds": exposure,
            "block_localization": {
                "annotated_frame_count": annotated,
                "tp_blocks_across_frames": tp,
                "fp_blocks_across_frames": fp,
                "fn_blocks_across_frames": fn,
                "precision": _divide(tp, tp + fp),
                "recall": _divide(tp, tp + fn),
                "iou": _divide(tp, tp + fp + fn),
            },
            "detection_latency_ms": {
                "mean": sum(found) / len(found) if found else None,
                "median": _median(found),
                "by_event_id": latency,
            },
        }
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(
    frames: Sequence[Frame],
    replay_bytes: bytes,
    protocol: dict[str, Any],
    prereg_bytes: bytes,
    calibration: tuple[Sequence[Frame], bytes] | None = None,
) -> dict[str, Any]:
    """Evaluate every preregistered detector (and the incumbent, if present) on the replay."""
    policy = protocol["threshold_policy"]
    names: list[str] = protocol["detectors"]
    calibration_frames: Sequence[Frame] = ()
    calibration_matrix: np.ndarray | None = None
    calibration_hash = None
    if calibration is not None:
        calibration_frames, calibration_bytes = calibration
        calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
        if any(frame.event_id is not None for frame in calibration_frames):
            raise ValidationError("the calibration replay must be event-free (event_label 'none' on every frame)")
        calibration_matrix = _matrix(calibration_frames)
    if policy["mode"] == "calibrated":
        if calibration is None:
            raise ValidationError("threshold_policy 'calibrated' needs --calibration")
        if calibration_hash != policy["calibration_sha256"]:
            raise ValidationError(
                "calibration replay SHA-256 does not match preregistration.threshold_policy.calibration_sha256"
            )
    matrix = _matrix(frames)
    detector_results: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for name in names:
        detector = _build_detector(name, protocol["warmup"])
        if detector.requires_fit:
            if calibration_matrix is None:
                raise ValidationError(f"detector {name!r} must be fitted; supply a clean --calibration replay")
            detector.fit(calibration_matrix)
        if policy["mode"] == "calibrated":
            record = calibrate_on_replay(detector, calibration_frames, calibration_matrix, policy["target_fp"])
            threshold = record["threshold"]
            thresholds[name] = {"threshold": threshold, "source": "calibrated", "calibration": record}
        else:
            threshold = policy["thresholds"][name]
            thresholds[name] = {"threshold": threshold, "source": "fixed"}
        flags = replay_scores(detector, matrix) >= threshold
        block_flags = [tuple(bool(v) for v in row) for row in flags]
        alarms = [any(row) for row in block_flags]
        detector_results[name] = {
            "rule": detector.rule,
            "temporal": detector.temporal,
            "params": detector.params(),
            "regimes": detector_metrics(frames, alarms, block_flags),
        }
    incumbent = any(f.incumbent_flags is not None or f.incumbent_alarm is not None for f in frames)
    if incumbent:
        alarms = [f.incumbent_alarm if f.incumbent_alarm is not None else any(f.incumbent_flags or ()) for f in frames]
        detector_results["incumbent"] = {
            "rule": "incumbent output supplied in the replay",
            "temporal": None,
            "params": {},
            "regimes": detector_metrics(frames, alarms, [f.incumbent_flags for f in frames]),
        }
    fair = policy["mode"] == "calibrated"
    return {
        "report_schema_version": 2,
        "scope": "offline replay evaluation only; not an operational alarm or process-control function",
        "evidence": "results describe the supplied replay only; synthetic inputs give SYNTHETIC results",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "locked_at": protocol["locked_at"],
            "schema_version": protocol["schema_version"],
            "sha256": hashlib.sha256(prereg_bytes).hexdigest(),
            "threshold_policy": policy,
            "warmup_for_temporal_detectors": protocol["warmup"],
            "score": "oes32 = 0.45 * max|x| + 0.35 * RMS(x) + 0.20 * mean|x| per 32-channel block; score >= threshold",
        },
        "input": {
            "sha256": hashlib.sha256(replay_bytes).hexdigest(),
            "frame_count": len(frames),
            "regimes_in_first_seen_order": list(dict.fromkeys(f.regime for f in frames)),
            "incumbent_available": incumbent,
            "calibration_sha256": calibration_hash,
            "calibration_frame_count": len(calibration_frames) if calibration is not None else None,
        },
        "thresholds": thresholds,
        "comparison_is_calibrated": fair,
        "fairness_note": None if fair else FAIRNESS_NOTE,
        "metric_definitions": {
            "event_recall": "fraction of contiguous event_id episodes with at least one alarm frame",
            "false_positive_frame_rate": "alarm frames among labelled no-event frames",
            "per_10min": "counts per contiguous regime exposure; each final sample extends by the median interval",
            "localization": "framewise block micro precision, recall and IoU over frames with affected_blocks",
            "latency_ms": "first alarm frame minus the first frame of the event episode; missed events are null",
            "temporal_detectors": "run over the whole replay as one stream; the first `warmup` frames score 0",
            "calibration": "frame unit, maximum block score per frame, per-regime rule of calibrate_threshold",
        },
        "detectors": detector_results,
    }


def scorecard_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a report into one CSV row per detector and regime (rounded for stable output)."""
    r = core._round
    rows = []
    for name, result in report["detectors"].items():
        entry = report["thresholds"].get(name, {"threshold": None, "source": "supplied"})
        for regime, m in result["regimes"].items():
            loc, lat = m["block_localization"], m["detection_latency_ms"]
            rows.append({
                "detector": name,
                "threshold": r(entry["threshold"]),
                "threshold_source": entry["source"],
                "regime": regime,
                "frame_count": m["frame_count"],
                "event_count": m["event_count"],
                "detected_event_count": m["detected_event_count"],
                "event_recall": r(m["event_recall"]),
                "missed_event_ids": ";".join(m["missed_event_ids"]),
                "no_event_frame_count": m["no_event_frame_count"],
                "false_positive_frame_count": m["false_positive_frame_count"],
                "false_positive_frame_rate": r(m["false_positive_frame_rate"]),
                "false_positive_frames_per_10min": r(m["false_positive_frames_per_10min"]),
                "false_positive_alarm_episodes_per_10min": r(m["false_positive_alarm_episodes_per_10min"]),
                "localization_precision": r(loc["precision"]),
                "localization_recall": r(loc["recall"]),
                "localization_iou": r(loc["iou"]),
                "mean_latency_ms": r(lat["mean"]),
                "median_latency_ms": r(lat["median"]),
            })
    return rows


# --------------------------------------------------------------------------
# Deterministic synthetic example inputs (stdlib random; byte-identical on CPython 3.x)
# --------------------------------------------------------------------------

EXAMPLE_SEED = 20260930
CALIBRATION_SEED = 20261001
EXAMPLE_FILES = {"replay": "synthetic_replay.jsonl", "calibration": "synthetic_calibration.jsonl"}


def _stamp(seconds: int) -> str:
    minute, second = divmod(seconds, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-09-30T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def synthetic_rows() -> list[dict[str, Any]]:
    """The 240-frame example replay: Stable, Noisy, Localized Burst, Global Shock (60 frames each)."""
    rng = random.Random(EXAMPLE_SEED)
    rows: list[dict[str, Any]] = []
    for segment, regime in enumerate(("Stable", "Noisy", "Localized Burst", "Global Shock")):
        for offset in range(60):
            event_id, label, affected = None, "none", []
            if regime == "Noisy":
                values = [rng.gauss(0.25, 0.10) for _ in range(CHANNEL_COUNT)]
            else:
                values = [rng.gauss(0.0, 0.05) for _ in range(CHANNEL_COUNT)]
            if regime == "Localized Burst" and (12 <= offset < 18 or 40 <= offset < 46):
                block = 4 if offset < 18 else 11
                event_id, label, affected = ("burst-01" if block == 4 else "burst-02"), "localized_burst", [block]
                for channel in range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE):
                    values[channel] += rng.gauss(0.80, 0.15)
            elif regime == "Global Shock" and 25 <= offset < 31:
                event_id, label, affected = "shock-01", "global_shock", list(range(BLOCK_COUNT))
                values = [value + 2.0 for value in values]
            rows.append({
                "timestamp": _stamp(segment * 60 + offset),
                "channels": [round(value, 6) for value in values],
                "regime": regime,
                "event_label": label,
                "event_id": event_id,
                "affected_blocks": affected,
            })
    return rows


def calibration_rows() -> list[dict[str, Any]]:
    """Event-free calibration replay (separate seed): 300 Stable then 300 Noisy frames."""
    rng = random.Random(CALIBRATION_SEED)
    rows = []
    for segment, (regime, mean, std) in enumerate((("Stable", 0.0, 0.05), ("Noisy", 0.25, 0.10))):
        for offset in range(300):
            rows.append({
                "timestamp": _stamp(segment * 300 + offset),
                "channels": [round(rng.gauss(mean, std), 6) for _ in range(CHANNEL_COUNT)],
                "regime": regime,
                "event_label": "none",
                "event_id": None,
                "affected_blocks": [],
            })
    return rows


def jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode("utf-8")


def write_example_inputs(out_dir: Path) -> dict[str, Path]:
    """Write the synthetic example replay and calibration replay; return their paths."""
    out_dir = Path(out_dir)
    paths = {key: out_dir / name for key, name in EXAMPLE_FILES.items()}
    atomic_write(paths["replay"], jsonl_bytes(synthetic_rows()))
    atomic_write(paths["calibration"], jsonl_bytes(calibration_rows()))
    return paths


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_replay_parser(sub: Any) -> argparse.ArgumentParser:
    """Register the ``replay`` subcommand."""
    parser = sub.add_parser("replay", help="evaluate detectors on a timestamped replay file against a preregistration")
    parser.add_argument("--input", type=Path, help="UTF-8 JSONL replay file")
    parser.add_argument("--prereg", type=Path, help="frozen preregistration JSON (thresholds come only from here)")
    parser.add_argument("--calibration", type=Path, help="event-free calibration replay (calibrated policy, fitting)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="output directory (default results)")
    parser.add_argument("--prefix", default="oes_resilience_replay", help="output file prefix")
    parser.add_argument("--write-example-inputs", type=Path, metavar="DIR",
                        help="write the deterministic synthetic example replay and calibration files, then exit")
    parser.add_argument("--plugins", action="store_true", help="load detectors from installed entry points")
    parser.add_argument("--verify", action="store_true", help="recompute the report and exit 4 unless its hash matches")
    parser.add_argument("--quiet", action="store_true", help="do not print the summary")
    return parser


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    frames, replay_bytes = load_replay(args.input)
    protocol, prereg_bytes = load_preregistration(args.prereg)
    calibration = load_replay(args.calibration) if args.calibration is not None else None
    return build_report(frames, replay_bytes, protocol, prereg_bytes, calibration)


def cmd_replay(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Handler for ``replay``."""
    if args.write_example_inputs is not None:
        paths = write_example_inputs(args.write_example_inputs)
        if not args.quiet:
            for key, path in paths.items():
                print(f"{key}: {path} sha256={core.sha256_file(path)}")
        return core.EXIT_OK
    if args.input is None or args.prereg is None:
        raise ValidationError("replay needs --input and --prereg (or --write-example-inputs DIR)")
    if args.plugins:
        load_plugins()
    started = time.perf_counter()
    report = _evaluate(args)
    rows = scorecard_rows(report)
    out = Path(args.out_dir)
    paths = {
        "report": out / f"{args.prefix}_report.json",
        "scorecard": out / f"{args.prefix}_scorecard.csv",
        "manifest": out / f"{args.prefix}_manifest.json",
        "metadata": out / f"{args.prefix}_run_metadata.json",
    }
    report_hash = atomic_write(paths["report"], json_bytes(report))
    csv_hash = atomic_write(paths["scorecard"], csv_bytes(rows, CSV_FIELDS))
    manifest = {"project": core.PROJECT, "version": core.__version__,
                "files": {paths["report"].name: {"sha256": report_hash}, paths["scorecard"].name: {"sha256": csv_hash}}}
    manifest_hash = atomic_write(paths["manifest"], json_bytes(manifest))
    atomic_write(paths["metadata"], json_bytes({**core.run_metadata(argv, time.perf_counter() - started),
                                                "manifest_sha256": manifest_hash}))
    if args.verify and hashlib.sha256(json_bytes(_evaluate(args))).hexdigest() != report_hash:
        print("error: replay report hash mismatch on recomputation", file=sys.stderr)
        return core.EXIT_NOT_REPRODUCIBLE
    if not args.quiet:
        if report["fairness_note"]:
            print(f"note: {report['fairness_note']}")
        for row in rows:
            print(f"{row['detector']:>12} thr={row['threshold']} {row['regime']:<16} recall={row['event_recall']} "
                  f"fp_rate={row['false_positive_frame_rate']} iou={row['localization_iou']}")
        for key, path in paths.items():
            print(f"{key}: {path}")
    return core.EXIT_OK
