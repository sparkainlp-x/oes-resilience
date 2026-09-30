# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for v0.4: replay evaluation against a preregistration (ported from oes512-replay-eval)."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oes_resilience import core, detectors, replay  # noqa: E402
from oes_resilience.replay import (  # noqa: E402
    BLOCK_COUNT,
    CHANNEL_COUNT,
    ValidationError,
    build_report,
    load_preregistration,
    load_replay,
)

EXAMPLES = ROOT / "examples"
CALIBRATED = EXAMPLES / "replay_prereg_calibrated.json"
FIXED = EXAMPLES / "replay_prereg_fixed050.json"


def protocol_v1() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "test-locked-v1",
        "locked_at": "2026-09-30T17:55:00-04:00",
        "candidate_threshold": 0.5,
        "baseline_max_abs_threshold": 0.5,
        "score_spec": {
            "channel_count": 512,
            "block_size": 32,
            "weights": {"peak_abs": 0.45, "rms": 0.35, "mean_abs": 0.2},
            "comparison": ">=",
        },
        "event_hit_rule": "any_alert_frame_during_event",
        "localization_rule": "framewise_micro_precision_recall_iou",
        "latency_rule": "first_alert_frame_minus_first_event_frame_ms",
    }


def protocol_v2(policy: dict, names: list[str] | None = None) -> dict:
    base = protocol_v1()
    for key in ("candidate_threshold", "baseline_max_abs_threshold"):
        base.pop(key)
    base.update(schema_version=2, detectors=names or ["oes32", "maxabs"], threshold_policy=policy, warmup=4)
    return base


def record(second: int, regime: str, channels=None, event_label="none", event_id=None, affected_blocks=None) -> dict:
    return {
        "timestamp": f"2026-09-30T00:00:{second:02d}+00:00",
        "channels": channels if channels is not None else [0.0] * CHANNEL_COUNT,
        "regime": regime,
        "event_label": event_label,
        "event_id": event_id,
        "affected_blocks": affected_blocks if affected_blocks is not None else [],
    }


def metric_records() -> list[dict]:
    burst_late = [0.0] * CHANNEL_COUNT
    burst_late[3 * 32 : 4 * 32] = [0.8] * 32
    shock = [0.8] * CHANNEL_COUNT
    noisy_peak = [0.0] * CHANNEL_COUNT
    noisy_peak[5 * 32] = 0.6
    return [
        record(0, "Stable"),
        record(1, "Stable"),
        record(2, "Burst", event_label="localized_burst", event_id="b1", affected_blocks=[3]),
        record(3, "Burst", burst_late, "localized_burst", "b1", [3]),
        record(4, "Burst", burst_late, "localized_burst", "b1", [3]),
        record(5, "Shock", shock, "global_shock", "s1", list(range(BLOCK_COUNT))),
        record(6, "Shock", shock, "global_shock", "s1", list(range(BLOCK_COUNT))),
        record(7, "Noisy", noisy_peak),
        record(8, "Noisy"),
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def write_prereg(self, protocol: dict, name: str = "prereg.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        return path

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = core.main(list(argv))
        return code, out.getvalue(), err.getvalue()


class ScoreTests(unittest.TestCase):
    def test_oes32_score_matches_documented_weighted_formula(self) -> None:
        det = detectors.OES32Detector()
        values = np.zeros(CHANNEL_COUNT)
        values[0] = 2.0
        values[32:64] = -1.0
        scores = det.score(values)
        self.assertAlmostEqual(scores[0], 0.45 * 2.0 + 0.35 * (2.0 / 32**0.5) + 0.20 * (2.0 / 32), places=15)
        self.assertAlmostEqual(scores[1], 1.0, places=15)
        self.assertEqual(scores[2], 0.0)

    def test_replay_reuses_core_score_signals(self) -> None:
        rng = np.random.default_rng(1)
        frames = rng.normal(rng.choice([0.0, 0.25, 0.8, 2.0], size=(200, 1)), 0.1, size=(200, CHANNEL_COUNT))
        det = replay._build_detector("oes32", 16)
        np.testing.assert_array_equal(replay.replay_scores(det, frames), core.score_signals(frames, det.config))

    def test_maxabs_detector(self) -> None:
        det = detectors.create_detector("maxabs")
        frame = np.zeros(CHANNEL_COUNT)
        frame[40] = -0.7
        self.assertEqual(det.score(frame)[1], 0.7)
        frame[41] = np.nan
        self.assertEqual(det.score(frame)[1], 0.7)


class ValidationTests(TempDirCase):
    def check_invalid(self, rows: list[dict]) -> None:
        path = self.root / "bad.jsonl"
        write_jsonl(path, rows)
        with self.assertRaises(ValidationError):
            load_replay(path)

    def test_valid_schema_and_timezone_normalization(self) -> None:
        path = self.root / "valid.jsonl"
        write_jsonl(path, [record(0, "Stable"), record(1, "Stable")])
        frames, raw = load_replay(path)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].timestamp.utcoffset().total_seconds(), 0)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_wrong_channel_count(self) -> None:
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["channels"] = [0.0] * (CHANNEL_COUNT - 1)
        self.check_invalid(rows)

    def test_rejects_boolean_or_nonfinite_channel_values(self) -> None:
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["channels"][0] = True
        self.check_invalid(rows)
        path = self.root / "nonfinite.jsonl"
        path.write_text(json.dumps(record(0, "Stable")).replace('"channels": [0.0', '"channels": [NaN') + "\n")
        with self.assertRaises(ValidationError):
            load_replay(path)
        big = self.root / "big.jsonl"
        big.write_text(json.dumps(record(0, "Stable")).replace('"channels": [0.0', '"channels": [1' + "0" * 400) + "\n")
        with self.assertRaises(ValidationError):
            load_replay(big)

    def test_rejects_naive_and_non_increasing_timestamps(self) -> None:
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["timestamp"] = "2026-09-30T00:00:00"
        self.check_invalid(rows)
        self.check_invalid([record(0, "Stable"), record(0, "Stable")])
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["timestamp"] = "not a time"
        self.check_invalid(rows)

    def test_rejects_bad_block_labels_unknown_and_missing_keys(self) -> None:
        self.check_invalid([record(0, "B", event_label="b", event_id="x", affected_blocks=[16]), record(1, "B")])
        self.check_invalid([record(0, "B", event_label="b", event_id="x", affected_blocks=[2, 2]), record(1, "B")])
        self.check_invalid([record(0, "B", event_label="b", event_id="x", affected_blocks=[]), record(1, "B")])
        self.check_invalid([record(0, "B", event_label="b", event_id="x", affected_blocks=["1"]), record(1, "B")])
        self.check_invalid([record(0, "B", affected_blocks=[1])])
        rows = [record(0, "Stable")]
        rows[0]["affected_blocks"] = "all"
        self.check_invalid(rows)
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["surprise"] = "reject"
        self.check_invalid(rows)
        rows = [record(0, "Stable")]
        del rows[0]["regime"]
        self.check_invalid(rows)
        rows = [record(0, " Stable")]
        self.check_invalid(rows)
        rows = [record(0, "Stable")]
        rows[0]["event_id"] = "x"
        self.check_invalid(rows)

    def test_rejects_noncontiguous_event_and_partial_incumbent(self) -> None:
        self.check_invalid([
            record(0, "B", event_label="b", event_id="x", affected_blocks=[1]),
            record(1, "B"),
            record(2, "B", event_label="b", event_id="x", affected_blocks=[1]),
        ])
        self.check_invalid([
            record(0, "B", event_label="b", event_id="x", affected_blocks=[1]),
            record(1, "C", event_label="b", event_id="x", affected_blocks=[1]),
        ])
        self.check_invalid([
            record(0, "B", event_label="b", event_id="x", affected_blocks=[1]),
            record(1, "B", event_label="b", event_id="y", affected_blocks=[1]),
            record(2, "B", event_label="b", event_id="x", affected_blocks=[1]),
        ])
        rows = [record(0, "Stable"), record(1, "Stable")]
        rows[0]["incumbent_alarm"] = False
        self.check_invalid(rows)
        rows = [record(0, "Stable")]
        rows[0]["incumbent_alarm"] = "no"
        self.check_invalid(rows)
        rows = [record(0, "Stable")]
        rows[0]["incumbent_flags"] = [0] * BLOCK_COUNT
        self.check_invalid(rows)
        rows = [record(0, "Stable")]
        rows[0]["incumbent_flags"] = [False]
        self.check_invalid(rows)

    def test_rejects_structural_problems(self) -> None:
        for text in ("", "[1]\n", '{"a": 1, "a": 2}\n', "{broken\n", "\xff"):
            path = self.root / "s.jsonl"
            path.write_bytes(text.encode("latin-1") if text == "\xff" else text.encode("utf-8"))
            with self.assertRaises(ValidationError):
                load_replay(path)
        with self.assertRaises(ValidationError):
            load_replay(self.root / "missing.jsonl")

    # Hardening fixes carried over from oes512-replay-eval.
    def test_deeply_nested_json_is_a_validation_error(self) -> None:
        path = self.root / "deep.jsonl"
        path.write_text("[" * 100_000 + "\n", encoding="utf-8")
        with self.assertRaises(ValidationError) as ctx:
            load_replay(path)
        self.assertIn("nested too deeply", str(ctx.exception))

    def test_oversized_integer_literal_is_a_validation_error(self) -> None:
        path = self.root / "bigint.jsonl"
        text = json.dumps(record(0, "Stable")).replace('"channels": [0.0', '"channels": [' + "9" * 5000, 1)
        path.write_text(text + "\n", encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_replay(path)

    def test_unicode_line_separator_is_not_a_record_break(self) -> None:
        path = self.root / "ls.jsonl"
        rows = [record(0, "Sta\u2028ble"), record(1, "Sta\u2028ble")]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        frames, _ = load_replay(path)
        self.assertEqual([f.regime for f in frames], ["Sta\u2028ble"] * 2)

    def test_crlf_accepted_and_blank_line_rejected(self) -> None:
        path = self.root / "crlf.jsonl"
        path.write_bytes("".join(json.dumps(r) + "\r\n" for r in [record(0, "S"), record(1, "S")]).encode())
        self.assertEqual(len(load_replay(path)[0]), 2)
        path.write_text(json.dumps(record(0, "S")) + "\n\n" + json.dumps(record(1, "S")) + "\n", encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_replay(path)


class PreregistrationTests(TempDirCase):
    def expect_invalid(self, protocol: dict) -> None:
        with self.assertRaises(ValidationError):
            load_preregistration(self.write_prereg(protocol))

    def test_v1_requires_both_finite_frozen_thresholds(self) -> None:
        protocol = protocol_v1()
        protocol.pop("candidate_threshold")
        self.expect_invalid(protocol)
        for key, value in (("candidate_threshold", -0.1), ("candidate_threshold", True), ("schema_version", 1.0)):
            protocol = protocol_v1()
            protocol[key] = value
            self.expect_invalid(protocol)

    def test_v1_maps_to_fixed_oes32_and_maxabs(self) -> None:
        protocol, raw = load_preregistration(self.write_prereg(protocol_v1()))
        self.assertEqual(protocol["detectors"], ["oes32", "maxabs"])
        self.assertEqual(protocol["threshold_policy"], {"mode": "fixed", "thresholds": {"oes32": 0.5, "maxabs": 0.5}})

    def test_score_spec_and_rules_are_locked(self) -> None:
        mutations = [
            ("score_spec", "channel_count", 256), ("score_spec", "block_size", 16),
            ("score_spec", "comparison", ">"), ("score_spec", "extra", 1),
        ]
        for outer, inner, value in mutations:
            protocol = protocol_v1()
            protocol[outer][inner] = value
            self.expect_invalid(protocol)
        protocol = protocol_v1()
        protocol["score_spec"]["weights"]["rms"] = 0.36
        self.expect_invalid(protocol)
        protocol = protocol_v1()
        protocol["latency_rule"] = "other"
        self.expect_invalid(protocol)
        for key, value in (("locked_at", "2026-09-30T17:55:00"), ("protocol_id", ""), ("score_spec", [])):
            protocol = protocol_v1()
            protocol[key] = value
            self.expect_invalid(protocol)

    def test_v2_policy_validation(self) -> None:
        good_cal = {"mode": "calibrated", "target_fp": 0.01, "unit": "frame", "calibration_sha256": "a" * 64}
        load_preregistration(self.write_prereg(protocol_v2(good_cal)))
        bad_policies = [
            {"mode": "tuned"},
            {**good_cal, "target_fp": 1.0},
            {**good_cal, "unit": "stream"},
            {**good_cal, "calibration_sha256": "A" * 64},
            {"mode": "fixed", "thresholds": {"oes32": 0.5}},
            {"mode": "fixed", "thresholds": {"oes32": 0.5, "maxabs": -1}},
        ]
        for policy in bad_policies:
            self.expect_invalid(protocol_v2(policy))
        fixed = {"mode": "fixed", "thresholds": {"oes32": 0.5, "maxabs": 0.5}}
        for names in (["oes32", "oes32"], [], ["nope"], [1]):
            self.expect_invalid(protocol_v2({**fixed, "thresholds": {}}, names) if names == [] else
                                protocol_v2(fixed, names))
        protocol = protocol_v2(fixed)
        protocol["warmup"] = 1
        self.expect_invalid(protocol)
        protocol = protocol_v2(fixed)
        protocol["schema_version"] = 3
        self.expect_invalid(protocol)
        with self.assertRaises(ValidationError):
            load_preregistration(self.write_prereg([1]))  # type: ignore[arg-type]


class MetricsTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        path = self.root / "replay.jsonl"
        write_jsonl(path, metric_records())
        self.frames, self.replay_bytes = load_replay(path)
        self.prereg_path = self.write_prereg(protocol_v1())
        self.protocol, self.prereg_bytes = load_preregistration(self.prereg_path)

    def report(self, frames=None, raw=None) -> dict:
        return build_report(frames or self.frames, raw or self.replay_bytes, self.protocol, self.prereg_bytes)

    def test_event_recall_misses_false_alarms_and_regimes(self) -> None:
        report = self.report()
        oes = report["detectors"]["oes32"]["regimes"]
        self.assertEqual(oes["Burst"]["event_count"], 1)
        self.assertEqual(oes["Burst"]["event_recall"], 1.0)
        self.assertEqual(oes["Stable"]["event_recall"], None)
        self.assertEqual(oes["Noisy"]["false_positive_frame_rate"], 0.0)
        self.assertEqual(oes["Shock"]["event_recall"], 1.0)
        base = report["detectors"]["maxabs"]["regimes"]
        self.assertEqual(base["Noisy"]["false_positive_frame_count"], 1)
        self.assertGreater(base["Noisy"]["false_positive_alarm_episodes_per_10min"], 0)
        self.assertFalse(report["comparison_is_calibrated"])
        self.assertIn("not calibrated", report["fairness_note"])

    def test_missed_events_are_named_and_have_null_latency(self) -> None:
        path = self.root / "missed.jsonl"
        write_jsonl(path, [
            record(0, "Miss", event_label="weak", event_id="miss-1", affected_blocks=[7]),
            record(1, "Miss", event_label="weak", event_id="miss-1", affected_blocks=[7]),
            record(2, "Miss"),
        ])
        frames, raw = load_replay(path)
        metrics = self.report(frames, raw)["detectors"]["oes32"]["regimes"]["Miss"]
        self.assertEqual(metrics["event_recall"], 0.0)
        self.assertEqual(metrics["missed_event_ids"], ["miss-1"])
        self.assertIsNone(metrics["detection_latency_ms"]["by_event_id"]["miss-1"])

    def test_block_localization_precision_recall_and_iou(self) -> None:
        loc = self.report()["detectors"]["oes32"]["regimes"]["Burst"]["block_localization"]
        self.assertEqual((loc["tp_blocks_across_frames"], loc["fn_blocks_across_frames"]), (2, 1))
        self.assertEqual(loc["fp_blocks_across_frames"], 0)
        self.assertEqual(loc["precision"], 1.0)
        self.assertAlmostEqual(loc["recall"], 2 / 3)
        self.assertAlmostEqual(loc["iou"], 2 / 3)

    def test_detection_latency_is_reported_from_event_onset(self) -> None:
        latency = self.report()["detectors"]["oes32"]["regimes"]["Burst"]["detection_latency_ms"]
        self.assertEqual(latency["by_event_id"]["b1"], 1000.0)
        self.assertEqual(latency["mean"], 1000.0)
        self.assertEqual(latency["median"], 1000.0)

    def test_optional_incumbent_comparison(self) -> None:
        rows = metric_records()
        for row in rows:
            row["incumbent_flags"] = [False] * BLOCK_COUNT
        rows[3]["incumbent_flags"][3] = True
        rows[4]["incumbent_alarm"] = True
        path = self.root / "incumbent.jsonl"
        write_jsonl(path, rows)
        frames, raw = load_replay(path)
        report = self.report(frames, raw)
        self.assertTrue(report["input"]["incumbent_available"])
        self.assertEqual(report["detectors"]["incumbent"]["regimes"]["Burst"]["event_recall"], 1.0)
        csv_rows = replay.scorecard_rows(report)
        self.assertIn("supplied", {row["threshold_source"] for row in csv_rows if row["detector"] == "incumbent"})

    def test_report_is_reproducible_and_hashes_inputs(self) -> None:
        first, second = self.report(), self.report()
        self.assertEqual(core.json_bytes(first), core.json_bytes(second))
        self.assertEqual(first["input"]["sha256"], hashlib.sha256(self.replay_bytes).hexdigest())
        self.assertEqual(first["protocol"]["sha256"], hashlib.sha256(self.prereg_bytes).hexdigest())

    def test_metrics_length_mismatch_and_helpers(self) -> None:
        with self.assertRaises(ValueError):
            replay.detector_metrics(self.frames, [False], [None])
        self.assertIsNone(replay._median([]))
        self.assertEqual(replay._exposure_seconds([], self.frames, 1.0), 0.0)


class CalibrationTests(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.paths = replay.write_example_inputs(Path(cls._dir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_example_inputs_are_deterministic_and_match_locked_hash(self) -> None:
        protocol, _ = load_preregistration(CALIBRATED)
        locked = protocol["threshold_policy"]["calibration_sha256"]
        self.assertEqual(core.sha256_file(self.paths["calibration"]), locked)
        self.assertEqual(self.paths["replay"].read_bytes(), replay.jsonl_bytes(replay.synthetic_rows()))

    def test_calibrated_thresholds_meet_target_on_calibration_data(self) -> None:
        cal_frames, cal_raw = load_replay(self.paths["calibration"])
        frames, raw = load_replay(self.paths["replay"])
        protocol, prereg = load_preregistration(CALIBRATED)
        report = build_report(frames, raw, protocol, prereg, (cal_frames, cal_raw))
        self.assertTrue(report["comparison_is_calibrated"])
        self.assertIsNone(report["fairness_note"])
        for name in protocol["detectors"]:
            record_ = report["thresholds"][name]["calibration"]
            for regime in record_["per_regime"].values():
                self.assertLessEqual(regime["calibration_fp_rate"], 0.01)

    def test_calibration_hash_mismatch_and_missing_calibration_are_rejected(self) -> None:
        frames, raw = load_replay(self.paths["replay"])
        protocol, prereg = load_preregistration(CALIBRATED)
        with self.assertRaises(ValidationError):
            build_report(frames, raw, protocol, prereg)
        tampered = protocol | {"threshold_policy": {**protocol["threshold_policy"], "calibration_sha256": "0" * 64}}
        cal = load_replay(self.paths["calibration"])
        with self.assertRaises(ValidationError):
            build_report(frames, raw, tampered, prereg, cal)

    def test_calibration_must_be_event_free_and_fit_needs_calibration(self) -> None:
        frames, raw = load_replay(self.paths["replay"])
        protocol = {"schema_version": 2, "protocol_id": "x", "locked_at": "2026-09-30T00:00:00Z",
                    "detectors": ["zscore"], "warmup": 16,
                    "threshold_policy": {"mode": "fixed", "thresholds": {"zscore": 3.5}}}
        with self.assertRaises(ValidationError):
            build_report(frames, raw, protocol, b"{}")
        with self.assertRaises(ValidationError):
            build_report(frames, raw, protocol, b"{}", (frames, raw))

    def test_temporal_detector_needs_more_frames_than_warmup(self) -> None:
        det = replay._build_detector("ewma", 16)
        with self.assertRaises(ValidationError):
            replay.replay_scores(det, np.zeros((16, CHANNEL_COUNT)))


class CLITests(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.inputs = Path(cls._dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_committed_example_scorecards_are_reproduced_byte_for_byte(self) -> None:
        code, out, _ = self.run_cli("replay", "--write-example-inputs", str(self.inputs))
        self.assertEqual(code, 0)
        self.assertIn("sha256=", out)
        replay_file, cal = self.inputs / "synthetic_replay.jsonl", self.inputs / "synthetic_calibration.jsonl"
        code, out, _ = self.run_cli("replay", "--input", str(replay_file), "--prereg", str(CALIBRATED),
                                    "--calibration", str(cal), "--out-dir", str(self.root), "--prefix", "replay",
                                    "--verify")
        self.assertEqual(code, 0)
        self.assertIn("replay_scorecard.csv", out)
        self.assertEqual((self.root / "replay_scorecard.csv").read_bytes(),
                         (EXAMPLES / "replay_scorecard.csv").read_bytes())
        code, out, _ = self.run_cli("replay", "--input", str(replay_file), "--prereg", str(FIXED),
                                    "--out-dir", str(self.root), "--prefix", "replay_fixed050", "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "replay_fixed050_scorecard.csv").read_bytes(),
                         (EXAMPLES / "replay_fixed050_scorecard.csv").read_bytes())
        manifest = json.loads((self.root / "replay_manifest.json").read_text())
        self.assertEqual(manifest["version"], core.__version__)

    def test_fixed_example_reproduces_original_harness_numbers(self) -> None:
        rows = {(r.split(",")[0], r.split(",")[3]): r.split(",")
                for r in (EXAMPLES / "replay_fixed050_scorecard.csv").read_text().splitlines()[1:]}
        header = (EXAMPLES / "replay_fixed050_scorecard.csv").read_text().splitlines()[0].split(",")
        rate = header.index("false_positive_frame_rate")
        self.assertEqual(float(rows[("maxabs", "Noisy")][rate]), round(58 / 60, 12))
        self.assertEqual(float(rows[("oes32", "Noisy")][rate]), 0.0)

    def test_exit_codes(self) -> None:
        missing = self.root / "missing.jsonl"
        code, _, err = self.run_cli("replay", "--input", str(missing), "--prereg", str(FIXED), "--out-dir",
                                    str(self.root))
        self.assertEqual(code, core.EXIT_USAGE)
        self.assertIn("cannot read", err)
        self.assertEqual(self.run_cli("replay", "--prereg", str(FIXED))[0], core.EXIT_USAGE)
        deep = self.root / "deep.jsonl"
        deep.write_text("[" * 100_000 + "\n", encoding="utf-8")
        code, _, err = self.run_cli("replay", "--input", str(deep), "--prereg", str(FIXED), "--out-dir",
                                    str(self.root))
        self.assertEqual(code, core.EXIT_USAGE)
        self.assertIn("nested too deeply", err)

    def test_verify_detects_mismatch(self) -> None:
        path = self.root / "r.jsonl"
        write_jsonl(path, metric_records())
        original = replay.build_report
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            report = original(*args, **kwargs)
            if calls["n"] > 1:
                report = copy.deepcopy(report)
                report["scope"] += " (changed)"
            return report

        replay.build_report = flaky
        try:
            code, _, err = self.run_cli("replay", "--input", str(path), "--prereg", str(FIXED), "--out-dir",
                                        str(self.root), "--verify", "--quiet")
        finally:
            replay.build_report = original
        self.assertEqual(code, core.EXIT_NOT_REPRODUCIBLE)
        self.assertIn("mismatch", err)

    def test_plugins_flag_and_help(self) -> None:
        path = self.root / "r.jsonl"
        write_jsonl(path, metric_records())
        code, out, _ = self.run_cli("replay", "--input", str(path), "--prereg", str(FIXED), "--out-dir",
                                    str(self.root), "--plugins")
        self.assertEqual(code, 0)
        self.assertIn("note: fixed thresholds are not calibrated", out)


if __name__ == "__main__":
    unittest.main()
