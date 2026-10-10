# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Test suite for oes_resilience (unittest-style; runs under pytest too).

Run:  python -m pytest tests -q      or   python oes_resilience.py test
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import oes_resilience  # noqa: E402
from oes_resilience import core as oes  # noqa: E402

SMALL = 60  # trials for fast tests


def reference_score_block(block, weights):
    """Literal v0.1.0 scoring loop, used as an independent oracle."""
    block = np.asarray(block, dtype=np.float64)
    maximum = float(np.max(np.abs(block)))
    rms = float(np.sqrt(np.mean(block**2)))
    mean_absolute = float(np.mean(np.abs(block)))
    return weights.maximum * maximum + weights.rms * rms + weights.mean_absolute * mean_absolute


def quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = oes.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestConfigValidation(unittest.TestCase):
    def test_defaults_match_documented_baseline(self):
        c = oes.Config()
        self.assertEqual((c.channels, c.block_size, c.blocks, c.threshold, c.seed), (512, 32, 16, 0.5, 42))
        w = c.score_weights
        self.assertEqual((w.maximum, w.rms, w.mean_absolute), (0.45, 0.35, 0.20))
        self.assertEqual(c.global_min_blocks, 16)

    def test_invalid_dimensions(self):
        for kwargs in ({"channels": 500}, {"channels": 0}, {"block_size": 0}, {"channels": -32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                oes.Config(**kwargs)

    def test_non_integer_dimensions_rejected(self):
        for kwargs in ({"channels": 512.0}, {"block_size": 32.0}, {"channels": True}, {"seed": 1.5}, {"seed": "1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                oes.Config(**kwargs)

    def test_numpy_integers_accepted_and_normalised(self):
        c = oes.Config(channels=np.int64(64), block_size=np.int32(8), seed=np.uint8(3))
        self.assertIs(type(c.channels), int)
        self.assertEqual(c.blocks, 8)

    def test_threshold_must_be_finite_non_negative(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -0.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oes.Config(threshold=bad)
        for bad in (True, "0.5", None):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                oes.Config(threshold=bad)
        self.assertEqual(oes.Config(threshold=0).threshold, 0.0)

    def test_negative_seed_rejected(self):
        with self.assertRaises(ValueError):
            oes.Config(seed=-1)

    def test_global_fraction(self):
        for bad in (0.0, -0.5, 1.01, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oes.Config(global_fraction=bad)
        self.assertEqual(oes.Config(global_fraction=0.5).global_min_blocks, 8)
        self.assertEqual(oes.Config(global_fraction=0.51).global_min_blocks, 9)
        self.assertEqual(oes.Config(channels=4, block_size=4, global_fraction=0.1).global_min_blocks, 1)

    def test_score_weights_validation(self):
        for bad in ((float("nan"), 0.5, 0.5), (float("inf"), 0.0, 0.0), (-0.1, 0.6, 0.5), (0.5, 0.5, 0.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oes.ScoreWeights(*bad)
        with self.assertRaises(TypeError):
            oes.ScoreWeights("0.45", 0.35, 0.20)
        with self.assertRaises(TypeError):
            oes.Config(score_weights=(0.45, 0.35, 0.20))
        oes.ScoreWeights(1.0, 0.0, 0.0)

    def test_to_dict_and_with_threshold(self):
        c = oes.Config()
        d = c.to_dict()
        self.assertEqual(d["blocks"], 16)
        self.assertEqual(d["score_weights"], {"maximum": 0.45, "rms": 0.35, "mean_absolute": 0.2})
        c2 = c.with_threshold(0.7)
        self.assertEqual(c2.threshold, 0.7)
        self.assertEqual(c2.to_dict() | {"threshold": 0.5}, d)
        json.dumps(d, allow_nan=False)

    def test_validate_trials(self):
        self.assertEqual(oes.validate_trials(1), 1)
        with self.assertRaises(ValueError):
            oes.validate_trials(0)
        with self.assertRaises(TypeError):
            oes.validate_trials(2.0)


class TestBlockMapping(unittest.TestCase):
    def test_block_range_arbitrary_sizes(self):
        cases = [
            ((0, 32, 32), [0]),
            ((32, 64, 32), [1]),
            ((32, 96, 16), [2, 3, 4, 5]),
            ((31, 33, 32), [0, 1]),
            ((5, 6, 1), [5]),
            ((0, 512, 512), [0]),
            ((100, 200, 64), [1, 2, 3]),
        ]
        for (start, stop, size), expected in cases:
            with self.subTest(start=start, stop=stop, size=size):
                self.assertEqual(list(oes.block_range(start, stop, size)), expected)
                self.assertEqual(list(oes.Event(start, stop).blocks(size)), expected)

    def test_block_range_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            size = int(rng.integers(1, 40))
            start = int(rng.integers(0, 300))
            stop = start + int(rng.integers(1, 100))
            expected = sorted({ch // size for ch in range(start, stop)})
            self.assertEqual(list(oes.block_range(start, stop, size)), expected)

    def test_event_validation(self):
        self.assertEqual(oes.Event(10, 42).width, 32)
        for bad in ((5, 5), (6, 5), (-1, 3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oes.Event(*bad)
        with self.assertRaises(ValueError):
            oes.block_range(3, 3, 8)


class TestScoring(unittest.TestCase):
    def test_formula_by_hand(self):
        # One block of four channels: x = [3, -4, 0, 0]
        # max|x| = 4, RMS = sqrt(25/4) = 2.5, mean|x| = 7/4 = 1.75
        # score = 0.45*4 + 0.35*2.5 + 0.20*1.75 = 1.8 + 0.875 + 0.35 = 3.025
        c = oes.Config(channels=8, block_size=4)
        scores = oes.score_signals([3, -4, 0, 0, 0, 0, 0, 0], c)
        self.assertAlmostEqual(scores[0], 3.025, places=12)
        self.assertEqual(scores[1], 0.0)

    def test_formula_constant_block(self):
        # Constant block of value v: max = RMS = mean = |v|, so score = |v|.
        c = oes.Config()
        signal = np.repeat(np.linspace(-1.5, 1.5, 16), 32)
        np.testing.assert_allclose(oes.score_signals(signal, c), np.abs(np.linspace(-1.5, 1.5, 16)), rtol=1e-12)

    def test_custom_weights(self):
        c = oes.Config(channels=4, block_size=4, score_weights=oes.ScoreWeights(0.0, 1.0, 0.0))
        self.assertAlmostEqual(oes.score_signals([1, 1, 1, 1], c)[0], 1.0)
        self.assertAlmostEqual(oes.score_signals([2, 0, 0, 0], c)[0], 1.0)

    def test_vectorised_matches_reference_loop(self):
        for channels, size in ((512, 32), (512, 8), (96, 12), (7, 1), (64, 64)):
            c = oes.Config(channels=channels, block_size=size)
            signals = np.random.default_rng(size).normal(0.3, 1.0, (25, channels))
            got = oes.score_signals(signals, c)
            self.assertEqual(got.shape, (25, c.blocks))
            for row in range(25):
                expected = [reference_score_block(b, c.score_weights) for b in signals[row].reshape(c.blocks, size)]
                np.testing.assert_allclose(got[row], expected, rtol=1e-13, atol=0)
            np.testing.assert_array_equal(oes.score_signals(signals[3], c), got[3])

    def test_zero_signal(self):
        scores = oes.score_signals(np.zeros(512), oes.Config())
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(np.all(scores == 0))

    def test_signal_validation(self):
        c = oes.Config()
        for bad in (np.zeros(511), np.zeros((2, 2, 128)), np.zeros((3, 511))):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                oes.score_signals(bad, c)
        for bad in (np.zeros((2, 2, 128)), np.zeros(513)):
            with self.assertRaises(ValueError):
                oes.validate_signal(bad, c)
        s = np.zeros(512)
        s[7] = np.nan
        with self.assertRaises(ValueError):
            oes.score_signals(s, c)
        with self.assertRaises(ValueError):
            oes.score_signals(np.vstack([np.zeros(512), s]), c)

    def test_detect(self):
        np.testing.assert_array_equal(oes.detect([0.49, 0.5, 0.51], 0.5), [False, True, True])
        with self.assertRaises(ValueError):
            oes.detect([0.1, np.nan], 0.5)
        with self.assertRaises(ValueError):
            oes.detect([0.1], float("nan"))


class TestStatus(unittest.TestCase):
    def test_default_requires_all_blocks(self):
        c = oes.Config()
        self.assertEqual(oes.classify_status([], c), "STABLE")
        self.assertEqual(oes.classify_status([3], c), "LOCALIZED_ALERT")
        self.assertEqual(oes.classify_status(range(2), c), "DISTRIBUTED_ALERT")
        self.assertEqual(oes.classify_status(range(15), c), "DISTRIBUTED_ALERT")
        self.assertEqual(oes.classify_status(range(16), c), "GLOBAL_SHOCK")

    def test_global_fraction_majority(self):
        c = oes.Config(global_fraction=0.75)
        self.assertEqual(oes.classify_status(range(11), c), "DISTRIBUTED_ALERT")
        self.assertEqual(oes.classify_status(range(12), c), "GLOBAL_SHOCK")

    def test_single_block_config(self):
        c = oes.Config(channels=32, block_size=32)
        self.assertEqual(oes.classify_status([0], c), "LOCALIZED_ALERT")
        c2 = oes.Config(channels=64, block_size=32)
        self.assertEqual(oes.classify_status([0, 1], c2), "GLOBAL_SHOCK")

    def test_vectorised(self):
        got = oes.classify_counts(np.array([0, 1, 5, 16]), oes.Config()).tolist()
        self.assertEqual(got, ["STABLE", "LOCALIZED_ALERT", "DISTRIBUTED_ALERT", "GLOBAL_SHOCK"])

    def test_assess_signal(self):
        c = oes.Config()
        signal = np.zeros(512)
        signal[64:96] = 1.0
        a = oes.assess_signal(signal, c)
        self.assertEqual(a.detected_blocks, (2,))
        self.assertEqual(a.status, "LOCALIZED_ALERT")
        self.assertAlmostEqual(a.scores[2], 1.0)


class TestSeedsAndGeneration(unittest.TestCase):
    def test_seed_sequence_equals_spawn_tree(self):
        for regime in oes.Regime:
            key = oes.REGIME_KEYS[regime]
            spawned = np.random.SeedSequence(42).spawn(key + 1)[key].spawn(8)[7]
            ours = oes.trial_seed_sequence(42, regime, 7)
            np.testing.assert_array_equal(ours.generate_state(4), spawned.generate_state(4))

    def test_regime_keys_distinct_and_explicit(self):
        self.assertEqual(sorted(oes.REGIME_KEYS.values()), [0, 1, 2, 3])
        self.assertEqual(set(oes.REGIME_KEYS), set(oes.Regime))

    def test_streams_do_not_collide(self):
        states = {
            tuple(oes.trial_seed_sequence(s, r, i).generate_state(2))
            for s in (42, 43) for r in oes.Regime for i in range(50)
        }
        self.assertEqual(len(states), 2 * 4 * 50)

    def test_trial_independent_of_trial_count(self):
        c = oes.Config()
        small, _ = oes.score_regime("noisy", 10, c)
        large, _ = oes.score_regime("noisy", 100, c)
        np.testing.assert_array_equal(small, large[:10])

    def test_chunking_does_not_change_results(self):
        c = oes.Config()
        full, truth_full = oes.score_regime("localized_burst", 25, c)
        with mock.patch.object(oes, "CHUNK_TRIALS", 4):
            chunked, truth_chunked = oes.score_regime("localized_burst", 25, c)
        np.testing.assert_array_equal(full, chunked)
        np.testing.assert_array_equal(truth_full, truth_chunked)

    def test_generation_draw_order_matches_v010(self):
        # Same generator state -> same signal as the original prototype.
        c = oes.Config()
        rng = np.random.default_rng(42)
        signal, event = oes.generate_signal("localized_burst", rng, c)
        ref = np.random.default_rng(42)
        expected = ref.normal(0.0, 0.05, 512)
        block = int(ref.integers(0, 16))
        expected[block * 32:(block + 1) * 32] += ref.normal(0.80, 0.15, 32)
        np.testing.assert_array_equal(signal, expected)
        self.assertEqual(event, oes.Event(block * 32, block * 32 + 32))

    def test_regime_distributions(self):
        c = oes.Config()
        for regime, (mean, sd) in (("stable", (0, 0.05)), ("noisy", (0.25, 0.10)), ("global_shock", (2.0, 0.5))):
            signals, truth = oes.generate_batch(regime, range(40), c)
            self.assertAlmostEqual(signals.mean(), mean, delta=0.02)
            self.assertAlmostEqual(signals.std(), sd, delta=0.02)
            self.assertEqual(truth.all(), regime == "global_shock")
            self.assertEqual(truth.any(), regime == "global_shock")

    def test_burst_injected_into_one_block(self):
        c = oes.Config()
        signals, truth = oes.generate_batch("localized_burst", range(200), c)
        np.testing.assert_array_equal(truth.sum(axis=1), np.ones(200))
        blocks = signals.reshape(200, 16, 32).mean(axis=2)
        np.testing.assert_array_equal(blocks.argmax(axis=1), truth.argmax(axis=1))
        self.assertGreater(len(set(truth.argmax(axis=1).tolist())), 10)  # uses many blocks

    def test_burst_other_block_sizes(self):
        c = oes.Config(channels=96, block_size=12)
        signals, truth = oes.generate_batch("localized_burst", range(30), c)
        self.assertEqual(truth.shape, (30, 8))
        np.testing.assert_array_equal(truth.sum(axis=1), np.ones(30))

    def test_unknown_regime(self):
        with self.assertRaises(ValueError):
            oes.generate_signal("bogus", np.random.default_rng(0), oes.Config())
        with self.assertRaises(ValueError):
            oes.trial_seed_sequence(1, "stable", -1)


class TestMetrics(unittest.TestCase):
    def evaluate(self, regime, detected_rows, truth_rows, config=None):
        config = config or oes.Config(channels=8, block_size=1)
        scores = np.asarray(detected_rows, dtype=float)
        return oes.evaluate_batch(regime, np.arange(len(scores)), scores, np.asarray(truth_rows, bool), config,
                                  threshold=0.5)

    def test_iou_exact_and_localization(self):
        T = [0, 0, 0, 1, 0, 0, 0, 0]
        rows = [
            [0, 0, 0, 1, 0, 0, 0, 0],  # exact
            [0, 0, 0, 1, 1, 0, 0, 0],  # hit + extra
            [1, 0, 0, 0, 0, 0, 0, 0],  # miss, 3 away
            [0, 0, 0, 0, 0, 0, 0, 0],  # nothing
            [0, 0, 0, 0, 0, 0, 1, 1],  # miss, 3 away (nearest)
        ]
        e = self.evaluate("localized_burst", rows, [T] * 5)
        np.testing.assert_allclose(e.overlap, [1, 0.5, 0, 0, 0])
        np.testing.assert_array_equal(e.exact_match, [True, False, False, False, False])
        np.testing.assert_array_equal(e.event_detected, [True, True, False, False, False])
        np.testing.assert_array_equal(e.extra_blocks, [0, 1, 1, 0, 2])
        np.testing.assert_array_equal(e.localization_error, [0, 0, 3, np.nan, 3])
        s = oes.summarize(e)
        self.assertEqual(s["event_detection_rate"], 0.4)
        self.assertEqual(s["exact_match_rate"], 0.2)
        self.assertEqual(s["mean_localization_error"], 1.5)
        self.assertEqual(s["localization_error_trials"], 4)
        self.assertEqual(s["extra_block_rate"], 0.6)
        self.assertEqual(s["false_positive_rate"], None)
        self.assertEqual(s["detection_rate"], 0.8)

    def test_multi_block_truth_localization(self):
        T = [0, 1, 1, 0, 0, 0, 0, 0]
        e = self.evaluate("localized_burst", [[0, 0, 0, 0, 0, 1, 0, 0]], [T])
        self.assertEqual(e.localization_error[0], 3)

    def test_no_detection_at_all_gives_none_not_nan(self):
        T = [0, 0, 1, 0, 0, 0, 0, 0]
        e = self.evaluate("localized_burst", [[0] * 8] * 3, [T] * 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = oes.summarize(e)
        self.assertIsNone(s["mean_localization_error"])
        self.assertEqual(s["localization_error_trials"], 0)
        self.assertEqual(s["event_detection_rate"], 0.0)
        json.dumps(s, allow_nan=False)

    def test_no_event_regime(self):
        e = self.evaluate("noisy", [[0] * 8, [0, 1, 0, 0, 0, 0, 0, 0]], [[0] * 8] * 2)
        s = oes.summarize(e)
        self.assertEqual(s["false_positive_rate"], 0.5)
        self.assertEqual(s["exact_match_rate"], 0.5)
        self.assertIsNone(s["event_detection_rate"])
        self.assertIsNone(s["mean_overlap"])
        self.assertIsNone(s["mean_localization_error"])
        self.assertEqual(s["status_stable"], 1)
        self.assertEqual(s["status_localized_alert"], 1)

    def test_shock_is_not_a_false_positive(self):
        c = oes.Config(channels=4, block_size=1)
        e = self.evaluate("global_shock", [[1, 1, 1, 0], [1, 1, 1, 1]], [[1, 1, 1, 1]] * 2, c)
        s = oes.summarize(e)
        self.assertIsNone(s["false_positive_rate"])
        self.assertEqual(s["exact_match_rate"], 0.5)
        self.assertEqual(s["all_blocks_rate"], 0.5)
        self.assertEqual(s["majority_blocks_rate"], 1.0)
        self.assertEqual(s["mean_block_coverage"], 0.875)
        self.assertEqual(s["mean_overlap"], 0.875)
        self.assertEqual(s["status_global_shock"], 1)
        self.assertEqual(s["status_distributed_alert"], 1)

    def test_shape_mismatch_and_empty(self):
        with self.assertRaises(ValueError):
            oes.evaluate_batch("stable", np.arange(1), np.zeros((1, 4)), np.zeros((1, 3), bool), oes.Config())
        e = oes.evaluate_batch("stable", np.arange(0), np.zeros((0, 16)), np.zeros((0, 16), bool), oes.Config())
        with self.assertRaises(ValueError):
            oes.summarize(e)

    def test_wilson_interval(self):
        low, high = oes.wilson_interval(0, 1000)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.003826, places=5)
        self.assertEqual(oes.wilson_interval(10, 10)[1], 1.0)
        low, high = oes.wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        with self.assertRaises(ValueError):
            oes.wilson_interval(0, 0)

    def test_mean_or_none(self):
        self.assertIsNone(oes._mean_or_none(np.array([])))
        self.assertEqual(oes._mean_or_none(np.array([1, 2])), 1.5)
        self.assertIsNone(oes._round(None))


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = oes.Config()
        cls.bench = oes.run_benchmark(cls.config, SMALL)

    def test_deterministic(self):
        again = oes.run_benchmark(self.config, SMALL)
        self.assertEqual(oes.json_bytes(self.bench), oes.json_bytes(again))

    def test_no_run_metadata_in_results(self):
        text = oes.json_bytes(self.bench).decode()
        for key in ("created_utc", "platform", "elapsed", "python"):
            self.assertNotIn(key, text)

    def test_reference_behaviour(self):
        s = {row["regime"]: row for row in self.bench["summaries"]}
        self.assertEqual(s["stable"]["false_positive_rate"], 0.0)
        self.assertEqual(s["noisy"]["false_positive_rate"], 0.0)
        self.assertEqual(s["localized_burst"]["exact_match_rate"], 1.0)
        self.assertGreaterEqual(s["global_shock"]["majority_blocks_rate"], 0.99)
        self.assertTrue(all(c["passed"] for c in self.bench["expectation_checks"]))
        for trial in self.bench["trials"]:
            if trial["regime"] == "localized_burst":
                self.assertEqual(trial["detected_blocks"], trial["expected_blocks"])
                self.assertEqual(trial["event_start"] // 32, trial["expected_blocks"][0])

    def test_trial_records_consistent(self):
        trials = self.bench["trials"]
        self.assertEqual(len(trials), 4 * SMALL)
        for t in trials:
            self.assertEqual(t["detected_count"], len(t["detected_blocks"]))
            self.assertEqual(t["detected_blocks"], [i for i, x in enumerate(t["scores"]) if x >= 0.5])
            self.assertEqual(t["seed_key"], [42, oes.REGIME_KEYS[oes.Regime(t["regime"])], t["trial_index"]])

    def test_run_trial_matches_benchmark_row(self):
        for regime in oes.Regime:
            row = next(t for t in self.bench["trials"] if t["regime"] == regime.value and t["trial_index"] == 17)
            self.assertEqual(oes.run_trial(regime, 17, self.config), row)

    def test_include_trials_false(self):
        slim = oes.run_benchmark(self.config, SMALL, include_trials=False)
        self.assertNotIn("trials", slim)
        self.assertEqual(slim["summaries"], self.bench["summaries"])

    def test_invalid_trials(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                oes.run_benchmark(self.config, bad)
        with self.assertRaises(TypeError):
            oes.run_benchmark(self.config, 1.5)

    def test_other_block_size_runs(self):
        c = oes.Config(channels=128, block_size=16)
        bench = oes.run_benchmark(c, 20)
        burst = next(s for s in bench["summaries"] if s["regime"] == "localized_burst")
        self.assertEqual(burst["exact_match_rate"], 1.0)

    def test_check_expectations_failure_path(self):
        summaries = [dict(s) for s in self.bench["summaries"]]
        summaries[1]["false_positive_rate"] = 0.2
        checks = oes.check_expectations(summaries)
        failed = [c for c in checks if not c["passed"]]
        self.assertEqual([(c["regime"], c["check"]) for c in failed], [("noisy", "false_positive_rate")])


class TestSweep(unittest.TestCase):
    def test_sweep_identical_to_separate_runs(self):
        config = oes.Config()
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.9, 1.0]
        sweep = oes.threshold_sweep(thresholds, SMALL, config)
        expected = []
        for t in thresholds:
            expected.extend(oes.run_benchmark(config.with_threshold(t), SMALL, include_trials=False)["summaries"])
        self.assertEqual(sweep["rows"], expected)
        self.assertEqual(sweep["thresholds"], thresholds)
        self.assertIsNone(sweep["config"]["threshold"])

    def test_sweep_monotone_and_passing_range(self):
        sweep = oes.threshold_sweep(oes.threshold_grid(0.3, 1.0, 0.05), SMALL, oes.Config())
        noisy = [r["false_positive_rate"] for r in sweep["rows"] if r["regime"] == "noisy"]
        self.assertEqual(noisy, sorted(noisy, reverse=True))
        self.assertEqual(noisy[0], 1.0)  # threshold 0.30 flags noisy signals
        self.assertIn(0.5, sweep["thresholds_meeting_expectations"])
        self.assertNotIn(0.3, sweep["thresholds_meeting_expectations"])

    def test_threshold_grid(self):
        grid = oes.threshold_grid(0.3, 1.0, 0.05)
        self.assertEqual(len(grid), 15)
        self.assertEqual(grid[0], 0.3)
        self.assertEqual(grid[-1], 1.0)
        self.assertIn(0.35, grid)  # no float drift like 0.35000000000000003
        self.assertEqual(oes.threshold_grid(0.5, 0.5, 0.1), [0.5])
        for bad in ((0.3, 1.0, 0.0), (0.3, 1.0, -0.1), (1.0, 0.3, 0.1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oes.threshold_grid(*bad)

    def test_sweep_validation(self):
        with self.assertRaises(ValueError):
            oes.threshold_sweep([], 10, oes.Config())
        with self.assertRaises(ValueError):
            oes.threshold_sweep([0.5, float("nan")], 10, oes.Config())
        with self.assertRaises(ValueError):
            oes.threshold_sweep([0.5], 0, oes.Config())


class TestExports(unittest.TestCase):
    def test_json_is_strict_and_canonical(self):
        self.assertEqual(oes.json_bytes({"b": 1, "a": None}), b'{\n  "a": null,\n  "b": 1\n}\n')
        with self.assertRaises(ValueError):
            oes.json_bytes({"x": float("nan")})

    def test_csv_fixed_header_and_none(self):
        data = oes.csv_bytes([{"a": 1, "b": None}, {"b": 2.5}], ["a", "b"])
        self.assertEqual(data, b"a,b\n1,\n,2.5\n")
        with self.assertRaises(ValueError):
            oes.csv_bytes([{"a": 1, "zzz": 2}], ["a"])
        self.assertEqual(oes.csv_bytes([], ["a"]), b"a\n")

    def test_atomic_write_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "f.bin"
            digest = oes.atomic_write(path, b"hello")
            self.assertEqual(path.read_bytes(), b"hello")
            self.assertEqual(digest, hashlib.sha256(b"hello").hexdigest())
            self.assertEqual(oes.sha256_file(path), digest)
            oes.atomic_write(path, b"bye")
            self.assertEqual(path.read_bytes(), b"bye")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["f.bin"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o666 & ~oes._current_umask())

    def test_atomic_write_cleans_up_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(b"old")
            with mock.patch.object(oes.os, "replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    oes.atomic_write(path, b"new")
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["f.bin"])

    def test_outputs_byte_reproducible(self):
        config = oes.Config()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            hashes = []
            for out in (a, b):
                bench = oes.run_benchmark(config, SMALL)
                meta = oes.run_metadata(["run"], 0.123)
                paths = oes.write_outputs(bench, bench["summaries"], Path(out), "x", meta)
                hashes.append({k: oes.sha256_file(p) for k, p in paths.items()})
            for key in ("results", "summary", "manifest"):
                self.assertEqual(hashes[0][key], hashes[1][key], key)
            manifest = json.loads((Path(a) / "x_manifest.json").read_text())
            self.assertEqual(manifest["files"]["x_results.json"]["sha256"], hashes[0]["results"])
            self.assertEqual(manifest["files"]["x_summary.csv"]["sha256"], hashes[0]["summary"])
            meta = json.loads((Path(a) / "x_run_metadata.json").read_text())
            self.assertEqual(meta["manifest_sha256"], hashes[0]["manifest"])
            self.assertIn("created_utc", meta)
            with open(Path(a) / "x_summary.csv", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(tuple(reader.fieldnames), oes.SUMMARY_FIELDS)
                self.assertEqual(len(list(reader)), 4)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_success_and_outputs(self):
        code, out, _ = quiet_main(["run", "--trials", "20", "--out-dir", str(self.tmp), "--prefix", "r"])
        self.assertEqual(code, oes.EXIT_OK)
        self.assertIn("localized_burst", out)
        self.assertIn("[PASS]", out)
        for suffix in ("results.json", "summary.csv", "manifest.json", "run_metadata.json"):
            self.assertTrue((self.tmp / f"r_{suffix}").is_file())

    def test_run_default_command_compat(self):
        code, _, _ = quiet_main(["--trials", "5", "--out-dir", str(self.tmp), "--quiet", "--no-trials"])
        self.assertEqual(code, oes.EXIT_OK)
        results = json.loads((self.tmp / "oes_resilience_results.json").read_text())
        self.assertNotIn("trials", results)

    def test_run_strict_exit_code(self):
        args = ["run", "--trials", "10", "--out-dir", str(self.tmp), "--quiet", "--strict"]
        self.assertEqual(quiet_main(args + ["--threshold", "0.5"])[0], oes.EXIT_OK)
        code, out, _ = quiet_main(args[:-2] + ["--strict", "--threshold", "0.3"])
        self.assertEqual(code, oes.EXIT_EXPECTATIONS)
        code, out, _ = quiet_main(["run", "--trials", "10", "--out-dir", str(self.tmp), "--threshold", "0.3"])
        self.assertEqual(code, oes.EXIT_OK)
        self.assertIn("[FAIL] noisy", out)

    def test_sweep(self):
        code, out, _ = quiet_main(["sweep", "--trials", "10", "--out-dir", str(self.tmp), "--step", "0.35"])
        self.assertEqual(code, oes.EXIT_OK)
        self.assertIn("thresholds meeting", out)
        data = json.loads((self.tmp / "oes_resilience_sweep_results.json").read_text())
        self.assertEqual(data["thresholds"], [0.3, 0.65, 1.0])
        code, _, _ = quiet_main(["sweep", "--trials", "10", "--out-dir", str(self.tmp), "--quiet",
                                 "--thresholds", "0.5,0.6"])
        self.assertEqual(code, oes.EXIT_OK)
        data = json.loads((self.tmp / "oes_resilience_sweep_results.json").read_text())
        self.assertEqual(data["thresholds"], [0.5, 0.6])

    def test_usage_errors_exit_2(self):
        cases = [
            ["run", "--trials", "0"],
            ["run", "--trials", "-3"],
            ["run", "--trials", "abc"],
            ["run", "--threshold", "nan"],
            ["run", "--threshold", "inf"],
            ["run", "--threshold", "-1"],
            ["run", "--channels", "500"],
            ["run", "--seed", "-1"],
            ["run", "--global-fraction", "0"],
            ["sweep", "--step", "0"],
            ["sweep", "--start", "1.0", "--stop", "0.5"],
            ["sweep", "--thresholds", "0.5,nan"],
            ["sweep", "--thresholds", ","],
            ["run", "--bogus"],
            ["frobnicate"],
            ["--test", "extra"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                code, _, err = quiet_main(argv + ["--out-dir", str(self.tmp), "--quiet"])
                self.assertEqual(code, oes.EXIT_USAGE, err)
                self.assertTrue(err)

    def test_help_and_version_exit_0(self):
        self.assertEqual(quiet_main(["--help"])[0], 0)
        code, out, _ = quiet_main(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(oes.__version__, out)
        self.assertEqual(oes_resilience.__version__, oes.__version__)
        self.assertRegex(oes.__version__, r"^0\.6\.0")

    def test_os_error_exit_1(self):
        blocker = self.tmp / "file"
        blocker.write_text("not a directory")
        code, _, err = quiet_main(["run", "--trials", "5", "--out-dir", str(blocker / "sub"), "--quiet"])
        self.assertEqual(code, oes.EXIT_FAILURE)
        self.assertIn("error", err)

    def _write_tests(self, body):
        d = self.tmp / "t"
        d.mkdir()
        (d / "test_dummy.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                                         f"        {body}\n")
        return d

    def test_test_subcommand_pass_and_fail(self):
        d = self._write_tests("self.assertTrue(True)")
        self.assertEqual(quiet_main(["test", "--tests-dir", str(d)])[0], oes.EXIT_OK)
        d2 = self.tmp / "t2"
        d2.mkdir()
        (d2 / "test_dummy_failing.py").write_text("import unittest\nclass T(unittest.TestCase):\n"
                                                  "    def test_x(self):\n        self.fail('x')\n")
        self.assertEqual(quiet_main(["test", "--tests-dir", str(d2), "-v"])[0], oes.EXIT_FAILURE)
        self.assertEqual(quiet_main(["test", "--tests-dir", str(self.tmp / "missing")])[0], oes.EXIT_USAGE)

    def test_legacy_test_flag(self):
        with mock.patch.object(oes, "run_tests", return_value=0) as fake:
            self.assertEqual(quiet_main(["--test"])[0], 0)
        fake.assert_called_once()

    def test_subprocess_exit_codes(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        ok = subprocess.run([sys.executable, "-m", "oes_resilience", "run", "--trials", "5", "--quiet",
                             "--out-dir", str(self.tmp)],
                            capture_output=True, text=True, env=env)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        bad = subprocess.run([sys.executable, "-m", "oes_resilience", "run", "--threshold", "nan"],
                             capture_output=True, text=True, env=env)
        self.assertEqual(bad.returncode, 2)
        self.assertNotIn("Traceback", bad.stderr)

    def test_format_table(self):
        table = oes.format_table([{"regime": "x", "threshold": 0.5, "false_positive_rate": None,
                                   "detection_rate": 1.0, "exact_match_rate": 1.0, "mean_detected_blocks": 1.0,
                                   "max_maximum_score": 2.0}])
        self.assertIn("-", table.splitlines()[1])
        self.assertIn("1.000", table)


if __name__ == "__main__":
    unittest.main()
