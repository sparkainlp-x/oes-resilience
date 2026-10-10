# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the v0.2 detector API, streams and scorecard (unittest-style; runs under pytest)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import oes_resilience as pkg  # noqa: E402
from oes_resilience import core, detectors, scorecard, streams  # noqa: E402

SMALL = scorecard.CompareConfig(
    trials=40, fit_trials=60, calibration_trials=100, streams=24, calibration_streams=60, fit_streams=30,
    stream=streams.StreamConfig(steps=24, warmup=8),
)


def quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = core.main(argv)
    return code, out.getvalue(), err.getvalue()


class ConstantDetector(detectors.Detector):
    """Test plugin: scores every block with the frame's first channel value (abs)."""

    name = "test-constant"
    rule = "abs(first channel)"

    def _score_frames(self, frames):
        return np.repeat(np.abs(frames[:, :1]), self.config.blocks, axis=1)


class TestDetectionResultAndBase(unittest.TestCase):
    def setUp(self):
        self.det = detectors.OES32Detector()

    def test_shapes_for_1d_2d_3d(self):
        rng = np.random.default_rng(0)
        one = self.det.detect(rng.normal(0, 0.05, 512))
        self.assertEqual(one.scores.shape, (16,))
        self.assertEqual(one.status.shape, ())
        self.assertEqual(one.detected_blocks, [])
        batch = rng.normal(0, 0.05, (3, 512))
        batch[1, 64:96] += 1.0
        res = self.det.detect(batch)
        self.assertEqual(res.scores.shape, (3, 16))
        self.assertEqual(res.detected_blocks, [[], [2], []])
        self.assertEqual(res.status.tolist(), ["STABLE", "LOCALIZED_ALERT", "STABLE"])
        stream = rng.normal(0, 0.05, (2, 5, 512))
        self.assertEqual(self.det.detect(stream).scores.shape, (2, 5, 16))
        self.assertEqual(len(self.det.detect(stream).detected_blocks[1]), 5)

    def test_to_dict_and_explanation(self):
        x = np.zeros(512)
        x[:32] = 2.0
        res = self.det.detect(x, threshold=0.5)
        d = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(d["detected_blocks"], [0])
        self.assertEqual(d["status"], "LOCALIZED_ALERT")
        exp = d["explanation"]
        self.assertEqual(exp["top_block_last_frame"], 0)
        self.assertAlmostEqual(exp["top_score_last_frame"], 2.0)
        self.assertEqual(exp["params"]["weights"]["maximum"], 0.45)
        self.assertIn("components", exp)
        self.assertEqual(exp["detected_count"], 1)

    def test_oes32_equals_core_scores(self):
        signals, _ = core.generate_batch("localized_burst", range(50), core.Config())
        np.testing.assert_array_equal(self.det.score(signals), core.score_signals(signals, core.Config()))

    def test_validation(self):
        for bad in (np.zeros(511), np.zeros((2, 2, 2, 512)), np.zeros((0, 512))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.det.score(bad)
        x = np.zeros(512)
        x[3] = np.inf
        with self.assertRaises(ValueError):
            self.det.score(x)
        with self.assertRaises(ValueError):
            self.det.detect(np.zeros(512), threshold=float("nan"))
        with self.assertRaises(ValueError):
            detectors.OES32Detector(threshold=-1)
        with self.assertRaises(TypeError):
            detectors.OES32Detector(config="512")

    def test_unfitted_and_bad_scores(self):
        with self.assertRaises(RuntimeError):
            detectors.RobustZScoreDetector().score(np.zeros(512))

        class Negative(detectors.Detector):
            name = "negative"

            def _score_frames(self, frames):
                return -np.ones((frames.shape[0], self.config.blocks))

        with self.assertRaises(ValueError):
            Negative().score(np.zeros(512))

        class Missing(detectors.Detector):
            name = "missing"

        with self.assertRaises(NotImplementedError):
            Missing().score(np.zeros(512))

        class MissingTemporal(detectors.Detector):
            name = "missing-temporal"
            temporal = True

        with self.assertRaises(NotImplementedError):
            MissingTemporal().score(np.zeros((4, 512)))

    def test_fit_frame_detector_requires_2d(self):
        with self.assertRaises(ValueError):
            detectors.RobustZScoreDetector().fit(np.zeros(512))
        det = detectors.OES32Detector().fit(np.zeros((2, 512)))
        self.assertTrue(det.fitted)


class TestRegistry(unittest.TestCase):
    def tearDown(self):
        detectors.unregister_detector(ConstantDetector.name)

    def test_builtins_registered(self):
        self.assertEqual(
            detectors.available_detectors(),
            ["cusum", "cusum-cp", "ewma", "iforest", "maxabs", "oes32", "oes32+ewma", "oes32-robust", "zscore"]
        )
        self.assertIs(detectors.get_detector_class("oes32"), detectors.OES32Detector)
        self.assertIsInstance(detectors.create_detector("cusum", warmup=4), detectors.CUSUMDetector)

    def test_register_custom_and_errors(self):
        self.assertIs(detectors.register_detector(ConstantDetector), ConstantDetector)
        self.assertIn("test-constant", detectors.available_detectors())
        detectors.register_detector(ConstantDetector)  # same class again is fine

        class Clash(ConstantDetector):
            pass

        with self.assertRaises(ValueError):
            detectors.register_detector(Clash)
        detectors.register_detector(replace=True)(Clash)
        self.assertIs(detectors.get_detector_class("test-constant"), Clash)
        with self.assertRaises(TypeError):
            detectors.register_detector(int)

        class BadName(detectors.Detector):
            name = "Bad Name"

        with self.assertRaises(ValueError):
            detectors.register_detector(BadName)
        with self.assertRaises(ValueError):
            detectors.get_detector_class("nope")

    def test_load_plugins(self):
        good = SimpleNamespace(name="test-constant", load=lambda: ConstantDetector)

        def broken_load():
            raise ImportError("missing dependency")

        broken = SimpleNamespace(name="broken", load=broken_load)
        calls = []

        def fake_entry_points(group):
            calls.append(group)
            return [good, broken]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = detectors.load_plugins(fake_entry_points)
        self.assertEqual(loaded, ["test-constant"])
        self.assertEqual(calls, [detectors.ENTRY_POINT_GROUP])
        self.assertTrue(any("broken" in str(w.message) for w in caught))
        self.assertEqual(detectors.load_plugins(), [])  # nothing installed under the real group


class TestZScore(unittest.TestCase):
    def test_hand_computed(self):
        cfg = core.Config(channels=4, block_size=2)
        frames = np.array([[1, 1, 0, 0], [2, 2, 0, 0], [3, 3, 0, 0]], dtype=float)  # block RMS: 1,2,3 | 0,0,0
        det = detectors.RobustZScoreDetector(cfg).fit(frames)
        np.testing.assert_allclose(det.median, [2.0, 0.0])
        np.testing.assert_allclose(det.scale, [1.4826, detectors.MIN_SCALE])
        scores = det.score(np.array([4.0, 4.0, 0.0, 0.0]))
        np.testing.assert_allclose(scores, [2.0 / 1.4826, 0.0])
        self.assertEqual(det.params()["statistic"], "rms")
        self.assertAlmostEqual(det.params()["median_mean_over_blocks"], 1.0)

    def test_statistics(self):
        cfg = core.Config(channels=2, block_size=2)
        x = np.array([[3.0, -1.0]])
        for stat, expected in (("rms", math.sqrt(5)), ("mean", 1.0), ("mean_abs", 2.0)):
            det = detectors.RobustZScoreDetector(cfg, statistic=stat)
            det.fit(np.zeros((3, 2)))
            self.assertAlmostEqual(det.score(x)[0, 0], expected / detectors.MIN_SCALE, delta=1e-3 / detectors.MIN_SCALE)
        with self.assertRaises(ValueError):
            detectors.RobustZScoreDetector(statistic="max")
        self.assertEqual(detectors.RobustZScoreDetector().params(), {"statistic": "rms"})


class TestTemporal(unittest.TestCase):
    cfg = core.Config(channels=4, block_size=2)

    def stream(self):
        # 6 steps, warm-up 3; block means: block0 = [0, 1, -1, 0, 3, 3], block1 = [0, 0, 0, 0, 0, 0]
        means0 = [0.0, 1.0, -1.0, 0.0, 3.0, 3.0]
        return np.array([[m, m, 0.0, 0.0] for m in means0])

    def test_standardize_by_hand(self):
        det = detectors.EWMADetector(self.cfg, warmup=3)
        z = det.standardize(self.stream()[None])
        # warm-up: mu0 = 0, mu1 = 0; pooled sigma = sqrt((0+1+1+0+0+0) / (3*2 - 2)) = sqrt(0.5)
        sigma = math.sqrt(0.5)
        np.testing.assert_allclose(z[0, :, 0], [0, 0, 0, 0, 3 / sigma, 3 / sigma])
        np.testing.assert_allclose(z[0, :, 1], 0)

    def test_ewma_recursion(self):
        det = detectors.EWMADetector(self.cfg, warmup=3, lam=0.5)
        scores = det.score(self.stream())  # 2-D stream input
        z = 3 / math.sqrt(0.5)
        norm = math.sqrt(0.5 / 1.5)
        e4 = 0.5 * z
        e5 = 0.5 * z + 0.5 * e4
        np.testing.assert_allclose(scores[:, 0], [0, 0, 0, 0, e4 / norm, e5 / norm])
        self.assertEqual(det.params()["lam"], 0.5)

    def test_cusum_recursion(self):
        det = detectors.CUSUMDetector(self.cfg, warmup=3, k=0.5)
        scores = det.score(self.stream()[None])
        z = 3 / math.sqrt(0.5)
        np.testing.assert_allclose(scores[0, :, 0], [0, 0, 0, 0, z - 0.5, 2 * z - 1.0])
        # negative shift is caught by the lower arm
        np.testing.assert_allclose(det.score(-self.stream())[:, 0], scores[0, :, 0])
        self.assertEqual(det.params()["k"], 0.5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            detectors.EWMADetector(lam=0.0)
        with self.assertRaises(ValueError):
            detectors.EWMADetector(lam=1.5)
        with self.assertRaises(ValueError):
            detectors.CUSUMDetector(k=-1)
        with self.assertRaises(ValueError):
            detectors.CUSUMDetector(warmup=1)
        det = detectors.CUSUMDetector(self.cfg, warmup=6)
        with self.assertRaises(ValueError):
            det.score(self.stream())
        with self.assertRaises(ValueError):
            det.score(np.zeros(4))  # temporal detectors need 2-D or 3-D input

    def test_flat_warmup_does_not_divide_by_zero(self):
        det = detectors.CUSUMDetector(self.cfg, warmup=3)
        x = np.zeros((5, 4))
        self.assertTrue(np.all(det.score(x) == 0))


class TestIsolationForest(unittest.TestCase):
    def test_missing_sklearn(self):
        with mock.patch.dict(sys.modules, {"sklearn": None, "sklearn.ensemble": None}):
            self.assertFalse(detectors.sklearn_available())
            with self.assertRaises(ImportError):
                detectors.IsolationForestDetector().fit(np.zeros((4, 512)))

    @unittest.skipUnless(detectors.sklearn_available(), "scikit-learn not installed")
    def test_fit_and_score(self):
        self.assertTrue(detectors.sklearn_available())
        clean, _ = core.generate_batch("stable", range(40), core.Config(), core.PURPOSE_FIT)
        det = detectors.IsolationForestDetector(n_estimators=20).fit(clean)
        burst, truth = core.generate_batch("localized_burst", range(10), core.Config())
        scores = det.score(burst)
        self.assertEqual(scores.shape, (10, 16))
        np.testing.assert_array_equal(scores.argmax(axis=1), truth.argmax(axis=1))
        again = detectors.IsolationForestDetector(n_estimators=20).fit(clean).score(burst)
        np.testing.assert_array_equal(scores, again)
        self.assertEqual(det.params()["random_state"], 42)

    def test_params_validation(self):
        with self.assertRaises(ValueError):
            detectors.IsolationForestDetector(n_estimators=0)
        self.assertEqual(detectors.IsolationForestDetector(random_state=7).params()["random_state"], 7)


class TestStreams(unittest.TestCase):
    cfg = core.Config()

    def test_config_defaults_and_validation(self):
        sc = streams.StreamConfig()
        self.assertEqual((sc.steps, sc.warmup, sc.onset_min, sc.onset_max), (64, 16, 24, 48))
        self.assertEqual(streams.StreamConfig(steps=24, warmup=8).to_dict(),
                         {"steps": 24, "warmup": 8, "onset_min": 10, "onset_max": 20})
        for kwargs in ({"steps": 10, "warmup": 9}, {"onset_min": 5}, {"onset_min": 40, "onset_max": 30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                streams.StreamConfig(**kwargs)
        with self.assertRaises(TypeError):
            streams.StreamConfig(steps=64.0)
        self.assertEqual(streams.StreamConfig(onset_max=500).onset_max, 63)

    def test_clean_streams(self):
        sc = streams.StreamConfig(steps=20, warmup=4)
        for regime, (mean, sd) in (("stream_stable", (0, 0.05)), ("stream_noisy", (0.25, 0.10))):
            x, truth, onsets = streams.generate_streams(regime, range(10), self.cfg, sc)
            self.assertEqual(x.shape, (10, 20, 512))
            self.assertFalse(truth.any())
            self.assertTrue(np.all(onsets == -1))
            self.assertAlmostEqual(x.mean(), mean, delta=0.01)
            self.assertAlmostEqual(x.std(), sd, delta=0.01)

    def test_event_streams(self):
        sc = streams.StreamConfig()
        for regime in ("stream_burst", "stream_shift", "stream_shock"):
            x, truth, onsets = streams.generate_streams(regime, range(30), self.cfg, sc)
            self.assertTrue(np.all((onsets >= 24) & (onsets <= 48)))
            if regime == "stream_shock":
                self.assertTrue(truth.all())
            else:
                np.testing.assert_array_equal(truth.sum(axis=1), np.ones(30))
            for i in range(30):
                block_means = x[i].reshape(64, 16, 32).mean(axis=2)
                pre = block_means[: onsets[i]]
                self.assertLess(np.abs(pre).max(), 0.05)
                if regime == "stream_burst":
                    self.assertGreater(block_means[onsets[i]:, truth[i]].min(), 0.6)

    def test_shift_is_exact_constant(self):
        sc = streams.StreamConfig()
        rng = streams.stream_rng(42, "stream_shift", 3, streams.PURPOSE_STREAM_EVAL)
        x, truth, onset = streams.generate_stream("stream_shift", rng, self.cfg, sc)
        ref = streams.stream_rng(42, "stream_shift", 3, streams.PURPOSE_STREAM_EVAL)
        base = ref.normal(0.0, 0.05, (64, 512))
        block = int(ref.integers(0, 16))
        self.assertEqual(int(ref.integers(24, 49)), onset)
        self.assertTrue(truth[block])
        diff = x - base
        cols = slice(block * 32, block * 32 + 32)
        np.testing.assert_allclose(diff[onset:, cols], streams.SHIFT_MAGNITUDE, atol=1e-15)
        diff[onset:, cols] = 0
        self.assertEqual(np.abs(diff).max(), 0)

    def test_deterministic_and_index_stable(self):
        sc = streams.StreamConfig(steps=12, warmup=4)
        a = streams.generate_streams("stream_burst", range(6), self.cfg, sc)
        b = streams.generate_streams("stream_burst", range(3, 6), self.cfg, sc)
        np.testing.assert_array_equal(a[0][3:], b[0])
        c = streams.generate_streams("stream_burst", range(6), self.cfg, sc, streams.PURPOSE_STREAM_CALIBRATION)
        self.assertFalse(np.array_equal(a[0], c[0]))
        with self.assertRaises(ValueError):
            streams.generate_streams("bogus", range(1), self.cfg, sc)


class TestCalibrationAndOutcomes(unittest.TestCase):
    def test_calibrate_by_hand(self):
        values = np.arange(1, 11) / 10.0  # 0.1 .. 1.0
        cal = scorecard.calibrate_threshold({"a": values}, 0.1)
        self.assertEqual(cal["per_regime"]["a"]["allowed_exceedances"], 1)
        self.assertEqual(cal["threshold"], float(np.nextafter(0.9, np.inf)))
        self.assertEqual(cal["per_regime"]["a"]["calibration_fp_rate"], 0.1)
        zero = scorecard.calibrate_threshold({"a": values, "b": values / 2}, 0.0)
        self.assertEqual(zero["threshold"], float(np.nextafter(1.0, np.inf)))
        self.assertEqual(zero["per_regime"]["b"]["calibration_fp_rate"], 0.0)
        for bad in ({}, {"a": []}):
            with self.assertRaises(ValueError):
                scorecard.calibrate_threshold(bad, 0.1)
        with self.assertRaises(ValueError):
            scorecard.calibrate_threshold({"a": values}, 1.0)

    def test_calibration_holds_each_regime_to_target(self):
        rng = np.random.default_rng(1)
        cal = scorecard.calibrate_threshold({"low": rng.random(500), "high": rng.random(500) + 1}, 0.02)
        self.assertLessEqual(cal["per_regime"]["high"]["calibration_fp_rate"], 0.02)
        self.assertEqual(cal["per_regime"]["low"]["calibration_fp_rate"], 0.0)

    def test_stream_outcomes_by_hand(self):
        steps, blocks, warmup = 8, 3, 2
        scores = np.zeros((4, steps, blocks))
        truth = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
        onsets = np.array([4, 4, 4, 4])
        scores[0, 6, 0] = 1  # true hit, latency 2, exact
        scores[1, 3, 2] = 1  # pre-onset alarm on wrong block
        scores[1, 5, [1, 2]] = 1  # hit at 5 with an extra block -> IoU 0.5
        scores[2, 1, 0] = 1  # alarm inside warm-up: ignored -> miss
        scores[3, 4, :2] = 1  # shock: 2 of 3 blocks at onset
        out = scorecard.stream_outcomes(scores, truth, onsets, 0.5, warmup)
        np.testing.assert_array_equal(out["hit"], [True, True, False, True])
        np.testing.assert_array_equal(out["latency"], [2, 1, -1, 0])
        np.testing.assert_array_equal(out["pre_alarm"], [False, True, False, False])
        np.testing.assert_array_equal(out["exact"], [True, False, False, False])
        np.testing.assert_allclose(out["iou"], [1.0, 0.5, 0.0, 2 / 3])
        np.testing.assert_array_equal(out["any_alarm"], [True, True, False, True])

    def test_stream_row_without_hits(self):
        outcomes = {"any_alarm": np.array([False]), "pre_alarm": np.array([False]), "hit": np.array([False]),
                    "latency": np.array([-1]), "exact": np.array([False]), "iou": np.array([0.0])}
        row = scorecard._stream_row("x", streams.StreamRegime.SHIFT, 1.0, outcomes)
        self.assertEqual(row["recall"], 0.0)
        self.assertIsNone(row["latency_mean"])


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card, cls.timing = scorecard.run_compare(SMALL)

    def rows(self, **match):
        return [r for r in self.card["rows"] if all(r[k] == v for k, v in match.items())]

    def test_deterministic_hash(self):
        again, _ = scorecard.run_compare(SMALL)
        self.assertEqual(core.json_bytes(self.card), core.json_bytes(again))

    def test_reference_rows_reproduce_v01(self):
        bench = core.run_benchmark(core.Config(), SMALL.trials, include_trials=False)
        ref = {r["regime"]: r for r in self.rows(detector=scorecard.REFERENCE_LABEL)}
        self.assertEqual(len(ref), 4)
        for s in bench["summaries"]:
            row = ref[s["regime"]]
            self.assertEqual(row["threshold"], 0.5)
            self.assertEqual(row["fp_rate"], s["false_positive_rate"])
            self.assertEqual(row["exact_match_rate"], s["exact_match_rate"])
            self.assertEqual(row["mean_iou"], s["mean_overlap"])
            self.assertEqual(row["mean_detected_blocks"], s["mean_detected_blocks"])

    def test_structure(self):
        frame_detectors = {r["detector"] for r in self.rows(track="frame")}
        self.assertEqual(frame_detectors, {"oes32", "oes32@0.50", "oes32-robust", "zscore"})
        stream_detectors = {r["detector"] for r in self.rows(track="stream")}
        self.assertEqual(stream_detectors, set(scorecard.DEFAULT_DETECTORS))
        self.assertEqual(len(self.rows(track="stream")), len(scorecard.DEFAULT_DETECTORS) * 5)
        expected = {("frame", "oes32"), ("frame", "oes32-robust"), ("frame", "zscore")} | {
            ("stream", d) for d in scorecard.DEFAULT_DETECTORS
        }
        self.assertEqual({(c["track"], c["detector"]) for c in self.card["calibration"]}, expected)
        for c in self.card["calibration"]:
            for entry in c["per_regime"].values():
                self.assertLessEqual(entry["calibration_fp_rate"], SMALL.target_fp + 1e-12)
        self.assertEqual({(t["track"], t["detector"]) for t in self.timing}, expected)
        self.assertTrue(all(t["cpu_us_per_frame"] >= 0 for t in self.timing))
        json.dumps(self.card, allow_nan=False)

    def test_burst_found_by_all_on_streams(self):
        for row in self.rows(track="stream", regime="stream_burst"):
            self.assertEqual(row["recall"], 1.0, row["detector"])
            self.assertEqual(row["latency_max"], 0.0, row["detector"])

    def test_cell_formatting(self):
        self.assertEqual(scorecard._cell(None), "–")
        self.assertEqual(scorecard._cell(0.5), "0.500")
        self.assertEqual(scorecard._cell(7), "7")

    def test_markdown(self):
        md = scorecard.scorecard_markdown(self.card, self.timing)
        for text in ("## Calibration", "## Frame track", "## Stream track", "machine-dependent", "Synthetic data only"):
            self.assertIn(text, md)

    def test_single_track_and_plugin_detector(self):
        detectors.register_detector(ConstantDetector)
        try:
            cc = scorecard.CompareConfig(
                trials=10, fit_trials=10, calibration_trials=20, detectors=("test-constant", "cusum"),
                tracks=("frame",),
            )
            card, _ = scorecard.run_compare(cc)
        finally:
            detectors.unregister_detector(ConstantDetector.name)
        self.assertEqual({r["detector"] for r in card["rows"]}, {"test-constant"})

    def test_temporal_detector_that_requires_fit(self):
        class FittedTemporal(detectors.CUSUMDetector):
            name = "test-fitted-temporal"
            requires_fit = True
            seen = None

            def _fit(self, telemetry):
                FittedTemporal.seen = telemetry.shape

        detectors.register_detector(FittedTemporal)
        try:
            cc = scorecard.CompareConfig(
                streams=4, calibration_streams=6, fit_streams=6, detectors=("test-fitted-temporal",),
                tracks=("stream",), stream=streams.StreamConfig(steps=12, warmup=4),
            )
            scorecard.run_compare(cc)
        finally:
            detectors.unregister_detector(FittedTemporal.name)
        self.assertEqual(FittedTemporal.seen, (12, 12, 512))

    def test_compare_config_validation(self):
        bad = [
            {"trials": 0}, {"target_fp": 1.0}, {"target_fp": -0.1}, {"detectors": ()},
            {"detectors": ("oes32", "oes32")}, {"detectors": ("nope",)}, {"tracks": ("frame", "video")}, {"tracks": ()},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scorecard.CompareConfig(**kwargs)
        self.assertEqual(scorecard.CompareConfig().to_dict()["streams"], 1000)

    def test_package_exports(self):
        self.assertIs(pkg.OES32Detector, detectors.OES32Detector)
        self.assertIn("run_compare", pkg.__all__)
        for name in pkg.__all__:
            self.assertTrue(hasattr(pkg, name), name)


class TestCompareCLI(unittest.TestCase):
    ARGS = ["compare", "--trials", "20", "--fit-trials", "30", "--calibration-trials", "50", "--streams", "8",
            "--calibration-streams", "20", "--steps", "20", "--warmup", "6"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_compare_writes_outputs_and_verifies(self):
        code, out, _ = quiet_main(self.ARGS + ["--out-dir", str(self.tmp), "--prefix", "c", "--verify"])
        self.assertEqual(code, core.EXIT_OK)
        self.assertIn("hash match", out)
        self.assertIn("## Stream track", out)
        for suffix in ("scorecard.json", "scorecard.csv", "manifest.json", "scorecard.md", "timing.json",
                       "run_metadata.json"):
            self.assertTrue((self.tmp / f"c_{suffix}").is_file(), suffix)
        manifest = json.loads((self.tmp / "c_manifest.json").read_text())
        digest = core.sha256_file(self.tmp / "c_scorecard.json")
        self.assertEqual(manifest["files"]["c_scorecard.json"]["sha256"], digest)
        header = (self.tmp / "c_scorecard.csv").read_text().splitlines()[0]
        self.assertEqual(header, ",".join(scorecard.SCORECARD_FIELDS))
        first = core.sha256_file(self.tmp / "c_scorecard.json")
        quiet_main(self.ARGS + ["--out-dir", str(self.tmp), "--prefix", "c", "--quiet"])
        self.assertEqual(core.sha256_file(self.tmp / "c_scorecard.json"), first)

    def test_verify_mismatch_exit_4(self):
        real = scorecard.run_compare
        calls = []

        def flaky(cc):
            card, timing = real(cc)
            calls.append(1)
            if len(calls) > 1:
                card = {**card, "rows": []}
            return card, timing

        with mock.patch.object(scorecard, "run_compare", side_effect=flaky):
            code, out, _ = quiet_main(self.ARGS + ["--out-dir", str(self.tmp), "--verify", "--quiet",
                                                   "--tracks", "frame"])
        self.assertEqual(code, core.EXIT_NOT_REPRODUCIBLE)
        self.assertIn("HASH MISMATCH", out)

    def test_plugins_flag(self):
        with mock.patch.object(scorecard, "load_plugins", return_value=[]) as fake:
            code, _, _ = quiet_main(self.ARGS + ["--out-dir", str(self.tmp), "--quiet", "--plugins",
                                                 "--tracks", "frame", "--detectors", "oes32"])
        self.assertEqual(code, core.EXIT_OK)
        fake.assert_called_once()

    def test_usage_errors(self):
        for extra in (["--detectors", "nope"], ["--target-fp", "1"], ["--warmup", "30"], ["--tracks", "video"],
                      ["--streams", "0"], ["--detectors", ","]):
            with self.subTest(extra=extra):
                code, _, err = quiet_main(self.ARGS + extra + ["--out-dir", str(self.tmp), "--quiet"])
                self.assertEqual(code, core.EXIT_USAGE, err)


if __name__ == "__main__":
    unittest.main()
