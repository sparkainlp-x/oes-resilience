# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for v0.3: missing-data scoring, the oes32+ewma hybrid and the stress suite."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import oes_resilience as pkg  # noqa: E402
from oes_resilience import core, detectors, scorecard, streams, stress  # noqa: E402

CFG = core.Config()
STREAM = streams.StreamConfig(steps=24, warmup=8)
SMALL_CC = scorecard.CompareConfig(
    trials=30, fit_trials=40, calibration_trials=60, streams=20, calibration_streams=40, fit_streams=20,
    stream=STREAM, detectors=stress.DEFAULT_STRESS_DETECTORS,
)
SMALL = stress.StressConfig(compare=SMALL_CC)


def quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = core.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestMissingData(unittest.TestCase):
    def frame_with_nan(self):
        x = np.zeros(512)
        x[:32] = np.arange(32) / 100.0
        x[0:4] = np.nan  # block 0: 28 observed channels
        x[32:64] = np.nan  # block 1: fully missing
        x[64:96] = 0.5
        x[70] = np.nan
        return x

    def test_masked_block_stats_by_hand(self):
        stats = detectors.masked_block_stats(self.frame_with_nan().reshape(16, 32))
        observed = np.arange(4, 32) / 100.0
        self.assertEqual(stats["count"][0], 28)
        self.assertEqual(stats["count"][1], 0)
        self.assertEqual(stats["count"][2], 31)
        self.assertAlmostEqual(stats["max_abs"][0], 0.31)
        self.assertAlmostEqual(stats["rms"][0], math.sqrt(np.mean(observed**2)))
        self.assertAlmostEqual(stats["mean_abs"][0], observed.mean())
        self.assertAlmostEqual(stats["mean"][2], 0.5)
        self.assertEqual(stats["max_abs"][1], 0.0)

    def test_oes32_masked_by_hand_and_fast_path(self):
        x = self.frame_with_nan()
        scores = detectors.OES32Detector().score(x)
        observed = np.arange(4, 32) / 100.0
        expected = 0.45 * 0.31 + 0.35 * math.sqrt(np.mean(observed**2)) + 0.20 * observed.mean()
        self.assertAlmostEqual(scores[0], expected)
        self.assertEqual(scores[1], 0.0)  # fully missing block scores 0 (documented)
        self.assertAlmostEqual(scores[2], 0.5)
        complete = np.random.default_rng(0).normal(0, 0.05, (5, 512))
        np.testing.assert_array_equal(detectors.OES32Detector().score(complete), core.score_signals(complete, CFG))

    def test_zscore_masked(self):
        rng = np.random.default_rng(1)
        det = detectors.RobustZScoreDetector().fit(rng.normal(0, 0.05, (50, 512)))
        x = rng.normal(0, 0.05, (3, 512))
        full = det.score(x)
        x[0, 32:64] = np.nan
        x[1, 0] = np.nan
        masked = det.score(x)
        self.assertEqual(masked[0, 1], 0.0)
        np.testing.assert_allclose(masked[2], full[2])
        stat = np.sqrt(np.mean(x[1, 1:32] ** 2))
        self.assertAlmostEqual(masked[1, 0], abs(stat - np.ravel(det.median)[0]) / np.ravel(det.scale)[0])

    def test_temporal_masked_matches_fast_path_on_complete_data(self):
        x, _, _ = streams.generate_streams("stream_shift", range(3), CFG, STREAM)
        for cls in (detectors.EWMADetector, detectors.CUSUMDetector):
            det = cls(warmup=8)
            np.testing.assert_allclose(det._standardize_masked(x), det.standardize(x), atol=1e-12)

    def test_temporal_masked_weights_and_missing(self):
        x = np.zeros((1, 12, 64))
        rng = np.random.default_rng(2)
        x[0, :, :] = rng.normal(0, 0.05, (12, 64))
        x[0, :, 0:16] = np.nan  # block 0 half observed at every step
        x[0, 10, 32:64] = np.nan  # block 1 fully missing at step 10
        det = detectors.EWMADetector(core.Config(channels=64), warmup=4)
        z = det.standardize(x)
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(z[0, 10, 1], 0.0)
        self.assertTrue((z[0, :4] == 0).all())
        scores = det.score(x)
        self.assertEqual(scores.shape, (1, 12, 2))

    def test_unsupported_and_invalid_inputs(self):
        class NoMissing(detectors.OES32Detector):
            supports_missing = False

        x = np.zeros(512)
        x[3] = np.nan
        with self.assertRaisesRegex(ValueError, "does not support missing"):
            NoMissing().score(x)
        with self.assertRaisesRegex(ValueError, "infinite"):
            detectors.OES32Detector().score(np.full(512, np.inf))
        with self.assertRaisesRegex(ValueError, "fit data must be complete"):
            detectors.RobustZScoreDetector().fit(np.tile(x, (3, 1)))
        self.assertFalse(detectors.IsolationForestDetector.supports_missing)
        for name in ("oes32", "zscore", "ewma", "cusum", "oes32+ewma"):
            self.assertTrue(detectors.get_detector_class(name).supports_missing, name)


class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean = np.concatenate(
            [streams.generate_streams(r, range(30), CFG, STREAM, streams.PURPOSE_STREAM_FIT)[0]
             for r in streams.CLEAN_STREAM_REGIMES]
        )
        cls.clean = clean
        cls.det = detectors.create_detector("oes32+ewma", warmup=8).fit(clean)

    def test_registered_with_plus_in_name(self):
        self.assertIn("oes32+ewma", detectors.available_detectors())
        self.assertIs(detectors.get_detector_class("oes32+ewma"), detectors.OES32EWMAHybridDetector)
        with self.assertRaises(ValueError):
            detectors.get_detector_class("Bad Name")

    def test_score_is_max_of_normalised_parts(self):
        x, _, _ = streams.generate_streams("stream_burst", range(4), CFG, STREAM)
        oes = detectors.OES32Detector().score(x)
        oes[:, :8] = 0.0
        ewma = detectors.EWMADetector(warmup=8).score(x)
        s = self.det.scales
        np.testing.assert_allclose(self.det.score(x), np.maximum(oes / s["oes32"], ewma / s["ewma"]))
        self.assertTrue((self.det.score(x)[:, :8] == 0).all())

    def test_scales_are_clean_quantiles(self):
        oes = detectors.OES32Detector().score(self.clean)[:, 8:].max(axis=(1, 2))
        self.assertAlmostEqual(self.det.scales["oes32"], float(np.quantile(oes, 0.99)))
        params = self.det.params()
        self.assertEqual(params["balance_quantile"], 0.99)
        self.assertIn("scales", params)
        self.assertNotIn("scales", detectors.create_detector("oes32+ewma").params())

    def test_detect_and_explain(self):
        x, _, _ = streams.generate_streams("stream_shock", range(1), CFG, STREAM)
        result = self.det.detect(x[0])
        self.assertIn("parts", result.explanation)
        self.assertEqual(result.threshold, 1.0)

    def test_validation_and_unfitted(self):
        for bad in (0.0, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                detectors.create_detector("oes32+ewma", balance_quantile=bad)
        with self.assertRaises(RuntimeError):
            detectors.create_detector("oes32+ewma").score(self.clean[:1])

    def test_fit_single_stream(self):
        det = detectors.create_detector("oes32+ewma", warmup=8).fit(self.clean[0])
        self.assertGreater(det.scales["ewma"], 0)


class TestScenarios(unittest.TestCase):
    P = stress.StressParams()

    def draw(self, name, index=0, track="stream"):
        x, truth, onsets = stress.generate_stress_batch(name, track, [index], CFG, STREAM, self.P)
        return x[0], truth[0], int(onsets[0])

    def test_registry(self):
        self.assertEqual(len(stress.SCENARIOS), len(set(s.key for s in stress.SCENARIOS)))
        self.assertEqual(stress.get_scenario("drift").tracks, ("stream",))
        self.assertIs(stress.get_scenario(stress.SCENARIOS[0]), stress.SCENARIOS[0])
        with self.assertRaisesRegex(ValueError, "Unknown stress scenario"):
            stress.get_scenario("nope")
        with self.assertRaisesRegex(ValueError, "Unknown track"):
            stress.generate_stress_batch("clean", "video", [0], CFG, STREAM, self.P)

    def test_background_scenarios(self):
        for name in ("clean", "heavy_tail", "dropout", "saturated", "global_shift"):
            x, truth, onset = self.draw(name)
            self.assertEqual(onset, -1, name)
            self.assertFalse(truth.any(), name)
        x, _, _ = self.draw("dropout")
        self.assertEqual(int(np.isnan(x[0]).sum()), round(0.10 * 512))
        self.assertTrue((np.isnan(x).all(axis=0) == np.isnan(x[0])).all())  # whole-stream dropout
        x, _, _ = self.draw("saturated")
        self.assertEqual(int((x == self.P.clip_level).all(axis=0).sum()), round(0.02 * 512))
        x, _, _ = self.draw("global_shift")
        self.assertAlmostEqual(float(x.mean()), 0.25, delta=0.01)

    def test_student_t_scale(self):
        self.assertAlmostEqual(self.P.t_scale(), 0.05 * math.sqrt(1 / 3))
        self.assertEqual(stress.StressParams(t_df=1.5).t_scale(), 0.05)
        x, _, _ = stress.generate_stress_batch("heavy_tail", "stream", range(20), CFG, STREAM, self.P)
        self.assertAlmostEqual(float(x.std()), 0.05, delta=0.01)

    def test_event_truths(self):
        for name in ("clean_burst", "heavy_tail_burst", "dropout_burst", "clipped_burst", "narrow_burst",
                     "global_shift_burst", "drift"):
            _, truth, onset = self.draw(name, 3)
            self.assertEqual(int(truth.sum()), 1, name)
            self.assertTrue(STREAM.onset_min <= onset <= STREAM.onset_max, name)
        _, truth, _ = self.draw("cross_block_burst", 5)
        blocks = np.flatnonzero(truth)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1] - blocks[0], 1)
        _, truth, _ = self.draw("correlated_burst", 7)
        blocks = np.flatnonzero(truth)
        self.assertEqual(list(blocks), list(range(blocks[0], blocks[0] + 3)))
        _, truth, _ = self.draw("baseline_step")
        self.assertTrue(truth.all())

    def test_event_values(self):
        x, truth, onset = self.draw("clipped_burst", 2)
        self.assertLessEqual(float(np.abs(x).max()), self.P.clip_level)
        x, truth, onset = self.draw("narrow_burst", 4)
        block = int(np.flatnonzero(truth)[0])
        active = x[onset:, block * 32:(block + 1) * 32].mean(axis=0) > 0.4
        self.assertEqual(int(active.sum()), self.P.narrow_width)
        self.assertLess(float(np.abs(x[:onset]).max()), 0.4)
        x, truth, onset = self.draw("cross_block_burst", 1)
        first = int(np.flatnonzero(truth)[0])
        active = np.flatnonzero(x[onset:].mean(axis=0) > 0.4)
        self.assertEqual((active[0], active[-1]), (first * 32 + 16, first * 32 + 47))
        x, truth, onset = self.draw("correlated_burst", 3)
        cols = np.flatnonzero(np.repeat(truth, 32))
        common = x[onset:, cols].mean(axis=1)
        self.assertGreater(float(np.corrcoef(x[onset:, cols[0]], x[onset:, cols[-1]])[0, 1]), 0.5)
        self.assertAlmostEqual(float(common.mean()), 0.6, delta=0.1)

    def test_drift_and_step_exact(self):
        # Same generator state with the deterministic part switched off differs only by that part.
        def pair(name, **off):
            rng = (stress.stress_rng(42, "stream", name, 0) for _ in range(2))
            onset_range = (STREAM.onset_min, STREAM.onset_max)
            on = stress.generate_stress(name, next(rng), CFG, STREAM.steps, onset_range, self.P)
            zero = stress.generate_stress(name, next(rng), CFG, STREAM.steps, onset_range, stress.StressParams(**off))
            return on, zero

        (x, truth, onset), (x0, _, _) = pair("drift", drift_slope=0.0)
        block = int(np.flatnonzero(truth)[0])
        diff = x - x0
        ramp = self.P.drift_slope * np.arange(1, STREAM.steps - onset + 1)
        np.testing.assert_allclose(diff[onset:, block * 32:(block + 1) * 32], np.repeat(ramp[:, None], 32, 1))
        diff[onset:, block * 32:(block + 1) * 32] = 0.0
        self.assertEqual(float(np.abs(diff).max()), 0.0)
        (x, _, onset), (x0, _, _) = pair("baseline_step", step_size=0.0)
        np.testing.assert_allclose((x - x0)[onset:], 0.10)
        self.assertEqual(float(np.abs(x - x0)[:onset].max()), 0.0)

    def test_frame_track(self):
        x, truth, onsets = stress.generate_stress_batch("clean_burst", "frame", range(3), CFG, STREAM, self.P)
        self.assertEqual(x.shape, (3, 1, 512))
        self.assertTrue((onsets == 0).all())
        self.assertTrue((truth.sum(axis=1) == 1).all())

    def test_deterministic_and_disjoint(self):
        a = stress.generate_stress_batch("heavy_tail_burst", "stream", [5], CFG, STREAM, self.P)
        b = stress.generate_stress_batch("heavy_tail_burst", "stream", [2, 5], CFG, STREAM, self.P)
        np.testing.assert_array_equal(a[0][0], b[0][1])
        c = stress.generate_stress_batch("clean", "stream", [5], CFG, STREAM, self.P)
        self.assertFalse(np.array_equal(a[0], c[0]))

    def test_generation_errors(self):
        tiny = core.Config(channels=32, block_size=32)
        with self.assertRaisesRegex(ValueError, "cross_block_burst"):
            stress.generate_stress_batch("cross_block_burst", "frame", [0], tiny, STREAM, self.P)
        with self.assertRaisesRegex(ValueError, "narrow_width"):
            stress.generate_stress_batch("narrow_burst", "frame", [0], CFG, STREAM,
                                         stress.StressParams(narrow_width=40))
        with self.assertRaisesRegex(ValueError, "correlated_blocks"):
            stress.generate_stress_batch("correlated_burst", "frame", [0], CFG, STREAM,
                                         stress.StressParams(correlated_blocks=17))

    def test_params_validation(self):
        for kwargs in ({"t_df": 0}, {"dropout_fraction": 1.0}, {"saturated_fraction": -0.1}, {"clip_level": 0},
                       {"narrow_width": 0}, {"correlated_blocks": 0}, {"drift_slope": float("nan")},
                       {"correlated_amplitude": (0.6, -1.0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stress.StressParams(**kwargs)
        self.assertEqual(stress.StressParams().to_dict()["t_df"], 3.0)


class TestRobustness(unittest.TestCase):
    def bg(self, fp, low, high, supported=True):
        return {"kind": "background", "supported": supported, "fp_rate": fp, "fp_ci95_low": low, "fp_ci95_high": high}

    def ev(self, recall, low, high, latency=0.0):
        return {"kind": "event", "supported": True, "recall": recall, "recall_ci95_low": low,
                "recall_ci95_high": high, "latency_median": latency}

    def test_background(self):
        out = stress.robustness(self.bg(0.2, 0.15, 0.25), self.bg(0.01, 0.005, 0.02))
        self.assertEqual((out["fp_inflation"], out["change"], out["ref_fp_rate"]), (0.19, "worse", 0.01))
        out = stress.robustness(self.bg(0.0, 0.0, 0.004), self.bg(0.2, 0.15, 0.25))
        self.assertEqual(out["change"], "better")
        out = stress.robustness(self.bg(0.012, 0.007, 0.02), self.bg(0.01, 0.005, 0.02))
        self.assertEqual(out["change"], "n.s.")

    def test_event(self):
        out = stress.robustness(self.ev(0.5, 0.45, 0.55, 3.0), self.ev(1.0, 0.99, 1.0, 1.0))
        self.assertEqual((out["recall_retention"], out["latency_delta"], out["change"]), (0.5, 2.0, "worse"))
        out = stress.robustness(self.ev(0.9, 0.85, 0.95, None), self.ev(0.0, 0.0, 0.01, None))
        self.assertIsNone(out["recall_retention"])
        self.assertNotIn("latency_delta", out)
        self.assertEqual(out["change"], "better")

    def test_no_reference_or_unsupported(self):
        self.assertEqual(stress.robustness(self.bg(0.1, 0.0, 0.2), None), {"change": None})
        self.assertEqual(stress.robustness(self.bg(0.1, 0.0, 0.2, False), self.bg(0, 0, 0)), {"change": None})

    def test_rate_uses_wilson(self):
        rate, low, high = stress._rate(5, 100)
        w = core.wilson_interval(5, 100)
        self.assertEqual((rate, low, high), (0.05, round(w[0], 12), round(w[1], 12)))


class TestRunStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card, cls.timing = stress.run_stress(SMALL)

    def rows(self, **match):
        return [r for r in self.card["rows"] if all(r[k] == v for k, v in match.items())]

    def test_deterministic_hash(self):
        again, _ = stress.run_stress(SMALL)
        self.assertEqual(core.json_bytes(self.card), core.json_bytes(again))

    def test_structure(self):
        frame = self.rows(track="frame")
        self.assertEqual({r["detector"] for r in frame}, {"oes32", "zscore"})
        self.assertEqual({r["scenario"] for r in frame}, {s.name for s in stress.SCENARIOS if "frame" in s.tracks})
        stream_rows = self.rows(track="stream")
        self.assertEqual(len(stream_rows), len(stress.SCENARIOS) * len(stress.DEFAULT_STRESS_DETECTORS))
        for r in self.card["rows"]:
            self.assertEqual(set(r), set(stress.STRESS_FIELDS))
            self.assertTrue(r["supported"])
        json.dumps(self.card, allow_nan=False)
        self.assertEqual(self.card["kind"], "stress_scorecard")

    def test_thresholds_match_compare(self):
        cc = SMALL_CC
        card, _ = scorecard.run_compare(cc)
        compare_thr = {(c["track"], c["detector"]): c["threshold"] for c in card["calibration"]}
        stress_thr = {(c["track"], c["detector"]): c["threshold"] for c in self.card["calibration"]}
        self.assertEqual(compare_thr, stress_thr)

    def test_reference_rows_and_robustness(self):
        for r in self.rows(scenario="heavy_tail"):
            ref = self.rows(track=r["track"], detector=r["detector"], scenario="clean")[0]
            self.assertAlmostEqual(r["fp_inflation"], round(r["fp_rate"] - ref["fp_rate"], 12))
        for r in self.rows(scenario="clean_burst"):
            self.assertIsNone(r["recall_retention"])
            self.assertIsNone(r["change"])
        for r in self.rows(scenario="drift"):
            self.assertIsNone(r["reference"])
        for r in self.rows(scenario="narrow_burst"):
            self.assertIsNotNone(r["recall_retention"])

    def test_frame_detectors_catch_default_burst(self):
        for r in self.rows(track="frame", scenario="clean_burst"):
            self.assertEqual(r["recall"], 1.0)

    def test_unsupported_detector_rows(self):
        class NoMissing(detectors.OES32Detector):
            name = "test-no-missing"
            supports_missing = False

        detectors.register_detector(NoMissing)
        try:
            cc = scorecard.CompareConfig(trials=10, calibration_trials=20, detectors=("test-no-missing",),
                                         tracks=("frame",))
            card, _ = stress.run_stress(stress.StressConfig(compare=cc, scenarios=("dropout", "dropout_burst")))
        finally:
            detectors.unregister_detector(NoMissing.name)
        by = {r["scenario"]: r for r in card["rows"]}
        self.assertEqual(set(by), {"clean", "clean_burst", "dropout", "dropout_burst"})
        self.assertFalse(by["dropout"]["supported"])
        self.assertIsNone(by["dropout"]["fp_rate"])
        self.assertTrue(by["clean"]["supported"])
        md = stress.stress_markdown(card, [])
        self.assertIn("unsupported (missing data)", md)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            stress.StressConfig(scenarios=())
        with self.assertRaises(ValueError):
            stress.StressConfig(scenarios=("drift", "drift"))
        with self.assertRaises(ValueError):
            stress.StressConfig(scenarios=("nope",))
        sc = stress.StressConfig(scenarios=("narrow_burst",))
        self.assertEqual(sc.scenarios, ("clean_burst", "narrow_burst"))
        self.assertEqual(sc.to_dict()["scenarios"], ["clean_burst", "narrow_burst"])

    def test_markdown(self):
        md = stress.stress_markdown(self.card, self.timing)
        for text in ("stress scorecard", "Synthetic data only", "background scenarios", "event scenarios",
                     "CPU time", "retention"):
            self.assertIn(text, md)

    def test_package_exports(self):
        self.assertIs(pkg.run_stress, stress.run_stress)
        self.assertIs(pkg.StressConfig, stress.StressConfig)
        self.assertIs(pkg.OES32EWMAHybridDetector, detectors.OES32EWMAHybridDetector)


class TestStressCLI(unittest.TestCase):
    ARGS = ["stress", "--trials", "10", "--fit-trials", "20", "--calibration-trials", "20", "--streams", "6",
            "--calibration-streams", "10", "--fit-streams", "6", "--steps", "12", "--warmup", "4",
            "--scenarios", "heavy_tail,narrow_burst,drift", "--quiet"]

    def test_writes_outputs_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = quiet_main([*self.ARGS, "--out-dir", tmp, "--prefix", "s", "--verify"])
            self.assertEqual(code, 0, err)
            self.assertIn("hash match", out)
            names = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(names, ["s_manifest.json", "s_run_metadata.json", "s_scorecard.csv",
                                     "s_scorecard.json", "s_scorecard.md", "s_timing.json"])
            header = (Path(tmp) / "s_scorecard.csv").read_text().splitlines()[0]
            self.assertEqual(header, ",".join(stress.STRESS_FIELDS))
            manifest = json.loads((Path(tmp) / "s_manifest.json").read_text())
            self.assertEqual(manifest["files"]["s_scorecard.csv"]["sha256"],
                             core.sha256_file(Path(tmp) / "s_scorecard.csv"))

    def test_parameters_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [*self.ARGS, "--out-dir", tmp, "--t-df", "5", "--burst-mean", "0.3", "--correlated-mean", "0.3",
                    "--dropout-fraction", "0.2", "--tracks", "stream", "--detectors", "ewma"]
            code, _, err = quiet_main(argv)
            self.assertEqual(code, 0, err)
            card = json.loads((Path(tmp) / "oes_resilience_stress_scorecard.json").read_text())
            params = card["stress"]["params"]
            self.assertEqual((params["t_df"], params["burst_mean"], params["dropout_fraction"]), (5.0, 0.3, 0.2))
            self.assertEqual(params["correlated_amplitude"], [0.3, 0.1])

    def test_console_output_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [a for a in self.ARGS if a != "--quiet"] + ["--out-dir", tmp]
            code, out, _ = quiet_main(argv)
            self.assertEqual(code, 0)
            self.assertIn("stress scorecard", out)
            real = stress.run_stress
            calls = []

            def flaky(sc):
                card, timing = real(sc)
                calls.append(1)
                if len(calls) == 2:
                    card = {**card, "rows": []}
                return card, timing

            with mock.patch.object(stress, "run_stress", flaky):
                code, out, _ = quiet_main([*self.ARGS, "--out-dir", tmp, "--verify"])
            self.assertEqual(code, core.EXIT_NOT_REPRODUCIBLE)
            self.assertIn("HASH MISMATCH", out)

    def test_plugins_flag_and_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(stress, "load_plugins") as loader:
            code, _, _ = quiet_main([*self.ARGS, "--out-dir", tmp, "--plugins", "--tracks", "frame"])
            self.assertEqual(code, 0)
            loader.assert_called_once()
        for extra in (["--scenarios", "nope"], ["--t-df", "0"], ["--detectors", "nope"], ["--streams", "0"]):
            with self.subTest(extra=extra):
                code, _, _ = quiet_main([*self.ARGS, *extra])
                self.assertEqual(code, core.EXIT_USAGE)

    def test_stress_listed_in_help(self):
        code, out, _ = quiet_main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("stress", out)
        self.assertIn("stress", core.COMMANDS)


if __name__ == "__main__":
    unittest.main()
