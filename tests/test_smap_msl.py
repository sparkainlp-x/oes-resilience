# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for v0.5: the NASA SMAP/MSL adapter and evaluation, on tiny synthetic fixtures only.

CI never downloads the real dataset; every file here is generated in a temporary directory.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oes_resilience import core, smap_msl  # noqa: E402
from oes_resilience.core import Config, score_signals  # noqa: E402
from oes_resilience.detectors import CUSUMDetector, EWMADetector  # noqa: E402
from oes_resilience.smap_msl import (  # noqa: E402
    DataError,
    alarm_runs,
    channel_metrics,
    cusum_scores,
    ewma_scores,
    holm,
    load_labels,
    merge_windows,
    method_scores,
    prf,
    standardize,
    trailing_windows,
    wilcoxon_signed_rank,
)

PROTOCOL = ROOT / "reports" / "smap_msl_protocol.json"
LABELS_HEADER = "chan_id,spacecraft,anomaly_sequences,class,num_values\n"


def make_fixture(root: Path, seed: int = 7) -> dict[str, np.ndarray]:
    """Tiny SMAP/MSL-shaped dataset: 3 SMAP + 2 MSL channels, one duplicated label row, one flat channel."""
    rng = np.random.default_rng(seed)
    spec = {  # chan: (spacecraft, width, n_train, n_test, windows)
        "A-1": ("SMAP", 25, 300, 400, [[100, 140], [300, 320]]),
        "A-2": ("SMAP", 25, 250, 350, [[200, 260]]),
        "D-2": ("SMAP", 25, 200, 300, [[150, 170]]),  # constant train
        "M-1": ("MSL", 55, 280, 360, [[50, 80]]),
        "M-2": ("MSL", 55, 260, 300, [[220, 240]]),
    }
    arrays = {}
    lines = [LABELS_HEADER]
    for chan, (craft, width, n_train, n_test, windows) in spec.items():
        train = np.zeros((n_train, width))
        test = np.zeros((n_test, width))
        if chan != "D-2":
            train[:, 0] = 0.1 * rng.standard_normal(n_train)
            test[:, 0] = 0.1 * rng.standard_normal(n_test)
        for start, end in windows:
            test[start : end + 1, 0] += 0.8
        arrays[chan] = test[:, 0]
        for split, data in (("train", train), ("test", test)):
            (root / split).mkdir(parents=True, exist_ok=True)
            np.save(root / split / f"{chan}.npy", data)
        classes = "[" + ", ".join("point" for _ in windows) + "]"
        lines.append(f'{chan},{craft},"{json.dumps(windows)}","{classes}",{n_test}\n')
    lines.append('A-2,SMAP,"[[255, 270]]",[contextual],350\n')  # duplicate row, overlapping window
    (root / "labeled_anomalies.csv").write_text("".join(lines), encoding="utf-8")
    return arrays


def small_protocol(directory: Path, resamples: int = 300) -> Path:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["uncertainty"]["bootstrap_resamples"] = resamples
    path = directory / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = core.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class LabelTests(unittest.TestCase):
    def test_merge_windows(self):
        events, classes = merge_windows([(10, 20, "point"), (5, 8, "point"), (21, 30, "contextual"), (40, 41, "x")])
        self.assertEqual(events, ((5, 8), (10, 30), (40, 41)))
        self.assertEqual(classes, ("point", "contextual+point", "x"))

    def test_load_labels_pools_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fixture(Path(tmp))
            channels = {c.chan_id: c for c in load_labels(Path(tmp) / "labeled_anomalies.csv")}
        self.assertEqual(len(channels), 5)
        self.assertEqual(channels["A-2"].events, ((200, 270),))
        self.assertEqual(channels["A-2"].label_rows, 2)
        self.assertEqual(channels["A-1"].events, ((100, 140), (300, 320)))
        self.assertEqual(channels["M-1"].spacecraft, "MSL")

    def test_load_labels_rejects_bad_rows(self):
        bad_rows = [
            'A-1,SMAP,"[[5, 3]]",[point],10\n',
            'A-1,SMAP,"[[5, 30]]",[point],10\n',
            'A-1,VOYAGER,"[[1, 3]]",[point],10\n',
            'A-1,SMAP,"[[1, 3]]","[point, point]",10\n',
            'A-1,SMAP,"not a list",[point],10\n',
            'A-1,SMAP,"[[1, 3]]",[point],ten\n',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            for row in bad_rows:
                path.write_text(LABELS_HEADER + row, encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(DataError):
                    load_labels(path)
            path.write_text(LABELS_HEADER + 'A-1,SMAP,"[[1, 3]]",[point],10\nA-1,SMAP,"[[5, 6]]",[point],11\n',
                            encoding="utf-8")
            with self.assertRaises(DataError):
                load_labels(path)

    def test_load_series_checks_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fixture(Path(tmp))
            channel = smap_msl.Channel("SMAP", "A-1", ((1, 2),), ("point",), 999, 1)
            with self.assertRaises(DataError):
                smap_msl.load_series(tmp, channel)
            np.save(Path(tmp) / "train" / "A-1.npy", np.array([1.0, np.nan]))
            with self.assertRaises(DataError):
                smap_msl.load_series(tmp, channel)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.train = rng.standard_normal(200) * 0.2 + 1.0
        self.test = rng.standard_normal(150) * 0.2 + 1.0
        self.test[60:80] += 2.0
        self.params = json.loads(PROTOCOL.read_text(encoding="utf-8"))["methods"]

    def test_standardize_uses_train_only(self):
        (z_train, z_test), mu, sigma = standardize(self.train, self.test)
        self.assertAlmostEqual(mu, float(np.mean(self.train)))
        self.assertAlmostEqual(sigma, float(np.std(self.train, ddof=1)))
        self.assertAlmostEqual(float(np.mean(z_train)), 0.0, places=12)
        (z_flat, z_other), _, sigma_flat = standardize(np.ones(10), np.array([1.0, 2.0]))
        self.assertEqual(sigma_flat, smap_msl.MIN_SCALE)
        self.assertTrue(np.all(z_flat == 0.0))
        self.assertEqual(z_other[0], 0.0)
        self.assertGreater(z_other[1], 1e8)

    def test_trailing_windows(self):
        windows = trailing_windows(np.arange(5.0), 3)
        self.assertEqual(windows.shape, (5, 3))
        np.testing.assert_array_equal(windows[4], [2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(windows[0, :2]).all())
        self.assertEqual(windows[0, 2], 0.0)

    def test_oes32_matches_reference_formula(self):
        (z_train,), _, _ = standardize(self.train)
        scores = method_scores("oes32", z_train, self.params["oes32"])
        frames = trailing_windows(z_train, 32)
        np.testing.assert_array_equal(scores[31:], score_signals(frames[31:], Config(channels=32, block_size=32))[:, 0])
        first = np.abs(z_train[:3])  # partial window of 3 observed samples, mask-aware
        expected = 0.45 * first.max() + 0.35 * np.sqrt(np.mean(first**2)) + 0.20 * first.mean()
        self.assertAlmostEqual(scores[2], expected, places=12)
        maxabs = method_scores("maxabs", z_train, self.params["maxabs"])
        self.assertAlmostEqual(maxabs[40], float(np.abs(z_train[9:41]).max()))
        self.assertEqual(method_scores("oes32", z_train[:10], self.params["oes32"]).shape, (10,))

    def test_zscore_is_pointwise(self):
        z = np.array([-2.0, 0.5, 3.0])
        np.testing.assert_array_equal(method_scores("zscore", z, {}), [2.0, 0.5, 3.0])
        with self.assertRaises(ValueError):
            method_scores("nope", z, {})

    def test_recursions_match_builtin_detectors(self):
        """With warm-up = train, the built-in EWMA/CUSUM give the same test scores as the adapter."""
        stream = np.concatenate([self.train, self.test])[:, None]
        (_, z_test), _, _ = standardize(self.train, self.test)
        config = Config(channels=1, block_size=1)
        ewma = EWMADetector(config, warmup=self.train.size, lam=0.2).score(stream)[self.train.size :, 0]
        cusum = CUSUMDetector(config, warmup=self.train.size, k=0.5).score(stream)[self.train.size :, 0]
        np.testing.assert_allclose(ewma_scores(z_test, 0.2), ewma, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(cusum_scores(z_test, 0.5), cusum, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(method_scores("ewma", z_test, self.params["ewma"]), ewma_scores(z_test, 0.2))
        np.testing.assert_array_equal(method_scores("cusum", z_test, self.params["cusum"]), cusum_scores(z_test, 0.5))


class MetricTests(unittest.TestCase):
    def test_alarm_runs(self):
        self.assertEqual(alarm_runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)), [(1, 2), (4, 4), (7, 7)])
        self.assertEqual(alarm_runs(np.zeros(4, dtype=bool)), [])

    def test_channel_metrics_worked_example(self):
        alarms = np.zeros(30, dtype=bool)
        alarms[[2, 3, 12, 13, 14, 25]] = True  # run 2-3 FP, run 12-14 inside event 1, 25 FP
        events = [(10, 15), (18, 22)]
        m = channel_metrics(alarms, events)
        self.assertEqual((m["tp_events"], m["fp_runs"], m["n_events"]), (1, 2, 2))
        self.assertEqual(m["detected"], [True, False])
        self.assertEqual(m["latencies"], [2])
        self.assertEqual((m["pa_tp"], m["pa_fp"], m["pa_fn"]), (6, 3, 5))
        self.assertEqual(m["normal_samples"], 30 - 11)
        self.assertEqual(m["normal_alarm_samples"], 3)
        straddle = np.zeros(30, dtype=bool)
        straddle[8:11] = True  # run starts before the window and reaches into it: TP, latency 0, not FP
        m2 = channel_metrics(straddle, events)
        self.assertEqual((m2["tp_events"], m2["fp_runs"], m2["latencies"]), (1, 0, [0]))

    def test_prf_and_vectorised_f1(self):
        self.assertEqual(prf(0, 0, 0), (0.0, 0.0, 0.0))
        p, r, f1 = prf(3, 1, 3)
        self.assertAlmostEqual(f1, 2 * p * r / (p + r))
        vec = smap_msl.event_f1_from_counts(np.array([3, 0]), np.array([6, 2]), np.array([1, 5]))
        self.assertAlmostEqual(vec[0], f1)
        self.assertEqual(vec[1], 0.0)

    def test_wilcoxon_hand_cases(self):
        self.assertEqual(wilcoxon_signed_rank([0.0, 0.0])["p_value"], 1.0)
        exact = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(exact["method"], "exact")
        self.assertAlmostEqual(exact["p_value"], 2 / 32)
        self.assertAlmostEqual(wilcoxon_signed_rank([-1.0, 2.0, -3.0, 4.0])["p_value"], 14 / 16)  # W+ = 6, n = 4
        tied = wilcoxon_signed_rank([0.5, 0.5, 0.5, -0.25, 1.0, 0.0])
        self.assertEqual(tied["method"], "normal approximation")
        self.assertEqual(tied["n_nonzero"], 5)

    def test_wilcoxon_matches_scipy(self):
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            self.skipTest("scipy not installed")
        rng = np.random.default_rng(11)
        exact = rng.normal(0.3, 1.0, 20)
        ours = wilcoxon_signed_rank(exact)
        self.assertAlmostEqual(ours["p_value"], wilcoxon(exact, method="exact").pvalue, places=10)
        tied = np.round(rng.normal(0.1, 0.5, 60), 1)
        ours = wilcoxon_signed_rank(tied)
        theirs = wilcoxon(tied, zero_method="wilcox", correction=True, method="approx").pvalue
        self.assertAlmostEqual(ours["p_value"], theirs, places=10)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])


class FetchTests(unittest.TestCase):
    def _zip(self, source: Path, archive: Path, prefix: str = "data/data/") -> None:
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    bundle.write(path, prefix + str(path.relative_to(source)))
            bundle.writestr(prefix + "2018-05-19_15.00.10/models/A-1.h5", b"ignored")
            bundle.writestr("../../evil/train/A-1.npy", b"zip slip")  # must be flattened, never escape

    def test_fetch_download_extract_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_fixture(tmp / "src")
            manifest = tmp / "manifest.sha256"
            manifest.write_text(smap_msl.manifest_text(tmp / "src"), encoding="utf-8")
            archive = tmp / "bundle.zip"
            self._zip(tmp / "src", archive)
            urls = ("file:///nonexistent/oes/missing.zip", archive.resolve().as_uri())
            with self.assertRaises(DataError):  # the zip-slip member overwrote train/A-1.npy -> hash mismatch
                smap_msl.fetch(tmp / "cache", manifest, urls)
            self.assertFalse((tmp / "evil").exists())
            with zipfile.ZipFile(archive, "w") as bundle:
                for path in sorted((tmp / "src").rglob("*")):
                    if path.is_file():
                        bundle.write(path, "data/data/" + str(path.relative_to(tmp / "src")))
            info = smap_msl.fetch(tmp / "cache2", manifest, urls, keep_archive=False)
            self.assertEqual(info["files"], 11)
            self.assertEqual(len(info["errors"]), 1)
            self.assertEqual(info["source"], urls[1])
            self.assertFalse((tmp / "cache2" / "smap_msl_download.zip").exists())
            local = smap_msl.fetch(tmp / "cache3", manifest, archive=archive)
            self.assertEqual(local["archive_sha256"], core.sha256_file(archive))
            code, out, _ = run_cli("smap-msl", "verify", "--data-dir", str(tmp / "cache3"), "--manifest", str(manifest))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["files"], 11)
            code, out, _ = run_cli("smap-msl", "fetch", "--data-dir", str(tmp / "cache4"), "--manifest",
                                   str(manifest), "--url", archive.resolve().as_uri())
            self.assertEqual(code, 0)
            np.save(tmp / "cache3" / "test" / "A-1.npy", np.zeros((3, 25)))
            code, _, err = run_cli("smap-msl", "verify", "--data-dir", str(tmp / "cache3"), "--manifest", str(manifest))
            self.assertEqual(code, 2)
            self.assertIn("different SHA-256", err)

    def test_fetch_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = tmp / "m.sha256"
            manifest.write_text("0" * 64 + "  labeled_anomalies.csv\n", encoding="utf-8")
            with self.assertRaises(DataError):
                smap_msl.fetch(tmp / "c", manifest, ("file:///nonexistent/oes/a.zip",))
            empty = tmp / "empty.zip"
            with zipfile.ZipFile(empty, "w") as bundle:
                bundle.writestr("readme.txt", "nothing")
            with self.assertRaises(DataError):
                smap_msl.fetch(tmp / "c", manifest, archive=empty)

    def test_manifest_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m"
            for text in ("", "nothex  a.npy\n", "0" * 64 + "  ../escape\n", "0" * 64 + "\n"):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(DataError):
                    smap_msl.read_manifest(path)
            path.write_text("# comment\n\n" + "a" * 64 + "  *train/A-1.npy\n", encoding="utf-8")
            self.assertEqual(smap_msl.read_manifest(path), {"train/A-1.npy": "a" * 64})


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_fixture(cls.root / "data")
        cls.manifest = cls.root / "manifest.sha256"
        cls.manifest.write_text(smap_msl.manifest_text(cls.root / "data"), encoding="utf-8")
        cls.protocol = small_protocol(cls.root)
        cls.out = cls.root / "out"
        cls.code, cls.stdout, cls.stderr = run_cli(
            "smap-msl", "evaluate", "--data-dir", str(cls.root / "data"), "--protocol", str(cls.protocol),
            "--manifest", str(cls.manifest), "--out-dir", str(cls.out), "--verify")
        cls.results = json.loads((cls.out / "smap_msl_results.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cli_outputs_and_sha256sums(self):
        self.assertEqual(self.code, 0, self.stderr)
        self.assertIn("headline criterion met", self.stdout)
        sums = (self.out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sums), 5)
        for line in sums:
            digest, name = line.split()
            self.assertEqual(core.sha256_file(self.out / name), digest)
        self.assertTrue((self.out / "smap_msl_event_f1.svg").read_text(encoding="utf-8").startswith("<svg"))

    def test_results_structure(self):
        r = self.results
        self.assertEqual(r["channels"], {"SMAP": ["A-1", "A-2", "D-2"], "MSL": ["M-1", "M-2"]})
        self.assertEqual(r["events"], {"SMAP": 4, "MSL": 2})
        self.assertEqual(r["merged_label_rows"], ["SMAP:A-2"])
        self.assertEqual(r["constant_train_channels"], ["SMAP:D-2"])
        self.assertIn("REAL", r["data_kind"])
        self.assertEqual(r["data_manifest"]["files"], 11)
        self.assertEqual(r["protocol_sha256"], core.sha256_file(self.protocol))
        primary = [c for c in r["comparisons"] if c["subset"] == "all_channels" and c["target_fp"] == 0.01]
        self.assertEqual(len(primary), 8)
        self.assertTrue(all(c["verdict"] in {"oes32 better", "oes32 worse", "no significant difference"}
                            for c in primary))
        self.assertEqual(len(r["per_channel"]), 5 * 3 * 5)
        self.assertIn("criterion_met", r["headline"])
        for s in r["summary"]:
            self.assertLessEqual(s["event_f1_ci_low"], s["event_f1"] + 1e-9)
            self.assertGreaterEqual(s["event_f1_ci_high"], s["event_f1"] - 1e-9)
        # the injected +0.8 level shifts (8 train sigmas) are easy: every method finds every event
        f1 = {(s["dataset"], s["method"]): s for s in r["summary"]
              if s["subset"] == "excluding_constant_train" and s["target_fp"] == 0.0}
        self.assertTrue(all(s["event_recall"] == 1.0 for s in f1.values()))

    def test_thresholds_never_depend_on_test_labels(self):
        """Blindness: moving every test label leaves every threshold unchanged."""
        protocol, _ = smap_msl.load_protocol(self.protocol)
        channels = load_labels(self.root / "data" / "labeled_anomalies.csv")
        moved = [smap_msl.Channel(c.spacecraft, c.chan_id, ((0, 1),), ("point",), c.num_values, 1) for c in channels]
        a = smap_msl.evaluate_channels(self.root / "data", channels, protocol)
        b = smap_msl.evaluate_channels(self.root / "data", moved, protocol)
        self.assertEqual([x["threshold"] for x in a], [y["threshold"] for y in b])
        self.assertNotEqual([x["tp_events"] for x in a], [y["tp_events"] for y in b])

    def test_rerun_is_byte_identical_and_protocol_lock(self):
        out2 = self.root / "out2"
        code, _, _ = run_cli("smap-msl", "evaluate", "--data-dir", str(self.root / "data"), "--protocol",
                             str(self.protocol), "--skip-verify", "--out-dir", str(out2), "--quiet")
        self.assertEqual(code, 0)
        for name in ("smap_msl_summary.csv", "smap_msl_comparisons.csv", "smap_msl_per_channel.csv"):
            self.assertEqual((out2 / name).read_bytes(), (self.out / name).read_bytes())
        code, _, err = run_cli("smap-msl", "evaluate", "--data-dir", str(self.root / "data"), "--protocol",
                               str(self.protocol), "--skip-verify", "--out-dir", str(out2),
                               "--expect-protocol-sha256", "0" * 64)
        self.assertEqual(code, 2)
        self.assertIn("does not match", err)

    def test_train_diagnostics_cli(self):
        code, out, _ = run_cli("smap-msl", "train-diagnostics", "--data-dir", str(self.root / "data"),
                               "--protocol", str(self.protocol))
        self.assertEqual(code, 0)
        rows = list(csv.DictReader(io.StringIO(out)))
        self.assertEqual(len(rows), 5 * 5 * 3)
        self.assertTrue(all(float(r["train_alarm_rate"]) <= float(r["target_fp"]) + 1e-12 for r in rows))

    def test_csv_summary_columns(self):
        rows = list(csv.DictReader((self.out / "smap_msl_summary.csv").open(encoding="utf-8")))
        self.assertEqual(len(rows), 2 * 3 * 2 * 5)
        self.assertEqual(tuple(rows[0]), smap_msl.SUMMARY_FIELDS)

    def test_protocol_validation(self):
        base = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        mutations = [
            lambda p: p.update(schema="other"),
            lambda p: p.update(candidate="maxabs"),
            lambda p: p["methods"].pop("cusum"),
            lambda p: p["calibration"].update(primary_target_fp=1.5),
            lambda p: p["uncertainty"].update(seed="42"),
        ]
        for mutate in mutations:
            protocol = json.loads(json.dumps(base))
            mutate(protocol)
            path = self.root / "bad_protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.subTest(protocol=protocol.get("schema")), self.assertRaises(DataError):
                smap_msl.load_protocol(path)
        code, _, err = run_cli("smap-msl", "evaluate", "--data-dir", str(self.root / "data"), "--protocol",
                               str(self.root / "bad_protocol.json"), "--skip-verify", "--out-dir", str(self.root))
        self.assertEqual(code, 2)

    def test_committed_protocol_is_locked_and_consistent(self):
        protocol, _ = smap_msl.load_protocol(PROTOCOL)
        self.assertEqual(protocol["calibration"]["primary_target_fp"], 0.01)
        self.assertEqual(protocol["uncertainty"]["seed"], 42)
        manifest = ROOT / protocol["data"]["file_manifest"]
        self.assertEqual(core.sha256_file(manifest), protocol["data"]["file_manifest_sha256"])
        self.assertEqual(len(smap_msl.read_manifest(manifest)), 165)


if __name__ == "__main__":
    unittest.main()
