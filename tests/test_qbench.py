# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the qbench evidence layer (concept, unreleased).

The statistics, hashing, preregistration, passport and CLI tests use a deterministic fake executor and need only
NumPy. The simulator smoke tests run only when the optional extra ``quantum`` (qiskit, qiskit-aer) is installed;
otherwise they are skipped. Nothing here touches quantum hardware.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oes_resilience import core, qbench, qbench_ibm, qbench_sign  # noqa: E402

HAVE_QISKIT = importlib.util.find_spec("qiskit") is not None and importlib.util.find_spec("qiskit_aer") is not None
HAVE_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def small_plan(**kw):
    plan = qbench.default_plan(seed=7, shots=200, widths=[3, 4], batches=3, resamples=300)
    plan.update(kw)
    return plan


class FakeExecutor:
    """Deterministic stand-in for the Aer executor: noisy conditions move a fixed share of shots off the ideal."""

    def software_versions(self):
        return {"fake": "1"}

    def __call__(self, family, width, params, condition, plan):
        ideal = qbench.ideal_distribution(family, width, params)
        shots = plan["shots"]
        keys = sorted(ideal)
        wrong = format((~int(keys[0], 2)) & ((1 << width) - 1), f"0{width}b")
        if wrong in ideal:  # GHZ: complement of 0..0 is 1..1, use a different wrong outcome
            wrong = "0" * (width - 1) + "1"
        bad = {"ideal": 0, "noisy_o1": 20 + params["seed_simulator"] % 7, "noisy_o3": 10 + params["seed_simulator"] % 5}
        n_bad = bad.get(condition["name"], 0)
        good = shots - n_bad
        counts = {}
        for i, k in enumerate(keys):
            counts[k] = good // len(keys) + (good % len(keys) if i == 0 else 0)
        if n_bad:
            counts[wrong] = counts.get(wrong, 0) + n_bad
        return {"counts": counts, "backend_name": f"fake:{condition['name']}",
                "circuit_qasm": f"{family}-{width}-{sorted(params.items())}",
                "transpiled_qasm": f"{family}-{width}-{condition['optimization_level']}",
                "transpile": {"optimization_level": condition["optimization_level"]},
                "circuit_stats": {"logical_qubits": width}}


class FakeHardwareExecutor(FakeExecutor):
    """FakeExecutor that labels the ``hw`` condition as IBM hardware and returns a calibration snapshot."""

    def __call__(self, family, width, params, condition, plan):
        out = FakeExecutor.__call__(self, family, width, params, condition, plan)
        if condition["target"] == "ibm":
            out["backend_name"] = "ibm:fake_device"
            out["calibration"] = {"schema": qbench_ibm.CALIBRATION_SCHEMA, "backend_name": "fake_device",
                                  "batch": params["seed_simulator"] % 2}
        return out


class FakeQubit:
    def __init__(self, t1, t2, frequency):
        self.t1, self.t2, self.frequency = t1, t2, frequency


class FakeInstructionProps:
    def __init__(self, error, duration):
        self.error, self.duration = error, duration


class FakeTarget:
    def __init__(self):
        self.qubit_properties = [FakeQubit(1e-4, 8e-5, 5e9), FakeQubit(None, float("nan"), "bad")]
        self._ops = {"cx": {(1, 0): FakeInstructionProps(0.01, 3e-7), (0, 1): FakeInstructionProps(0.02, None)},
                     "measure": {(0,): FakeInstructionProps(0.03, 1e-6)}, "barrier": {None: None}}
        self.operation_names = list(self._ops)

    def __getitem__(self, name):
        return self._ops[name]


class FakeBackend:
    name = "fake_device"
    backend_version = "1.2.3"
    num_qubits = 2

    def __init__(self, properties=None):
        self.target = FakeTarget()
        self._properties = properties

    def properties(self):
        if self._properties is None:
            raise RuntimeError("no properties")
        return self._properties


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class MetricTests(unittest.TestCase):
    def test_hand_worked_metrics(self):
        p = {"00": 0.5, "11": 0.5}
        q = {"00": 1.0}
        self.assertAlmostEqual(qbench.hellinger_fidelity(p, q), 0.5)
        self.assertAlmostEqual(qbench.total_variation_distance(p, q), 0.5)
        self.assertAlmostEqual(qbench.success_probability(p, q), 0.5)
        self.assertAlmostEqual(qbench.hellinger_fidelity(p, p), 1.0)
        self.assertAlmostEqual(qbench.total_variation_distance(p, p), 0.0)
        self.assertAlmostEqual(qbench.hellinger_fidelity({"01": 1.0}, q), 0.0)
        self.assertAlmostEqual(qbench.total_variation_distance({"01": 1.0}, q), 1.0)

    def test_compute_metrics_from_counts(self):
        m = qbench.compute_metrics({"000": 90, "111": 90, "010": 20}, {"000": 0.5, "111": 0.5})
        self.assertAlmostEqual(m["success_probability"], 0.9)
        self.assertAlmostEqual(m["total_variation_distance"], 0.1)
        self.assertAlmostEqual(m["hellinger_fidelity"], (2 * (0.45 * 0.5) ** 0.5) ** 2)

    def test_normalize_counts_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            qbench.normalize_counts({"0": 0})
        with self.assertRaises(ValueError):
            qbench.normalize_counts({"0": 5, "1": -1})


class StatsTests(unittest.TestCase):
    def test_constant_differences_give_degenerate_interval(self):
        r = qbench.block_bootstrap_ci([0.1] * 6, ["a", "a", "b", "b", "c", "c"], 500, 1)
        self.assertAlmostEqual(r["mean"], 0.1)
        self.assertAlmostEqual(r["ci_low"], 0.1)
        self.assertAlmostEqual(r["ci_high"], 0.1)
        self.assertEqual((r["n_units"], r["n_blocks"], r["resamples"]), (6, 3, 500))
        self.assertTrue(r["few_blocks"])

    def test_deterministic_and_seed_dependent(self):
        diffs = [0.0, 0.1, 0.3, -0.2, 0.5, 0.05, 0.2, -0.1, 0.4, 0.0]
        blocks = [("f", i % 5) for i in range(10)]
        a = qbench.block_bootstrap_ci(diffs, blocks, 1000, 3)
        b = qbench.block_bootstrap_ci(diffs, blocks, 1000, 3)
        c = qbench.block_bootstrap_ci(diffs, blocks, 1000, 4)
        self.assertEqual(a, b)
        self.assertNotEqual((a["ci_low"], a["ci_high"]), (c["ci_low"], c["ci_high"]))
        self.assertFalse(a["few_blocks"])
        self.assertLessEqual(a["ci_low"], a["mean"])
        self.assertGreaterEqual(a["ci_high"], a["mean"])

    def test_interval_stays_within_block_means(self):
        # block means 0 and 1 with unequal sizes: every resample statistic lies in [0, 1]
        r = qbench.block_bootstrap_ci([0.0, 0.0, 0.0, 1.0], ["x", "x", "x", "y"], 400, 0)
        self.assertAlmostEqual(r["mean"], 0.25)
        self.assertGreaterEqual(r["ci_low"], 0.0)
        self.assertLessEqual(r["ci_high"], 1.0)

    def test_bootstrap_input_errors(self):
        for diffs, blocks in (([], []), ([1.0], ["a"]), ([1.0, 2.0], ["a", "a"]), ([1.0, 2.0], ["a"]),
                              ([float("nan"), 1.0], ["a", "b"])):
            with self.assertRaises(ValueError):
                qbench.block_bootstrap_ci(diffs, blocks, 200, 0)

    def test_bootstrap_parameter_errors(self):
        for resamples, alpha in ((0, 0.05), (1.5, 0.05), (True, 0.05), (100, 0.0), (100, 1.0)):
            with self.subTest(resamples=resamples, alpha=alpha), self.assertRaises(ValueError):
                qbench.block_bootstrap_ci([0.1, 0.2], ["a", "b"], resamples, 0, alpha)

    def test_verdict(self):
        self.assertEqual(qbench.verdict(0.01, 0.2, "A", "B"), "B better")
        self.assertEqual(qbench.verdict(-0.2, -0.01, "A", "B"), "A better")
        self.assertEqual(qbench.verdict(-0.1, 0.1, "A", "B"), "no significant difference")

    def _records(self):
        recs = []
        for fam in ("ghz", "bv"):
            for batch in range(3):
                for cond, hf, tvd in (("A", 0.80, 0.15), ("B", 0.90, 0.10)):
                    recs.append({"family": fam, "width": 3, "batch": batch, "condition": cond,
                                 "metrics": {"hellinger_fidelity": hf + 0.01 * batch, "total_variation_distance": tvd,
                                             "success_probability": hf}})
        return recs

    def test_paired_differences_orientation(self):
        recs = self._records()
        d, blocks, units = qbench.paired_differences(recs, "A", "B", "hellinger_fidelity", ["family", "batch"])
        self.assertEqual(len(d), 6)
        for x in d:
            self.assertAlmostEqual(x, 0.10)
        d2, _, _ = qbench.paired_differences(recs, "A", "B", "total_variation_distance", ["family"])
        for x in d2:  # lower TVD is better, so B's lower TVD gives a positive oriented difference
            self.assertAlmostEqual(x, 0.05)
        self.assertEqual(blocks[0], "family=bv|batch=0")
        self.assertEqual(units[0]["family"], "bv")

    def test_paired_differences_errors(self):
        recs = self._records()
        with self.assertRaises(ValueError):
            qbench.paired_differences(recs + [recs[0]], "A", "B", "hellinger_fidelity", ["family"])
        with self.assertRaises(ValueError):
            qbench.paired_differences(recs[:-1], "A", "B", "hellinger_fidelity", ["family"])
        with self.assertRaises(ValueError):
            qbench.paired_differences(recs, "A", "C", "hellinger_fidelity", ["family"])
        with self.assertRaisesRegex(ValueError, "metric"):
            qbench.paired_differences(recs, "A", "B", "accuracy", ["family"])
        with self.assertRaisesRegex(ValueError, "block"):
            qbench.paired_differences(recs, "A", "B", "hellinger_fidelity", ["day"])

    def test_compare_and_metric_swap(self):
        recs = self._records()
        r = qbench.compare(recs, "A", "B", "hellinger_fidelity", ["family", "batch"], 500, 0, 0.05)
        self.assertEqual(r["verdict"], "B better")
        self.assertEqual(set(r["per_family_mean"]), {"bv", "ghz"})
        swap = qbench.metric_swap({"m1": {"verdict": "B better"}, "m2": {"verdict": "no significant difference"}},
                                  "m1")
        self.assertTrue(swap["conclusion_flips"])
        self.assertEqual(swap["primary_verdict"], "B better")
        same = qbench.metric_swap({"m1": {"verdict": "B better"}, "m2": {"verdict": "B better"}}, "m1")
        self.assertFalse(same["conclusion_flips"])


class InstanceTests(unittest.TestCase):
    def test_instance_params_are_seeded_and_order_free(self):
        a = qbench.instance_params(7, "bv", 4, 2)
        self.assertEqual(a, qbench.instance_params(7, "bv", 4, 2))
        self.assertNotEqual(a, qbench.instance_params(8, "bv", 4, 2))
        self.assertTrue(1 <= a["secret"] < 16)
        self.assertIn("value", qbench.instance_params(7, "qft", 3, 0))
        self.assertNotIn("secret", qbench.instance_params(7, "ghz", 3, 0))
        with self.assertRaisesRegex(ValueError, "family"):
            qbench.instance_params(7, "shor", 3, 0)

    def test_ideal_distributions(self):
        self.assertEqual(qbench.ideal_distribution("ghz", 3, {}), {"000": 0.5, "111": 0.5})
        self.assertEqual(qbench.ideal_distribution("bv", 4, {"secret": 5}), {"0101": 1.0})
        self.assertEqual(qbench.ideal_distribution("qft", 3, {"value": 6}), {"110": 1.0})
        with self.assertRaises(ValueError):
            qbench.ideal_distribution("shor", 3, {})

    def test_iter_instances_count(self):
        self.assertEqual(len(qbench.iter_instances(small_plan())), 3 * 2 * 3)


class PlanTests(TempDirCase):
    def test_default_plan_matches_preregistered_example(self):
        committed = ROOT / "examples" / "qbench_plan.json"
        digest = qbench.sidecar_path(committed).read_text().split()[0]
        self.assertEqual(qbench.sha256_bytes(committed.read_bytes()), digest)
        self.assertEqual(qbench.sha256_bytes(qbench.canonical_json(qbench.default_plan())), digest)

    def test_canonical_json_is_order_free(self):
        self.assertEqual(qbench.canonical_json({"b": 1, "a": [1, 2]}), qbench.canonical_json({"a": [1, 2], "b": 1}))
        self.assertTrue(qbench.canonical_json({}).endswith(b"\n"))

    def test_write_prereg_and_load(self):
        path = self.tmp / "plan.json"
        digest = qbench.write_prereg(small_plan(), path)
        self.assertEqual(digest, qbench.sha256_bytes(path.read_bytes()))
        self.assertEqual(qbench.sidecar_path(path).read_text(), f"{digest}  plan.json\n")
        plan, d2 = qbench.load_plan(path, expect_sha256=digest.upper())
        self.assertEqual(d2, digest)
        self.assertEqual(plan, json.loads(path.read_text()))
        with self.assertRaises(FileExistsError):
            qbench.write_prereg(small_plan(), path)
        self.assertEqual(qbench.write_prereg(small_plan(), path, force=True), digest)

    def test_load_plan_detects_tampering(self):
        path = self.tmp / "plan.json"
        digest = qbench.write_prereg(small_plan(), path)
        with self.assertRaises(ValueError):
            qbench.load_plan(path, expect_sha256="0" * 64)
        path.write_text(path.read_text().replace('"shots": 200', '"shots": 201'))
        with self.assertRaisesRegex(ValueError, "sidecar"):
            qbench.load_plan(path)
        qbench.sidecar_path(path).unlink()
        self.assertNotEqual(qbench.load_plan(path)[1], digest)
        path.write_bytes(b"\xff not json")
        with self.assertRaises(ValueError):
            qbench.load_plan(path)

    def test_validate_plan_rejects_bad_plans(self):
        qbench.validate_plan(qbench.default_plan())
        bad = []

        def mutate(fn):
            p = copy.deepcopy(small_plan())
            fn(p)
            bad.append(p)

        mutate(lambda p: p.update(schema="x"))
        mutate(lambda p: p.update(seed=-1))
        mutate(lambda p: p.update(shots=True))
        mutate(lambda p: p.update(shots=0))
        mutate(lambda p: p["instances"].update(families=["ghz", "ghz"]))
        mutate(lambda p: p["instances"].update(families=["shor"]))
        mutate(lambda p: p["instances"].update(widths=[1]))
        mutate(lambda p: p["instances"].update(batches=0))
        mutate(lambda p: p["noise_models"]["synthetic_v1"].update(p2=0.7))
        mutate(lambda p: p["conditions"].append(dict(p["conditions"][0])))
        mutate(lambda p: p["conditions"][0].update(target="cloud"))
        mutate(lambda p: p["conditions"][1].update(noise_model="nope"))
        mutate(lambda p: p["conditions"][0].update(optimization_level=5))
        mutate(lambda p: p["conditions"].append({"name": "hw", "target": "ibm", "optimization_level": 1}))
        mutate(lambda p: p.update(comparisons=[]))
        mutate(lambda p: p["comparisons"][0].update(b="noisy_o1"))
        mutate(lambda p: p["metrics"].update(primary="accuracy"))
        mutate(lambda p: p["blocks"].update(primary=["day"]))
        mutate(lambda p: p["bootstrap"].update(resamples=10))
        mutate(lambda p: p["bootstrap"].update(seed=-3))
        mutate(lambda p: p["bootstrap"].update(alpha=1))
        for p in bad:
            with self.subTest(p=p), self.assertRaises(ValueError):
                qbench.validate_plan(p)
        with self.assertRaises(ValueError):
            qbench.validate_plan([])


class RunAndPassportTests(TempDirCase):
    def _run(self, verify=False):
        plan_path = self.tmp / "plan.json"
        if not plan_path.exists():
            qbench.write_prereg(small_plan(), plan_path)
        return qbench.run_and_write(plan_path, self.tmp / "out", "fake", executor=FakeExecutor(), verify=verify,
                                    argv=["qbench", "run"])

    def test_run_plan_records_and_determinism(self):
        plan = small_plan()
        outcome = qbench.run_plan(plan, "a" * 64, FakeExecutor())
        recs, times = outcome.records, outcome.timestamps
        self.assertEqual(outcome.calibrations, {})
        self.assertIsNone(recs[0]["calibration_sha256"])
        self.assertEqual(len(recs), 18 * 3)
        self.assertEqual(set(times), {r["run_id"] for r in recs})
        r0 = recs[0]
        for key in ("backend_name", "software", "seed_simulator", "seed_transpiler", "shots", "circuit_sha256",
                    "transpiled_circuit_sha256", "transpile", "metrics", "plan_sha256"):
            self.assertIn(key, r0)
        self.assertEqual(r0["software"], {"fake": "1"})
        recs2 = qbench.run_plan(plan, "a" * 64, FakeExecutor(), software={"fake": "1"}).records
        self.assertEqual(qbench.canonical_json(qbench.build_results(plan, "a" * 64, recs)),
                         qbench.canonical_json(qbench.build_results(plan, "a" * 64, recs2)))

    def test_run_plan_rejects_wrong_shot_total(self):
        def short(*args):
            out = FakeExecutor()(*args)
            out["counts"] = {"000": 1}
            return out

        with self.assertRaisesRegex(ValueError, "shots"):
            qbench.run_plan(small_plan(), "a" * 64, short)

    def test_analysis_on_fake_runs(self):
        out = self._run(verify=True)
        self.assertTrue(out["reproducible"])
        analysis = out["results"]["analysis"]
        primary = analysis["comparisons"][0]
        self.assertEqual(primary["by_metric"]["hellinger_fidelity"]["verdict"], "noisy_o3 better")
        self.assertFalse(primary["metric_swap"]["conclusion_flips"])
        self.assertEqual(len(primary["block_sensitivity"]), 2)
        self.assertAlmostEqual(analysis["condition_summary"]["ideal"]["mean_success_probability"], 1.0)

    def test_outputs_and_passport(self):
        out = self._run()
        d = self.tmp / "out"
        for name in ("plan.json", "results.json", "runs.jsonl", "run_metadata.json", "report.md", "passport.json"):
            self.assertTrue((d / f"fake_{name}").is_file(), name)
        passport = json.loads((d / "fake_passport.json").read_text())
        self.assertEqual(passport, out["passport"])
        self.assertEqual(qbench.validate_passport(passport, d), [])
        self.assertEqual(passport["evidence_class"], "synthetic")
        self.assertEqual(passport["result_label"], "synthetic_example")
        self.assertIn(qbench.HASH_DISCLAIMER, passport["limitations"])
        self.assertEqual(passport["protocol"]["sha256"], qbench.sha256_bytes((d / "fake_plan.json").read_bytes()))
        self.assertIn("no external timestamp", passport["protocol"]["locked_at"])
        report = (d / "fake_report.md").read_text()
        self.assertIn("SIMULATOR ONLY", report)
        self.assertIn("not signatures", report)
        runs = [json.loads(line) for line in (d / "fake_runs.jsonl").read_text().splitlines()]
        self.assertEqual(len(runs), 54)
        self.assertIn("started_utc", runs[0])
        results = json.loads((d / "fake_results.json").read_text())
        self.assertNotIn("started_utc", json.dumps(results))
        # tampering with an artifact is detected
        (d / "fake_report.md").write_text(report + "edit")
        self.assertTrue(any("hash mismatch" in e for e in qbench.validate_passport(passport, d)))

    def test_plan_copy_skipped_when_plan_lives_in_out_dir(self):
        d = self.tmp / "out"
        qbench.write_prereg(small_plan(), d / "fake_plan.json")
        out = qbench.run_and_write(d / "fake_plan.json", d, "fake", executor=FakeExecutor())
        self.assertEqual(qbench.validate_passport(out["passport"], d), [])

    def test_validate_passport_error_paths(self):
        out = self._run()
        d = self.tmp / "out"
        good = out["passport"]
        self.assertTrue(qbench.validate_passport({"schema_version": 1}, d)[0].startswith("missing keys"))
        cases = {
            "unexpected keys": lambda p: p.update(extra=1),
            "schema_version": lambda p: p.update(schema_version=2),
            "invalid run_id": lambda p: p.update(run_id="../x"),
            "invalid evidence_class": lambda p: p.update(evidence_class="hardware"),
            "invalid result_label": lambda p: p.update(result_label="great"),
            "protocol.sha256": lambda p: p["protocol"].update(sha256="xyz"),
            "results must not be empty": lambda p: p.update(results=[]),
            "malformed result entry": lambda p: p["results"][0].update(extra=1),
            "malformed metric": lambda p: p["results"][0]["metrics"][0].update(colour="red"),
            "interval_95": lambda p: p["results"][0]["metrics"][0].update(interval_95=[1.0]),
            "artifacts must not be empty": lambda p: p.update(artifacts=[]),
            "path not allowed": lambda p: p["artifacts"][0].update(path="../escape.json"),
            "artifact missing": lambda p: p["artifacts"][0].update(path="nope.json"),
        }
        for expected, fn in cases.items():
            p = copy.deepcopy(good)
            fn(p)
            with self.subTest(expected=expected):
                self.assertTrue(any(expected in e for e in qbench.validate_passport(p, d)), expected)
        p = copy.deepcopy(good)
        p["artifacts"][0]["path"] = "/etc/passwd"
        self.assertTrue(any("not allowed" in e for e in qbench.validate_passport(p, d)))
        (self.tmp / "outside.json").write_text("{}")
        link = d / "link.json"
        try:
            link.symlink_to(self.tmp / "outside.json")
        except OSError:  # pragma: no cover - platforms without symlinks
            self.skipTest("symlinks unavailable")
        p["artifacts"][0]["path"] = "link.json"
        self.assertTrue(any("outside" in e for e in qbench.validate_passport(p, d)))

    def test_hardware_labelled_runs_write_calibration_artifact(self):
        plan = small_plan()
        plan["conditions"].append({"name": "hw", "target": "ibm", "backend": "fake_device",
                                   "noise_model": None, "optimization_level": 1})
        qbench.write_prereg(plan, self.tmp / "hw.json")
        d = self.tmp / "out"
        out = qbench.run_and_write(self.tmp / "hw.json", d, "hw", executor=FakeHardwareExecutor())
        calibrations = json.loads((d / "hw_calibration.json").read_text())
        self.assertEqual(len(calibrations), 2)  # two distinct snapshots, deduplicated by hash
        for digest, snap in calibrations.items():
            self.assertEqual(digest, qbench.sha256_bytes(qbench.canonical_json(snap)))
        hw_runs = [r for r in out["results"]["records"] if r["condition"] == "hw"]
        self.assertTrue(all(r["calibration_sha256"] in calibrations for r in hw_runs))
        passport = out["passport"]
        self.assertIn("calibration_snapshot", [a["role"] for a in passport["artifacts"]])
        self.assertEqual(qbench.validate_passport(passport, d), [])
        self.assertIn("includes IBM hardware runs", passport["results"][0]["scope"])
        self.assertTrue(passport["interpretation"].startswith("Hardware-including"))
        self.assertIn("CONTAINS HARDWARE RUNS", (d / "hw_report.md").read_text())

    def test_run_and_write_rejects_bad_prefix(self):
        qbench.write_prereg(small_plan(), self.tmp / "plan.json")
        with self.assertRaisesRegex(ValueError, "prefix"):
            qbench.run_and_write(self.tmp / "plan.json", self.tmp / "out", "../x", executor=FakeExecutor())

    def test_build_passport_rejects_bad_run_id(self):
        out = self._run()
        with self.assertRaises(ValueError):
            qbench.build_passport(out["results"], self.tmp / "out", "bad id!", [], "note")

    def test_git_commit_is_none_or_sha(self):
        commit = qbench._git_commit()
        self.assertTrue(commit is None or len(commit) == 40)
        with mock.patch("subprocess.run", side_effect=OSError("no git")):
            self.assertIsNone(qbench._git_commit())


class IBMGateTests(TempDirCase):
    def test_gate_reads_environment_only(self):
        self.assertFalse(qbench_ibm.ibm_enabled({}))
        self.assertFalse(qbench_ibm.ibm_enabled({qbench_ibm.ENV_TOKEN: "  "}))
        self.assertTrue(qbench_ibm.ibm_enabled({qbench_ibm.ENV_TOKEN: "t"}))
        with self.assertRaises(qbench_ibm.HardwareNotEnabled):
            qbench_ibm.require_ibm_enabled({})
        qbench_ibm.require_ibm_enabled({qbench_ibm.ENV_TOKEN: "t"})

    def _ibm_plan(self):
        plan = small_plan()
        plan["conditions"].append({"name": "hw", "target": "ibm", "backend": "ibm_example",
                                   "noise_model": None, "optimization_level": 1})
        return plan

    def test_hardware_plan_refused_without_token(self):
        env = {k: v for k, v in os.environ.items() if k != qbench_ibm.ENV_TOKEN}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(qbench_ibm.HardwareNotEnabled):
                qbench.run_plan(self._ibm_plan(), "a" * 64)
            path = self.tmp / "hw.json"
            qbench.write_prereg(self._ibm_plan(), path)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = core.main(["qbench", "run", "--prereg", str(path), "--out-dir", str(self.tmp / "o")])
            self.assertEqual(code, core.EXIT_USAGE)
            self.assertIn(qbench_ibm.ENV_TOKEN, err.getvalue())


class CalibrationSnapshotTests(unittest.TestCase):
    def test_snapshot_from_duck_typed_backend(self):
        props = mock.Mock(last_update_date="2026-10-07T12:00:00Z")
        snap = qbench_ibm.calibration_snapshot(FakeBackend(props), captured_utc="2026-10-07T12:30:00+00:00")
        self.assertEqual(snap["schema"], qbench_ibm.CALIBRATION_SCHEMA)
        self.assertEqual((snap["backend_name"], snap["backend_version"], snap["num_qubits"]),
                         ("fake_device", "1.2.3", 2))
        self.assertEqual(snap["captured_utc"], "2026-10-07T12:30:00+00:00")
        self.assertEqual(snap["properties_last_update"], "2026-10-07T12:00:00Z")
        self.assertEqual(snap["qubits"][0], {"qubit": 0, "t1_s": 1e-4, "t2_s": 8e-5, "frequency_hz": 5e9})
        self.assertEqual(snap["qubits"][1], {"qubit": 1, "t1_s": None, "t2_s": None, "frequency_hz": None})
        self.assertEqual([e["qargs"] for e in snap["instructions"]["cx"]], [[0, 1], [1, 0]])
        self.assertEqual(snap["instructions"]["cx"][1], {"qargs": [1, 0], "error": 0.01, "duration_s": 3e-7})
        self.assertEqual(snap["instructions"]["barrier"], [{"qargs": None, "error": None, "duration_s": None}])
        self.assertEqual(list(snap["instructions"]), ["barrier", "cx", "measure"])
        json.dumps(snap, allow_nan=False)

    def test_snapshot_tolerates_missing_parts(self):
        snap = qbench_ibm.calibration_snapshot(FakeBackend(properties=None))
        self.assertIsNone(snap["properties_last_update"])
        self.assertTrue(snap["captured_utc"].endswith("+00:00"))
        bare = qbench_ibm.calibration_snapshot(object())
        self.assertEqual((bare["backend_name"], bare["num_qubits"], bare["qubits"], bare["instructions"]),
                         ("unknown", 0, [], {}))


class SignatureTests(TempDirCase):
    def test_missing_ssh_keygen_is_reported(self):
        target = self.tmp / "p.json"
        target.write_text("{}")
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "ssh-keygen"):
                qbench_sign.sign_file(target, target)
            code = core.main(["qbench", "sign-passport", str(target), "--key", str(target)])
        self.assertEqual(code, core.EXIT_USAGE)

    def test_missing_inputs(self):
        with self.assertRaises(FileNotFoundError):
            qbench_sign.sign_file(self.tmp / "nope.json", self.tmp / "key")
        with self.assertRaises(FileNotFoundError):
            qbench_sign.verify_signature(self.tmp / "nope.json", self.tmp / "signers", "me")

    def test_sign_failure_is_reported(self):
        target = self.tmp / "p.json"
        target.write_text("{}")
        failed = subprocess.CompletedProcess([], 1, "", "bad key")
        with mock.patch("shutil.which", return_value="/usr/bin/ssh-keygen"), \
                mock.patch("subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "bad key"):
                qbench_sign.sign_file(target, target)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = core.main(["qbench", "sign-passport", str(target), "--key", str(target)])
        self.assertEqual(code, core.EXIT_FAILURE)

    @unittest.skipUnless(HAVE_SSH_KEYGEN, "ssh-keygen not installed")
    def test_sign_and_verify_round_trip(self):
        key = self.tmp / "key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)], check=True)
        signers = self.tmp / "allowed_signers"
        signers.write_text(f"qbench@example.org {(self.tmp / 'key.pub').read_text().strip()}\n")
        passport = self.tmp / "run_passport.json"
        passport.write_text('{"run_id": "x"}\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(core.main(["qbench", "sign-passport", str(passport), "--key", str(key)]), 0)
            # signing twice replaces the old signature instead of failing
            self.assertEqual(core.main(["qbench", "sign-passport", str(passport), "--key", str(key)]), 0)
        self.assertIn("key possession only", out.getvalue())
        self.assertTrue(qbench_sign.signature_path(passport).is_file())
        with contextlib.redirect_stdout(io.StringIO()):
            code = core.main(["qbench", "verify-signature", str(passport), "--allowed-signers", str(signers),
                              "--identity", "qbench@example.org"])
        self.assertEqual(code, 0)
        ok, _ = qbench_sign.verify_signature(passport, signers, "someone@else.org")
        self.assertFalse(ok)
        passport.write_text('{"run_id": "y"}\n')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = core.main(["qbench", "verify-signature", str(passport), "--allowed-signers", str(signers),
                              "--identity", "qbench@example.org"])
        self.assertEqual(code, core.EXIT_FAILURE)


class CLITests(TempDirCase):
    def _main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = core.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_prereg_cli(self):
        path = self.tmp / "p.json"
        code, out, _ = self._main("qbench", "prereg", "--out", str(path), "--widths", "3,4", "--batches", "2")
        self.assertEqual(code, 0)
        self.assertIn("Zenodo/OSF", out)
        self.assertEqual(json.loads(path.read_text())["instances"]["widths"], [3, 4])
        code, _, err = self._main("qbench", "prereg", "--out", str(path))
        self.assertEqual(code, core.EXIT_USAGE)
        self.assertIn("must not be overwritten", err)
        code, _, _ = self._main("qbench", "prereg", "--out", str(path), "--widths", "a,b")
        self.assertEqual(code, core.EXIT_USAGE)

    def test_validate_passport_cli(self):
        qbench.write_prereg(small_plan(), self.tmp / "plan.json")
        qbench.run_and_write(self.tmp / "plan.json", self.tmp / "out", "fake", executor=FakeExecutor())
        passport = self.tmp / "out" / "fake_passport.json"
        code, out, _ = self._main("qbench", "validate-passport", str(passport))
        self.assertEqual(code, 0)
        self.assertIn("not signatures", out)
        (self.tmp / "out" / "fake_results.json").write_text("{}")
        code, _, err = self._main("qbench", "validate-passport", str(passport))
        self.assertEqual(code, core.EXIT_USAGE)
        self.assertIn("hash mismatch", err)

    @unittest.skipIf(HAVE_QISKIT, "qiskit is installed")
    def test_run_without_optional_extra_fails_cleanly(self):
        qbench.write_prereg(small_plan(), self.tmp / "plan.json")
        code, _, err = self._main("qbench", "run", "--prereg", str(self.tmp / "plan.json"),
                                  "--out-dir", str(self.tmp / "o"))
        self.assertEqual(code, core.EXIT_FAILURE)
        self.assertIn("oes-resilience[quantum]", err)

    def test_verify_mismatch_exit_code(self):
        qbench.write_prereg(small_plan(), self.tmp / "plan.json")
        fake = {"out_dir": str(self.tmp), "results_sha256": "0" * 64, "reproducible": False}
        with mock.patch.object(qbench, "run_and_write", return_value=fake):
            code, _, err = self._main("qbench", "run", "--prereg", str(self.tmp / "plan.json"), "--verify")
        self.assertEqual(code, core.EXIT_NOT_REPRODUCIBLE)
        self.assertIn("different results hash", err)


@unittest.skipUnless(HAVE_QISKIT, "optional extra 'quantum' (qiskit, qiskit-aer) not installed")
class SimulatorSmokeTests(TempDirCase):
    def test_noiseless_circuits_hit_their_ideal_outputs(self):
        from oes_resilience.qbench_sim import QiskitExecutor

        ex = QiskitExecutor()
        plan = small_plan()
        ideal_cond = plan["conditions"][0]
        for family in ("ghz", "bv", "qft"):
            for width in (3, 4):
                params = qbench.instance_params(plan["seed"], family, width, 0)
                out = ex(family, width, params, ideal_cond, plan)
                ideal = qbench.ideal_distribution(family, width, params)
                self.assertEqual(sum(out["counts"].values()), plan["shots"])
                self.assertTrue(set(out["counts"]) <= set(ideal), (family, width, out["counts"]))
                self.assertEqual(out["transpile"]["seed_transpiler"], params["seed_transpiler"])
        self.assertIn("qiskit", ex.software_versions())

    def test_noisy_simulator_is_seeded_and_lossy(self):
        from oes_resilience.qbench_sim import QiskitExecutor, build_noise_model

        ex = QiskitExecutor()
        plan = small_plan(shots=400)
        cond = plan["conditions"][1]
        params = qbench.instance_params(plan["seed"], "bv", 4, 1)
        a, b = ex("bv", 4, params, cond, plan), ex("bv", 4, params, cond, plan)
        self.assertEqual(a["counts"], b["counts"])
        self.assertLess(qbench.compute_metrics(a["counts"], qbench.ideal_distribution("bv", 4, params))
                        ["success_probability"], 1.0)
        self.assertIsNotNone(build_noise_model({"p1": 0.0, "p2": 0.0, "readout": 0.0}))

    def test_ibm_branch_with_mocked_backend(self):
        from qiskit.providers.fake_provider import GenericBackendV2

        from oes_resilience.qbench_sim import QiskitExecutor

        backend = GenericBackendV2(num_qubits=6, seed=11)
        plan = small_plan()
        cond = {"name": "hw", "target": "ibm", "backend": "generic", "noise_model": None, "optimization_level": 1}
        params = qbench.instance_params(plan["seed"], "bv", 3, 0)
        ideal_key = next(iter(qbench.ideal_distribution("bv", 3, params)))
        with mock.patch.object(qbench_ibm, "get_backend", return_value=backend) as get_backend, \
                mock.patch.object(qbench_ibm, "run_counts", return_value={ideal_key: plan["shots"]}) as run_counts:
            ex = QiskitExecutor()
            first = ex("bv", 3, params, cond, plan)
            second = ex("bv", 3, params, cond, plan)
        get_backend.assert_called_once_with("generic")
        self.assertEqual(run_counts.call_count, 2)
        self.assertEqual(first["backend_name"], f"ibm:{backend.name}")
        self.assertEqual(first["transpile"]["basis_gates"], "backend")
        self.assertEqual(first["transpile"]["n_physical_qubits"], 6)
        self.assertGreater(first["circuit_stats"]["transpiled_2q"], 0)
        snap = first["calibration"]
        self.assertIs(second["calibration"], snap)  # unchanged calibration is stored once
        self.assertEqual(snap["num_qubits"], 6)
        self.assertEqual(len(snap["qubits"]), 6)
        self.assertIsNotNone(snap["qubits"][0]["t1_s"])
        self.assertTrue(any(e["error"] is not None for ops in snap["instructions"].values() for e in ops))
        json.dumps(snap, allow_nan=False)

    def test_ibm_condition_needs_token_before_any_backend_call(self):
        plan = small_plan()
        plan["conditions"].append({"name": "hw", "target": "ibm", "backend": "generic", "noise_model": None,
                                   "optimization_level": 1})
        env = {k: v for k, v in os.environ.items() if k != qbench_ibm.ENV_TOKEN}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(qbench_ibm, "get_backend") as gb:
            with self.assertRaises(qbench_ibm.HardwareNotEnabled):
                qbench.default_executor(plan)
        gb.assert_not_called()

    def test_unknown_family(self):
        from oes_resilience.qbench_sim import build_circuit

        with self.assertRaises(ValueError):
            build_circuit("shor", 3, {})

    def test_cli_run_end_to_end_with_verify(self):
        path = self.tmp / "plan.json"
        qbench.write_prereg(qbench.default_plan(shots=100, widths=[3], batches=2, resamples=200), path)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = core.main(["qbench", "run", "--prereg", str(path), "--out-dir", str(self.tmp / "o"), "--verify"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("SIMULATOR ONLY", out.getvalue())
        passport = json.loads((self.tmp / "o" / "qbench_passport.json").read_text())
        self.assertEqual(qbench.validate_passport(passport, self.tmp / "o"), [])


if __name__ == "__main__":
    unittest.main()
