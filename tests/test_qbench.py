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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oes_resilience import core, qbench, qbench_ibm  # noqa: E402

HAVE_QISKIT = importlib.util.find_spec("qiskit") is not None and importlib.util.find_spec("qiskit_aer") is not None


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

    def test_ideal_distributions(self):
        self.assertEqual(qbench.ideal_distribution("ghz", 3, {}), {"000": 0.5, "111": 0.5})
        self.assertEqual(qbench.ideal_distribution("bv", 4, {"secret": 5}), {"0101": 1.0})
        self.assertEqual(qbench.ideal_distribution("qft", 3, {"value": 6}), {"110": 1.0})
        with self.assertRaises(ValueError):
            qbench.ideal_distribution("shor", 3, {})

    def test_iter_instances_count(self):
        self.assertEqual(len(qbench.iter_instances(small_plan())), 3 * 2 * 3)


class PlanTests(TempDirCase):
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
        recs, times = qbench.run_plan(plan, "a" * 64, FakeExecutor())
        self.assertEqual(len(recs), 18 * 3)
        self.assertEqual(set(times), {r["run_id"] for r in recs})
        r0 = recs[0]
        for key in ("backend_name", "software", "seed_simulator", "seed_transpiler", "shots", "circuit_sha256",
                    "transpiled_circuit_sha256", "transpile", "metrics", "plan_sha256"):
            self.assertIn(key, r0)
        self.assertEqual(r0["software"], {"fake": "1"})
        recs2, _ = qbench.run_plan(plan, "a" * 64, FakeExecutor(), software={"fake": "1"})
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
        from oes_resilience.qbench_sim import AerExecutor

        ex = AerExecutor()
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
        from oes_resilience.qbench_sim import AerExecutor, build_noise_model

        ex = AerExecutor()
        plan = small_plan(shots=400)
        cond = plan["conditions"][1]
        params = qbench.instance_params(plan["seed"], "bv", 4, 1)
        a, b = ex("bv", 4, params, cond, plan), ex("bv", 4, params, cond, plan)
        self.assertEqual(a["counts"], b["counts"])
        self.assertLess(qbench.compute_metrics(a["counts"], qbench.ideal_distribution("bv", 4, params))
                        ["success_probability"], 1.0)
        self.assertIsNotNone(build_noise_model({"p1": 0.0, "p2": 0.0, "readout": 0.0}))

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
