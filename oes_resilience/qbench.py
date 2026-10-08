# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Quantum benchmark evidence layer (concept, unreleased): preregistration, paired statistics, evidence passport.

**Classical software only.** This module orchestrates *simulators* of a few small, textbook benchmark circuits
(GHZ, Bernstein-Vazirani, a QFT-style phase-encoding circuit) and adds an evidence and statistics layer on top:

* :func:`write_prereg` writes the analysis plan and its SHA-256 **before** any run. The lock is self-attested:
  a credible lock needs an external timestamp (for example a Zenodo or OSF deposit of the plan file).
* :func:`run_plan` executes every (circuit instance x condition) pair through an *executor*. The default executor
  (:mod:`.qbench_sim`, optional extra ``quantum``) uses Qiskit Aer: a noiseless simulator and a simulator with a
  simple, documented, **synthetic** noise model. Nothing here is a hardware result.
* :func:`analyze` computes paired differences between conditions with block-bootstrap confidence intervals
  (blocks = circuit family x seed batch by default) and a metric-swap check (Hellinger fidelity vs total
  variation distance vs success probability) that reports whether the conclusion flips.
* :func:`build_passport` emits a run manifest compatible with the evidence-passport v1 schema
  (https://github.com/sparkainlp-x/evidence-passport), listing SHA-256 digests of the plan, results, per-run
  records and report. **A SHA-256 digest is not a signature**: it shows byte consistency only, not who produced
  a file, when, or whether a method is valid.

It is meant to complement open benchmark efforts such as Metriq (Unitary Foundation) and the QED-C
application-oriented benchmarks, not to compete with them. It makes no quantum error correction, threshold,
advantage or hardware claim. This module needs only NumPy; Qiskit is imported lazily by the executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .core import EXIT_FAILURE, EXIT_NOT_REPRODUCIBLE, EXIT_OK, EXIT_USAGE, atomic_write, sha256_file

PLAN_SCHEMA = "oes-resilience/qbench-plan/1"
REPO_URL = "https://github.com/sparkainlp-x/oes-resilience"
FAMILIES = ("ghz", "bv", "qft")
TARGETS = ("aer", "ibm")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
FEW_BLOCKS = 5  # fewer blocks than this: the bootstrap interval is flagged as unreliable

HASH_DISCLAIMER = (
    "SHA-256 digests show only that a listed file is byte-for-byte identical to the recorded digest. "
    "They are not signatures and do not prove who produced a file, when it was produced, its provenance, "
    "or that a method or measurement is valid."
)
LOCK_DISCLAIMER = (
    "The plan hash is self-recorded before the first run; that is not an independent lock. "
    "Deposit the plan file with an external timestamp (for example Zenodo or OSF) before running for a credible lock."
)

# metric name -> True if higher is better
METRICS: dict[str, bool] = {
    "hellinger_fidelity": True,
    "total_variation_distance": False,
    "success_probability": True,
}


# --------------------------------------------------------------------------
# Hashing and canonical JSON
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, 2-space indent, UTF-8, trailing newline)."""
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Analysis plan (preregistration)
# --------------------------------------------------------------------------


def default_plan(seed: int = 20261007, shots: int = 2000, widths: Sequence[int] = (3, 4, 5), batches: int = 4,
                 resamples: int = 2000) -> dict[str, Any]:
    """The default simulator-only analysis plan."""
    return {
        "schema": PLAN_SCHEMA,
        "plan_version": "1",
        "title": "qbench simulator demo: transpiler optimization level under a synthetic noise model",
        "evidence_class": "synthetic",
        "seed": int(seed),
        "shots": int(shots),
        "instances": {"families": list(FAMILIES), "widths": [int(w) for w in widths], "batches": int(batches)},
        "target_device": {
            "name": "synthetic_line_v1",
            "description": "Hypothetical device: linear qubit coupling, basis {rz, sx, x, cx}. Not any real device.",
            "basis_gates": ["rz", "sx", "x", "cx"],
            "coupling": "line",
        },
        "noise_models": {
            "synthetic_v1": {
                "description": "Synthetic, hand-chosen: depolarizing error p1 after each sx/x, p2 after each cx, "
                               "symmetric readout flip probability readout. rz is error-free (virtual). "
                               "Not calibrated to any device.",
                "p1": 0.001,
                "p2": 0.01,
                "readout": 0.02,
            }
        },
        "conditions": [
            {"name": "ideal", "target": "aer", "noise_model": None, "optimization_level": 1},
            {"name": "noisy_o1", "target": "aer", "noise_model": "synthetic_v1", "optimization_level": 1},
            {"name": "noisy_o3", "target": "aer", "noise_model": "synthetic_v1", "optimization_level": 3},
        ],
        "comparisons": [
            {"name": "primary", "a": "noisy_o1", "b": "noisy_o3", "role": "primary"},
            {"name": "sanity", "a": "noisy_o1", "b": "ideal", "role": "sanity"},
        ],
        "metrics": {"primary": "hellinger_fidelity", "swap": ["total_variation_distance", "success_probability"]},
        "blocks": {"primary": ["family", "batch"], "sensitivity": [["family"], ["batch"]]},
        "bootstrap": {"resamples": int(resamples), "seed": int(seed) + 1, "alpha": 0.05},
        "pairing": "Every condition runs the same transpiled-from logical circuit with the same seed_simulator "
                   "and seed_transpiler per instance (common random numbers), so differences are paired by instance.",
        "decision_rule": "For each comparison and metric, the oriented mean paired difference (positive favours "
                         "condition b) is called 'b better' if the 95% block-bootstrap percentile interval lies "
                         "above 0, 'a better' if it lies below 0, and 'no significant difference' otherwise. "
                         "The metric-swap check reports whether the verdict differs across the three metrics.",
        "notes": "Simulator-only demonstration of the evidence layer. Results describe this synthetic noise model "
                 "and these small circuits only; they are not hardware results and support no QEC, threshold or "
                 "advantage claim.",
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if the plan is malformed."""
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be a JSON object")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA!r}")
    for key in ("seed", "shots"):
        if not isinstance(plan.get(key), int) or isinstance(plan.get(key), bool) or plan[key] < 0:
            raise ValueError(f"plan {key!r} must be a non-negative integer")
    if plan["shots"] < 1:
        raise ValueError("plan 'shots' must be >= 1")
    inst = plan.get("instances") or {}
    fams = inst.get("families")
    if not fams or any(f not in FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise ValueError(f"instances.families must be a non-empty subset of {FAMILIES} without duplicates")
    widths = inst.get("widths")
    if not widths or any(not isinstance(w, int) or isinstance(w, bool) or not 2 <= w <= 12 for w in widths):
        raise ValueError("instances.widths must be integers in [2, 12]")
    if not isinstance(inst.get("batches"), int) or inst["batches"] < 1:
        raise ValueError("instances.batches must be an integer >= 1")
    noise = plan.get("noise_models") or {}
    for name, cfg in noise.items():
        for key in ("p1", "p2", "readout"):
            value = cfg.get(key)
            if not isinstance(value, (int, float)) or not 0.0 <= value < 0.5:
                raise ValueError(f"noise model {name!r}: {key} must be a number in [0, 0.5)")
    conditions = plan.get("conditions") or []
    names = [c.get("name") for c in conditions]
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("conditions must have unique, non-empty names")
    for cond in conditions:
        if cond.get("target") not in TARGETS:
            raise ValueError(f"condition {cond['name']!r}: target must be one of {TARGETS}")
        if cond.get("noise_model") is not None and cond["noise_model"] not in noise:
            raise ValueError(f"condition {cond['name']!r}: unknown noise model {cond['noise_model']!r}")
        if cond.get("optimization_level") not in (0, 1, 2, 3):
            raise ValueError(f"condition {cond['name']!r}: optimization_level must be 0..3")
        if cond["target"] == "ibm" and not cond.get("backend"):
            raise ValueError(f"condition {cond['name']!r}: an ibm target needs a 'backend' name")
    comparisons = plan.get("comparisons") or []
    if not comparisons:
        raise ValueError("at least one comparison is required")
    for comp in comparisons:
        if comp.get("a") not in names or comp.get("b") not in names or comp["a"] == comp["b"]:
            raise ValueError(f"comparison {comp.get('name')!r} must reference two different conditions")
    metrics = plan.get("metrics") or {}
    all_metrics = [metrics.get("primary"), *metrics.get("swap", [])]
    if any(m not in METRICS for m in all_metrics):
        raise ValueError(f"metrics must be among {sorted(METRICS)}")
    blocks = plan.get("blocks") or {}
    for scheme in [blocks.get("primary"), *blocks.get("sensitivity", [])]:
        if not scheme or any(k not in ("family", "width", "batch") for k in scheme):
            raise ValueError("block schemes must be non-empty lists of 'family', 'width', 'batch'")
    boot = plan.get("bootstrap") or {}
    if not isinstance(boot.get("resamples"), int) or boot["resamples"] < 100:
        raise ValueError("bootstrap.resamples must be an integer >= 100")
    if not isinstance(boot.get("seed"), int) or boot["seed"] < 0:
        raise ValueError("bootstrap.seed must be a non-negative integer")
    if not isinstance(boot.get("alpha"), float) or not 0.0 < boot["alpha"] < 0.5:
        raise ValueError("bootstrap.alpha must be a float in (0, 0.5)")


def write_prereg(plan: Mapping[str, Any], path: Path, force: bool = False) -> str:
    """Validate and write the plan plus ``<path>.sha256``; return the plan's SHA-256.

    Refuses to overwrite an existing plan unless ``force`` (a plan should not change after it is locked).
    """
    validate_plan(plan)
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; a preregistered plan must not be overwritten (use --force)")
    digest = atomic_write(path, canonical_json(plan))
    atomic_write(sidecar_path(path), f"{digest}  {path.name}\n".encode())
    return digest


def sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def load_plan(path: Path, expect_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    """Load and validate a plan; check it against ``expect_sha256`` and the ``.sha256`` sidecar if present."""
    path = Path(path)
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if expect_sha256 is not None and digest != expect_sha256.lower():
        raise ValueError(f"plan SHA-256 {digest} does not match the expected {expect_sha256}")
    side = sidecar_path(path)
    if side.exists():
        recorded = side.read_text(encoding="utf-8").split()[0].lower()
        if recorded != digest:
            raise ValueError(f"plan SHA-256 {digest} does not match its sidecar {side.name} ({recorded})")
    try:
        plan = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"plan is not valid UTF-8 JSON: {exc}") from exc
    validate_plan(plan)
    return plan, digest


# --------------------------------------------------------------------------
# Instances and ideal distributions (pure Python; the circuits live in qbench_sim)
# --------------------------------------------------------------------------


def instance_params(seed: int, family: str, width: int, batch: int) -> dict[str, int]:
    """Seeded per-instance parameters, independent of run order."""
    ss = np.random.SeedSequence(int(seed), spawn_key=(FAMILIES.index(family), int(width), int(batch)))
    rng = np.random.default_rng(ss)
    seed_transpiler, seed_simulator = (int(x) for x in rng.integers(0, 2**31 - 1, size=2))
    params = {"seed_transpiler": seed_transpiler, "seed_simulator": seed_simulator}
    if family == "bv":
        params["secret"] = int(rng.integers(1, 2**width))  # non-zero hidden string
    elif family == "qft":
        params["value"] = int(rng.integers(0, 2**width))
    return params


def ideal_distribution(family: str, width: int, params: Mapping[str, int]) -> dict[str, float]:
    """Exact output distribution of the noiseless circuit (Qiskit bit order: clbit 0 is the rightmost character)."""
    if family == "ghz":
        return {"0" * width: 0.5, "1" * width: 0.5}
    if family == "bv":
        return {format(int(params["secret"]), f"0{width}b"): 1.0}
    if family == "qft":
        return {format(int(params["value"]), f"0{width}b"): 1.0}
    raise ValueError(f"unknown family {family!r}")


def iter_instances(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    inst = plan["instances"]
    out = []
    for family in inst["families"]:
        for width in inst["widths"]:
            for batch in range(inst["batches"]):
                out.append({"family": family, "width": width, "batch": batch,
                            "params": instance_params(plan["seed"], family, width, batch)})
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def normalize_counts(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    if any(int(v) < 0 for v in counts.values()):
        raise ValueError("counts must be non-negative")
    return {str(k): int(v) / total for k, v in counts.items()}


def hellinger_fidelity(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """(sum_x sqrt(p_x q_x))^2, the squared Bhattacharyya coefficient (same definition as Qiskit's)."""
    bc = sum(math.sqrt(p[k] * q[k]) for k in set(p) & set(q))
    return min(1.0, bc * bc)


def total_variation_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def success_probability(p: Mapping[str, float], ideal: Mapping[str, float]) -> float:
    """Measured probability mass on the support of the ideal distribution."""
    return sum(p.get(k, 0.0) for k, v in ideal.items() if v > 0)


def compute_metrics(counts: Mapping[str, int], ideal: Mapping[str, float]) -> dict[str, float]:
    p = normalize_counts(counts)
    return {
        "hellinger_fidelity": hellinger_fidelity(p, ideal),
        "total_variation_distance": total_variation_distance(p, ideal),
        "success_probability": success_probability(p, ideal),
    }


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------


def block_bootstrap_ci(diffs: Sequence[float], blocks: Sequence[Any], resamples: int, seed: int,
                       alpha: float = 0.05) -> dict[str, Any]:
    """Mean of paired differences with a percentile block-bootstrap interval.

    Whole blocks are resampled with replacement (``len(unique blocks)`` draws per resample); the statistic is the
    mean over all units in the drawn blocks. Deterministic for a given ``seed``.
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or d.size == 0 or len(blocks) != d.size:
        raise ValueError("diffs and blocks must be non-empty and the same length")
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs must be finite")
    keys = sorted({_block_str(b) for b in blocks})
    if len(keys) < 2:
        raise ValueError("block bootstrap needs at least 2 blocks")
    index = {k: i for i, k in enumerate(keys)}
    ids = np.array([index[_block_str(b)] for b in blocks])
    sums = np.bincount(ids, weights=d, minlength=len(keys))
    counts = np.bincount(ids, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(keys), size=(int(resamples), len(keys)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": float(alpha),
        "n_units": int(d.size),
        "n_blocks": len(keys),
        "resamples": int(resamples),
        "few_blocks": len(keys) < FEW_BLOCKS,
        "method": "percentile block bootstrap over whole blocks",
    }


def _block_str(block: Any) -> str:
    return block if isinstance(block, str) else "|".join(str(x) for x in block)


def verdict(ci_low: float, ci_high: float, a: str, b: str) -> str:
    if ci_low > 0:
        return f"{b} better"
    if ci_high < 0:
        return f"{a} better"
    return "no significant difference"


def paired_differences(records: Sequence[Mapping[str, Any]], a: str, b: str, metric: str,
                       block_keys: Sequence[str]) -> tuple[list[float], list[str], list[dict[str, Any]]]:
    """Oriented paired differences (positive favours ``b``) per instance, with block labels."""
    higher_better = METRICS[metric]
    by_cond: dict[str, dict[tuple, Mapping[str, Any]]] = {a: {}, b: {}}
    for rec in records:
        if rec["condition"] in by_cond:
            key = (rec["family"], rec["width"], rec["batch"])
            if key in by_cond[rec["condition"]]:
                raise ValueError(f"duplicate record for {rec['condition']} {key}")
            by_cond[rec["condition"]][key] = rec
    if set(by_cond[a]) != set(by_cond[b]) or not by_cond[a]:
        raise ValueError(f"conditions {a!r} and {b!r} do not cover the same instances")
    diffs, blocks, units = [], [], []
    for key in sorted(by_cond[a]):
        ma, mb = by_cond[a][key]["metrics"][metric], by_cond[b][key]["metrics"][metric]
        diff = (mb - ma) if higher_better else (ma - mb)
        fields = dict(zip(("family", "width", "batch"), key, strict=True))
        diffs.append(diff)
        blocks.append("|".join(f"{k}={fields[k]}" for k in block_keys))
        units.append({**fields, "a": ma, "b": mb, "oriented_diff": diff})
    return diffs, blocks, units


def compare(records: Sequence[Mapping[str, Any]], a: str, b: str, metric: str, block_keys: Sequence[str],
            resamples: int, seed: int, alpha: float) -> dict[str, Any]:
    diffs, blocks, units = paired_differences(records, a, b, metric, block_keys)
    ci = block_bootstrap_ci(diffs, blocks, resamples, seed, alpha)
    per_family: dict[str, float] = {}
    for fam in sorted({u["family"] for u in units}):
        per_family[fam] = float(np.mean([u["oriented_diff"] for u in units if u["family"] == fam]))
    return {
        "metric": metric,
        "higher_is_better": METRICS[metric],
        "orientation": f"positive favours {b}",
        "blocks": list(block_keys),
        **ci,
        "verdict": verdict(ci["ci_low"], ci["ci_high"], a, b),
        "per_family_mean": per_family,
    }


def metric_swap(results_by_metric: Mapping[str, Mapping[str, Any]], primary: str) -> dict[str, Any]:
    verdicts = {m: r["verdict"] for m, r in results_by_metric.items()}
    return {
        "primary_metric": primary,
        "primary_verdict": verdicts[primary],
        "verdicts": verdicts,
        "conclusion_flips": len(set(verdicts.values())) > 1,
    }


def analyze(records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Condition summaries, paired comparisons, block-scheme sensitivity and the metric-swap check."""
    metrics = [plan["metrics"]["primary"], *plan["metrics"]["swap"]]
    boot = plan["bootstrap"]
    summary = {}
    for cond in (c["name"] for c in plan["conditions"]):
        recs = [r for r in records if r["condition"] == cond]
        summary[cond] = {"n_runs": len(recs),
                         **{f"mean_{m}": float(np.mean([r["metrics"][m] for r in recs])) for m in metrics}}
    comparisons = []
    for comp in plan["comparisons"]:
        by_metric = {m: compare(records, comp["a"], comp["b"], m, plan["blocks"]["primary"], boot["resamples"],
                                boot["seed"], boot["alpha"]) for m in metrics}
        sensitivity = [compare(records, comp["a"], comp["b"], plan["metrics"]["primary"], scheme, boot["resamples"],
                               boot["seed"], boot["alpha"]) for scheme in plan["blocks"].get("sensitivity", [])]
        comparisons.append({**comp, "by_metric": by_metric, "block_sensitivity": sensitivity,
                            "metric_swap": metric_swap(by_metric, plan["metrics"]["primary"])})
    return {"condition_summary": summary, "comparisons": comparisons}


# --------------------------------------------------------------------------
# Running a plan
# --------------------------------------------------------------------------

Executor = Callable[..., Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def run_plan(plan: Mapping[str, Any], plan_sha256: str, executor: Executor | None = None,
             software: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Run every instance under every condition; return (records, run timestamps by run_id).

    ``executor(family, width, params, condition, plan)`` must return a mapping with ``counts``, ``backend_name``,
    ``circuit_qasm``, ``transpiled_qasm``, ``transpile`` and ``circuit_stats``. Records are deterministic for a
    deterministic executor; wall-clock timestamps are returned separately.
    """
    validate_plan(plan)
    if executor is None:
        if any(c["target"] == "ibm" for c in plan["conditions"]):
            from .qbench_ibm import require_ibm_enabled

            require_ibm_enabled()
        from .qbench_sim import AerExecutor

        executor = AerExecutor()
    if software is None:
        software = dict(getattr(executor, "software_versions", dict)())
    records: list[dict[str, Any]] = []
    times: dict[str, dict[str, str]] = {}
    for inst in iter_instances(plan):
        ideal = ideal_distribution(inst["family"], inst["width"], inst["params"])
        for cond in plan["conditions"]:
            run_id = f"{inst['family']}-w{inst['width']}-b{inst['batch']}-{cond['name']}"
            started = _utc_now()
            out = executor(inst["family"], inst["width"], inst["params"], cond, plan)
            finished = _utc_now()
            counts = {str(k): int(v) for k, v in sorted(out["counts"].items())}
            if sum(counts.values()) != plan["shots"]:
                raise ValueError(f"{run_id}: executor returned {sum(counts.values())} shots, plan says {plan['shots']}")
            records.append({
                "run_id": run_id,
                "family": inst["family"],
                "width": inst["width"],
                "batch": inst["batch"],
                "condition": cond["name"],
                "backend_name": str(out["backend_name"]),
                "noise_model": cond.get("noise_model"),
                "shots": plan["shots"],
                "seed_simulator": inst["params"]["seed_simulator"],
                "seed_transpiler": inst["params"]["seed_transpiler"],
                "instance_params": dict(inst["params"]),
                "circuit_sha256": sha256_text(out["circuit_qasm"]),
                "transpiled_circuit_sha256": sha256_text(out["transpiled_qasm"]),
                "transpile": dict(out["transpile"]),
                "circuit_stats": dict(out["circuit_stats"]),
                "software": dict(software),
                "plan_sha256": plan_sha256,
                "ideal_distribution": ideal,
                "counts": counts,
                "metrics": compute_metrics(counts, ideal),
            })
            times[run_id] = {"started_utc": started, "finished_utc": finished}
    return records, times


def build_results(plan: Mapping[str, Any], plan_sha256: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic results document (no wall-clock data)."""
    return {
        "project": "OES-Resilience qbench evidence layer",
        "version": __version__,
        "status": "concept, unreleased",
        "evidence": "SIMULATOR ONLY (classical simulation; synthetic noise model). No hardware results.",
        "plan_sha256": plan_sha256,
        "plan": plan,
        "records": list(records),
        "analysis": analyze(records, plan),
    }


# --------------------------------------------------------------------------
# Report and evidence passport
# --------------------------------------------------------------------------


def _f(x: float) -> str:
    return f"{x:+.4f}"


def render_report(results: Mapping[str, Any], lock_note: str) -> str:
    plan, analysis = results["plan"], results["analysis"]
    lines = [
        "# qbench evidence report (SIMULATOR ONLY)",
        "",
        "> **SIMULATOR ONLY.** Classical software running Qiskit Aer simulators with a synthetic noise model. "
        "No quantum hardware was used. Nothing here is a QEC, threshold, advantage or hardware claim.",
        "",
        f"- Plan: `{plan['title']}` (schema `{plan['schema']}`), SHA-256 `{results['plan_sha256']}`",
        f"- Lock: {lock_note}",
        f"- Seed {plan['seed']}, {plan['shots']} shots per run, families {', '.join(plan['instances']['families'])}, "
        f"widths {plan['instances']['widths']}, {plan['instances']['batches']} seed batches, "
        f"{len(results['records'])} runs.",
        "",
        "## Condition means",
        "",
        "| condition | runs | " + " | ".join(m for m in METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for cond, s in analysis["condition_summary"].items():
        vals = " | ".join(f"{s[f'mean_{m}']:.4f}" if f"mean_{m}" in s else "-" for m in METRICS)
        lines.append(f"| `{cond}` | {s['n_runs']} | {vals} |")
    lines += ["", "## Paired comparisons (block bootstrap, 95% percentile interval)", "",
              "Differences are oriented so that a positive value favours condition b "
              "(for total variation distance, lower is better, so the sign is flipped).", "",
              "| comparison | a | b | metric | blocks | mean diff | 95% CI | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for comp in analysis["comparisons"]:
        for m, r in comp["by_metric"].items():
            lines.append(f"| {comp['name']} | `{comp['a']}` | `{comp['b']}` | {m} | {r['n_blocks']} "
                         f"({' x '.join(r['blocks'])}) | {_f(r['mean'])} | [{_f(r['ci_low'])}, {_f(r['ci_high'])}] "
                         f"| {r['verdict']} |")
    lines += ["", "## Metric-swap check", "", "| comparison | primary verdict | flips across metrics? | verdicts |",
              "|---|---|---|---|"]
    for comp in analysis["comparisons"]:
        ms = comp["metric_swap"]
        vs = "; ".join(f"{m}: {v}" for m, v in ms["verdicts"].items())
        lines.append(f"| {comp['name']} | {ms['primary_verdict']} | {'YES' if ms['conclusion_flips'] else 'no'} "
                     f"| {vs} |")
    lines += ["", "## Block-scheme sensitivity (primary metric)", "",
              "| comparison | blocks | n blocks | mean diff | 95% CI | verdict | note |",
              "|---|---|---|---|---|---|---|"]
    for comp in analysis["comparisons"]:
        for r in comp["block_sensitivity"]:
            note = "fewer than 5 blocks: interval unreliable" if r["few_blocks"] else ""
            lines.append(f"| {comp['name']} | {' x '.join(r['blocks'])} | {r['n_blocks']} | {_f(r['mean'])} | "
                         f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {r['verdict']} | {note} |")
    lines += ["", "## What the hashes do and do not show", "", HASH_DISCLAIMER, "", LOCK_DISCLAIMER, ""]
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


def build_passport(results: Mapping[str, Any], out_dir: Path, run_id: str, artifacts: Sequence[tuple[str, str, str]],
                   lock_note: str, commit: str | None = None) -> dict[str, Any]:
    """Run manifest in the evidence-passport v1 schema. ``artifacts``: (file name in out_dir, role, description)."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} is not valid for the passport schema")
    out_dir = Path(out_dir)
    plan, analysis = results["plan"], results["analysis"]
    alpha = plan["bootstrap"]["alpha"]
    out_results = []
    for comp in analysis["comparisons"]:
        metrics = []
        for m, r in comp["by_metric"].items():
            entry = {"name": f"mean paired difference, {m} (positive favours {comp['b']})", "value": r["mean"],
                     "unit": "probability difference",
                     "note": f"{r['verdict']}; {r['n_units']} paired instances in {r['n_blocks']} blocks "
                             f"({' x '.join(r['blocks'])}); {r['resamples']} block-bootstrap resamples"}
            if math.isclose(alpha, 0.05):
                entry["interval_95"] = [r["ci_low"], r["ci_high"]]
            metrics.append(entry)
        flips = comp["metric_swap"]["conclusion_flips"]
        metrics.append({"name": "metric-swap conclusion flips", "value": 1 if flips else 0, "unit": "boolean (1=yes)",
                        "note": "; ".join(f"{m}: {v}" for m, v in comp["metric_swap"]["verdicts"].items())})
        out_results.append({"scope": f"{comp['name']} comparison, SIMULATOR ONLY (Qiskit Aer, synthetic noise)",
                            "method": f"{comp['b']} vs {comp['a']}", "metrics": metrics})
    summary_metrics = []
    for cond, s in analysis["condition_summary"].items():
        summary_metrics.append({"name": f"{cond}: mean {plan['metrics']['primary']}",
                                "value": s[f"mean_{plan['metrics']['primary']}"], "unit": "fraction",
                                "note": f"{s['n_runs']} runs"})
    out_results.append({"scope": "condition means, SIMULATOR ONLY", "method": "all conditions",
                        "metrics": summary_metrics})
    primary = next((c for c in analysis["comparisons"] if c.get("role") == "primary"), analysis["comparisons"][0])
    ms = primary["metric_swap"]
    interpretation = (
        f"Simulator-only demonstration. Primary comparison {primary['b']} vs {primary['a']}: "
        f"{ms['primary_verdict']} on {ms['primary_metric']}; the metric-swap check "
        f"{'FLIPS' if ms['conclusion_flips'] else 'does not flip'} the conclusion across "
        f"{', '.join(ms['verdicts'])}. These numbers describe a synthetic noise model and small textbook circuits "
        "only; they are not hardware results."
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "project": {"name": "OES-Resilience qbench evidence layer", "version": f"{__version__}+qbench (unreleased)",
                    "source_url": REPO_URL,
                    "source_note": "Generated by oes_resilience.qbench (concept, unreleased). The code commit is "
                                   "read from git when available; it is descriptive metadata, not verification."},
        "evidence_class": "synthetic",
        "dataset": {"name": "qbench simulator runs",
                    "kind": "classical simulation of small benchmark circuits (Qiskit Aer)",
                    "description": f"{len(results['records'])} runs: families "
                                   f"{', '.join(plan['instances']['families'])}; widths {plan['instances']['widths']};"
                                   f" {plan['instances']['batches']} seed batches; conditions "
                                   f"{', '.join(c['name'] for c in plan['conditions'])}; {plan['shots']} shots each.",
                    "source_url": None},
        "code": {"version": __version__, "commit": commit},
        "seed": plan["seed"],
        "protocol": {"name": plan["title"], "version": plan["plan_version"], "sha256": results["plan_sha256"],
                     "locked_at": lock_note, "lock_commit": None, "description": plan["decision_rule"]},
        "baselines": sorted({c["a"] for c in plan["comparisons"]}),
        "results": out_results,
        "result_label": "synthetic_example",
        "interpretation": interpretation,
        "limitations": [
            "SIMULATOR ONLY: no quantum hardware was used; the noise model is synthetic and hand-chosen, not "
            "calibrated to any device.",
            "Small textbook circuits (GHZ, Bernstein-Vazirani, QFT-style phase encoding) implemented locally; they "
            "are not imported from MQT Bench, QED-C or Metriq and are not a recognized benchmark suite.",
            "No quantum error correction, threshold, advantage or hardware-performance claim is made or supported.",
            "Simulator counts depend on the Qiskit and Qiskit Aer versions; byte-identical reruns are expected only "
            "with the same versions.",
            HASH_DISCLAIMER,
            LOCK_DISCLAIMER,
        ],
        "artifacts": [],
    }
    for name, role, description in artifacts:
        path = out_dir / name
        manifest["artifacts"].append({"path": name, "role": role, "description": description,
                                      "sha256": sha256_file(path)})
    return manifest


PASSPORT_REQUIRED = ("schema_version", "run_id", "project", "evidence_class", "dataset", "code", "seed", "protocol",
                     "baselines", "results", "result_label", "interpretation", "limitations", "artifacts")
RESULT_LABELS = ("criterion_met", "criterion_not_met", "descriptive_only", "synthetic_example", "not_stated")


def validate_passport(manifest: Mapping[str, Any], base_dir: Path) -> list[str]:
    """Minimal structural check against the evidence-passport v1 schema plus artifact hash re-check.

    Returns a list of problems (empty if none). Not a full JSON-Schema validator.
    """
    errors: list[str] = []
    missing = [k for k in PASSPORT_REQUIRED if k not in manifest]
    extra = sorted(set(manifest) - set(PASSPORT_REQUIRED))
    if missing:
        errors.append(f"missing keys: {missing}")
    if extra:
        errors.append(f"unexpected keys: {extra}")
    if missing:
        return errors
    if manifest["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if not isinstance(manifest["run_id"], str) or not RUN_ID_RE.fullmatch(manifest["run_id"]):
        errors.append("invalid run_id")
    if manifest["evidence_class"] not in ("public_dataset", "synthetic"):
        errors.append("invalid evidence_class")
    if manifest["result_label"] not in RESULT_LABELS:
        errors.append("invalid result_label")
    if not SHA256_RE.fullmatch(str(manifest["protocol"].get("sha256", "")).lower()):
        errors.append("protocol.sha256 is not a SHA-256 hex digest")
    if not manifest["results"]:
        errors.append("results must not be empty")
    for res in manifest["results"]:
        if set(res) != {"scope", "method", "metrics"} or not res["metrics"]:
            errors.append(f"malformed result entry: {res.get('scope')!r}")
            continue
        for met in res["metrics"]:
            if not {"name", "value", "unit"} <= set(met) or set(met) - {"name", "value", "unit", "note",
                                                                       "interval_95"}:
                errors.append(f"malformed metric in {res['scope']!r}")
            elif "interval_95" in met and len(met["interval_95"]) != 2:
                errors.append(f"interval_95 must have 2 numbers in {res['scope']!r}")
    if not manifest["artifacts"]:
        errors.append("artifacts must not be empty")
    base = Path(base_dir).resolve()
    for art in manifest["artifacts"]:
        rel = str(art.get("path", ""))
        if not rel or rel.startswith("/") or "\\" in rel or ".." in Path(rel).parts:
            errors.append(f"artifact path not allowed: {rel!r}")
            continue
        path = (base / rel).resolve()
        if base not in path.parents:
            errors.append(f"artifact outside the manifest directory: {rel!r}")
        elif not path.is_file():
            errors.append(f"artifact missing: {rel!r}")
        elif sha256_file(path) != str(art.get("sha256", "")).lower():
            errors.append(f"artifact hash mismatch: {rel!r}")
    return errors


# --------------------------------------------------------------------------
# End-to-end run with outputs
# --------------------------------------------------------------------------


def run_and_write(plan_path: Path, out_dir: Path, prefix: str, expect_sha256: str | None = None,
                  executor: Executor | None = None, verify: bool = False,
                  argv: Sequence[str] = ()) -> dict[str, Any]:
    """Load the plan (hash recorded first), run it, write results, per-run records, report and passport."""
    plan, plan_sha = load_plan(plan_path, expect_sha256)
    hash_recorded_utc = _utc_now()  # recorded before the first circuit runs
    plan_mtime_utc = datetime.fromtimestamp(Path(plan_path).stat().st_mtime, timezone.utc).isoformat()
    lock_note = (f"self-reported: plan SHA-256 recorded at {hash_recorded_utc} before the first run; plan file "
                 f"mtime {plan_mtime_utc}; no external timestamp")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, times = run_plan(plan, plan_sha, executor)
    results = build_results(plan, plan_sha, records)
    results_bytes = canonical_json(results)
    reproducible = None
    if verify:
        records2, _ = run_plan(plan, plan_sha, executor)
        reproducible = sha256_bytes(canonical_json(build_results(plan, plan_sha, records2))) == sha256_bytes(
            results_bytes)
    plan_copy = f"{prefix}_plan.json"
    if Path(plan_path).resolve() != (out_dir / plan_copy).resolve():
        shutil.copyfile(plan_path, out_dir / plan_copy)
    atomic_write(out_dir / f"{prefix}_results.json", results_bytes)
    runs_lines = [json.dumps({**{k: r[k] for k in ("run_id", "condition", "backend_name", "shots", "seed_simulator",
                                                    "seed_transpiler", "circuit_sha256", "transpiled_circuit_sha256",
                                                    "transpile", "software", "metrics", "plan_sha256")},
                              **times[r["run_id"]]}, sort_keys=True, ensure_ascii=False) for r in records]
    atomic_write(out_dir / f"{prefix}_runs.jsonl", ("\n".join(runs_lines) + "\n").encode("utf-8"))
    meta = {"project": "OES-Resilience qbench", "version": __version__, "created_utc": _utc_now(),
            "plan_hash_recorded_utc": hash_recorded_utc, "plan_mtime_utc": plan_mtime_utc,
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "software": records[0]["software"] if records else {}, "argv": list(argv),
            "verify_reproducible": reproducible}
    atomic_write(out_dir / f"{prefix}_run_metadata.json", canonical_json(meta))
    atomic_write(out_dir / f"{prefix}_report.md", render_report(results, lock_note).encode("utf-8"))
    passport = build_passport(results, out_dir, prefix, [
        (plan_copy, "protocol", "Analysis plan, byte-identical copy of the preregistered file."),
        (f"{prefix}_results.json", "results", "Deterministic results: plan, per-run records, paired analysis."),
        (f"{prefix}_runs.jsonl", "run_records", "Per-run records including wall-clock timestamps."),
        (f"{prefix}_run_metadata.json", "run_metadata", "Platform, versions, argv and lock timestamps."),
        (f"{prefix}_report.md", "report", "Human-readable report generated from the results."),
    ], lock_note, commit=_git_commit())
    atomic_write(out_dir / f"{prefix}_passport.json", canonical_json(passport))
    return {"results": results, "passport": passport, "results_sha256": sha256_bytes(results_bytes),
            "reproducible": reproducible, "out_dir": str(out_dir)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc


def add_qbench_parser(sub: Any) -> argparse.ArgumentParser:
    parser = sub.add_parser("qbench", help="quantum benchmark evidence layer on local simulators (concept, unreleased)")
    ops = parser.add_subparsers(dest="qbench_command", required=True)
    pre = ops.add_parser("prereg", help="write the analysis plan and its SHA-256 BEFORE any run")
    pre.add_argument("--out", type=Path, required=True, help="plan file to write (a .sha256 sidecar is added)")
    pre.add_argument("--seed", type=int, default=20261007)
    pre.add_argument("--shots", type=int, default=2000)
    pre.add_argument("--widths", type=_int_list, default=[3, 4, 5])
    pre.add_argument("--batches", type=int, default=4)
    pre.add_argument("--resamples", type=int, default=2000)
    pre.add_argument("--force", action="store_true", help="overwrite an existing plan (breaks the lock)")
    run = ops.add_parser("run", help="run a preregistered plan on local simulators and write the evidence passport")
    run.add_argument("--prereg", type=Path, required=True, help="plan written by 'qbench prereg'")
    run.add_argument("--expect-plan-sha256", help="refuse to run unless the plan has this SHA-256")
    run.add_argument("--out-dir", type=Path, default=Path("outputs/qbench"))
    run.add_argument("--prefix", default="qbench")
    run.add_argument("--verify", action="store_true", help="run twice; exit 4 unless the results hash matches")
    run.add_argument("--quiet", action="store_true")
    val = ops.add_parser("validate-passport", help="check a passport's structure and re-hash its artifacts")
    val.add_argument("passport", type=Path)
    return parser


def cmd_qbench(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Handler for ``qbench``."""
    from .qbench_ibm import HardwareNotEnabled

    if args.qbench_command == "prereg":
        try:
            plan = default_plan(args.seed, args.shots, args.widths, args.batches, args.resamples)
            digest = write_prereg(plan, args.out, force=args.force)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        print(f"plan: {args.out}\nsha256: {digest}\nsidecar: {sidecar_path(args.out)}")
        print("Next: deposit the plan with an external timestamp (Zenodo/OSF) BEFORE running, for a credible lock.")
        return EXIT_OK
    if args.qbench_command == "validate-passport":
        manifest = json.loads(Path(args.passport).read_text(encoding="utf-8"))
        errors = validate_passport(manifest, Path(args.passport).parent)
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        if not errors:
            print(f"OK: {args.passport} ({len(manifest['artifacts'])} artifacts re-hashed). {HASH_DISCLAIMER}")
        return EXIT_USAGE if errors else EXIT_OK
    try:
        out = run_and_write(args.prereg, args.out_dir, args.prefix, args.expect_plan_sha256, verify=args.verify,
                            argv=argv)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except HardwareNotEnabled as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not args.quiet:
        print(Path(out["out_dir"]) / f"{args.prefix}_report.md")
        print(f"results sha256: {out['results_sha256']} (SIMULATOR ONLY)")
    if args.verify and not out["reproducible"]:
        print("error: rerun produced a different results hash", file=sys.stderr)
        return EXIT_NOT_REPRODUCIBLE
    return EXIT_OK


__all__ = [
    "METRICS", "PLAN_SCHEMA", "analyze", "block_bootstrap_ci", "build_passport", "build_results", "canonical_json",
    "compute_metrics", "default_plan", "hellinger_fidelity", "ideal_distribution", "instance_params", "load_plan",
    "metric_swap", "paired_differences", "render_report", "run_and_write", "run_plan", "success_probability",
    "total_variation_distance", "validate_passport", "validate_plan", "write_prereg",
]
