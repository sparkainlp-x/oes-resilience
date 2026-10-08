# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Qiskit / Qiskit Aer executor for :mod:`.qbench` (optional extra ``quantum``).

Circuits are small textbook constructions written here (not imported from MQT Bench, QED-C or Metriq):

* ``ghz``: H on qubit 0, then a CX chain; ideal output is 0...0 or 1...1 with probability 1/2 each.
* ``bv``: Bernstein-Vazirani with a seeded non-zero hidden string on ``width`` data qubits plus one ancilla.
* ``qft``: QFT-style phase encoding of a seeded integer ``x`` (H on every qubit, phase ``2*pi*x*2^k/2^n`` on
  qubit ``k``), followed by an explicit inverse QFT; ideal output is ``x``.

All conditions are transpiled for one hypothetical target (basis gates and a linear coupling map from the plan)
with the instance's ``seed_transpiler``, and simulated with the instance's ``seed_simulator``. The noise model is
synthetic: depolarizing errors after ``sx``/``x`` (p1) and ``cx`` (p2) plus a symmetric readout flip. It is not
calibrated to any device. Classical simulation only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from ._version import __version__

INSTALL_HINT = "qbench simulators need the optional extra: pip install 'oes-resilience[quantum]'"


def _require() -> Any:
    try:
        import qiskit
        import qiskit_aer
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return qiskit, qiskit_aer


def build_circuit(family: str, width: int, params: Mapping[str, int]) -> Any:
    """Logical circuit for one benchmark instance (measures ``width`` classical bits)."""
    from qiskit import QuantumCircuit

    if family == "ghz":
        qc = QuantumCircuit(width, width, name=f"ghz_{width}")
        qc.h(0)
        for q in range(width - 1):
            qc.cx(q, q + 1)
        qc.measure(range(width), range(width))
        return qc
    if family == "bv":
        secret = int(params["secret"])
        qc = QuantumCircuit(width + 1, width, name=f"bv_{width}")
        anc = width
        qc.x(anc)
        qc.h(range(width + 1))
        for q in range(width):
            if (secret >> q) & 1:
                qc.cx(q, anc)
        qc.h(range(width))
        qc.measure(range(width), range(width))
        return qc
    if family == "qft":
        value = int(params["value"])
        n = width
        qc = QuantumCircuit(n, n, name=f"qft_{n}")
        qc.h(range(n))
        for k in range(n):
            qc.p(2 * math.pi * value * (2**k) / (2**n), k)
        # explicit inverse QFT (Qiskit convention: QFT|x> = sum_y exp(2 pi i x y / 2^n) |y>)
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        for j in range(n):
            for m in range(j):
                qc.cp(-math.pi / (2 ** (j - m)), m, j)
            qc.h(j)
        qc.measure(range(n), range(n))
        return qc
    raise ValueError(f"unknown family {family!r}")


def build_noise_model(cfg: Mapping[str, float]) -> Any:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel()
    if cfg["p1"] > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(cfg["p1"], 1), ["sx", "x"])
    if cfg["p2"] > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(cfg["p2"], 2), ["cx"])
    r = cfg["readout"]
    if r > 0:
        model.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return model


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


class AerExecutor:
    """Default :mod:`.qbench` executor: Qiskit Aer, noiseless or with the plan's synthetic noise model."""

    def __init__(self) -> None:
        _require()
        self._backends: dict[str, Any] = {}

    def software_versions(self) -> dict[str, str]:
        import platform

        import numpy

        return {"python": platform.python_version(), "numpy": numpy.__version__, "qiskit": _version("qiskit"),
                "qiskit-aer": _version("qiskit-aer"), "oes-resilience": __version__}

    def __call__(self, family: str, width: int, params: Mapping[str, int], condition: Mapping[str, Any],
                 plan: Mapping[str, Any]) -> dict[str, Any]:
        from qiskit import qasm2, transpile
        from qiskit.transpiler import CouplingMap
        from qiskit_aer import AerSimulator

        circuit = build_circuit(family, width, params)
        target = plan["target_device"]
        n_qubits = circuit.num_qubits
        coupling = CouplingMap.from_line(n_qubits) if target.get("coupling") == "line" else None
        settings = {"optimization_level": condition["optimization_level"], "basis_gates": list(target["basis_gates"]),
                    "coupling_map": "line" if coupling is not None else None, "n_physical_qubits": n_qubits,
                    "seed_transpiler": params["seed_transpiler"]}
        if condition["target"] == "ibm":  # pragma: no cover - opt-in hardware path, never run in CI
            from . import qbench_ibm

            backend = qbench_ibm.get_backend(condition["backend"])
            transpiled = transpile(circuit, backend=backend, optimization_level=condition["optimization_level"],
                                   seed_transpiler=params["seed_transpiler"])
            counts = qbench_ibm.run_counts(transpiled, backend, plan["shots"])
            backend_name = f"ibm:{backend.name}"
            settings = {**settings, "basis_gates": "backend", "coupling_map": "backend"}
        else:
            transpiled = transpile(circuit, basis_gates=settings["basis_gates"], coupling_map=coupling,
                                   optimization_level=condition["optimization_level"],
                                   seed_transpiler=params["seed_transpiler"])
            noise_name = condition.get("noise_model")
            noise = build_noise_model(plan["noise_models"][noise_name]) if noise_name else None
            sim = AerSimulator(noise_model=noise, seed_simulator=params["seed_simulator"], max_parallel_threads=1)
            result = sim.run(transpiled, shots=plan["shots"], seed_simulator=params["seed_simulator"]).result()
            counts = result.get_counts()
            backend_name = f"aer_simulator ({'noise=' + noise_name if noise_name else 'noiseless'})"
        ops = transpiled.count_ops()
        return {
            "counts": dict(counts),
            "backend_name": backend_name,
            "circuit_qasm": qasm2.dumps(circuit),
            "transpiled_qasm": qasm2.dumps(transpiled),
            "transpile": settings,
            "circuit_stats": {"logical_qubits": n_qubits, "logical_depth": circuit.depth(),
                              "transpiled_depth": transpiled.depth(), "transpiled_cx": int(ops.get("cx", 0)),
                              "transpiled_size": transpiled.size()},
        }
