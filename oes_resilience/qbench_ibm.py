# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Optional IBM Quantum hardware adapter for :mod:`oes_resilience.qbench`: OFF by default, never run in CI.

The adapter is used only when both of these hold:

* the environment variable ``OES_QBENCH_IBM_TOKEN`` is set (``OES_QBENCH_IBM_INSTANCE`` is optional), and
* the plan contains a condition with ``"target": "ibm"`` and a ``"backend"`` name.

The default plan has no such condition. Hardware runs need the optional extra ``ibm`` (``qiskit-ibm-runtime``).
For every hardware run, :func:`calibration_snapshot` records the backend's reported qubit properties (T1, T2,
frequency) and per-instruction error and duration at run time. The snapshot goes into the passport as an
artifact.

**Status: untested against real hardware.** Neither the author nor CI has run this on a QPU, and the repository
contains no hardware result. The snapshot code is tested against Qiskit's ``GenericBackendV2`` and plain mock
objects only. The token is read from the environment and never written to any output.

Author: Jean-François Brisson (ORCID 0009-0000-9778-5374), Spark AI NLP.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

ENV_TOKEN = "OES_QBENCH_IBM_TOKEN"
ENV_INSTANCE = "OES_QBENCH_IBM_INSTANCE"
CALIBRATION_SCHEMA = "oes-resilience/qbench-calibration/1"


class HardwareNotEnabled(RuntimeError):
    """Raised when a plan asks for IBM hardware but no token is set."""


def ibm_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True if a non-blank ``OES_QBENCH_IBM_TOKEN`` is set in ``env`` (default: the process env)."""
    env = os.environ if env is None else env
    return bool(env.get(ENV_TOKEN, "").strip())


def require_ibm_enabled(env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`HardwareNotEnabled` unless the IBM token is set."""
    if not ibm_enabled(env):
        raise HardwareNotEnabled(
            f"the plan has an IBM hardware condition but {ENV_TOKEN} is not set; hardware runs are opt-in only "
            "(remove the 'ibm' condition to run on local simulators)"
        )


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def calibration_snapshot(backend: Any, captured_utc: str | None = None) -> dict[str, Any]:
    """Capture the calibration data a backend reports at run time.

    Works with any object shaped like a Qiskit ``BackendV2``: it reads ``name``, ``num_qubits``,
    ``backend_version``, ``target.qubit_properties`` (``t1``, ``t2``, ``frequency``) and
    ``target[op][qargs].error/duration``, plus ``properties().last_update_date`` if the backend provides it.
    Missing or non-finite values become ``None``.

    Parameters
    ----------
    backend : object
        Backend at run time.
    captured_utc : str, optional
        Capture time; defaults to the current UTC time.

    Returns
    -------
    dict
        A JSON-ready snapshot with schema ``oes-resilience/qbench-calibration/1``.
    """
    target = getattr(backend, "target", None)
    snapshot: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "backend_name": str(getattr(backend, "name", "unknown")),
        "backend_version": str(getattr(backend, "backend_version", "unknown")),
        "num_qubits": int(getattr(backend, "num_qubits", 0) or 0),
        "captured_utc": captured_utc or datetime.now(timezone.utc).isoformat(),
        "properties_last_update": None,
        "qubits": [],
        "instructions": {},
    }
    if target is not None:
        for index, qp in enumerate(getattr(target, "qubit_properties", None) or []):
            snapshot["qubits"].append({"qubit": index, "t1_s": _num(getattr(qp, "t1", None)),
                                       "t2_s": _num(getattr(qp, "t2", None)),
                                       "frequency_hz": _num(getattr(qp, "frequency", None))})
        for name in sorted(getattr(target, "operation_names", [])):
            entries = []
            for qargs, props in dict(target[name]).items():
                entries.append({"qargs": None if qargs is None else [int(q) for q in qargs],
                                "error": _num(getattr(props, "error", None)),
                                "duration_s": _num(getattr(props, "duration", None))})
            snapshot["instructions"][name] = sorted(entries, key=lambda e: (e["qargs"] is None, e["qargs"] or []))
    properties = getattr(backend, "properties", None)
    if callable(properties):
        try:
            props = properties()
        except Exception:  # noqa: BLE001 - a backend without properties must not abort the snapshot
            props = None
        last = getattr(props, "last_update_date", None)
        snapshot["properties_last_update"] = None if last is None else str(last)
    return snapshot


def get_backend(name: str, env: Mapping[str, str] | None = None) -> Any:  # pragma: no cover - needs a token
    """Return an IBM backend through ``QiskitRuntimeService`` (needs ``qiskit-ibm-runtime`` and the token)."""
    env = os.environ if env is None else env
    require_ibm_enabled(env)
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError("IBM hardware needs the optional extra: pip install 'oes-resilience[quantum,ibm]'") from exc
    kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform", "token": env[ENV_TOKEN].strip()}
    if env.get(ENV_INSTANCE):
        kwargs["instance"] = env[ENV_INSTANCE]
    return QiskitRuntimeService(**kwargs).backend(name)


def run_counts(transpiled: Any, backend: Any, shots: int) -> dict[str, int]:  # pragma: no cover - needs hardware
    """Submit one transpiled circuit with ``SamplerV2`` and return its counts (blocks until the job finishes)."""
    from qiskit_ibm_runtime import SamplerV2

    result = SamplerV2(mode=backend).run([transpiled], shots=shots).result()
    data = result[0].data
    register = next(iter(data.keys()))
    return {str(k): int(v) for k, v in getattr(data, register).get_counts().items()}
