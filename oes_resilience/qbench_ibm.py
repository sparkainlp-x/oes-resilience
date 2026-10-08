# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Optional IBM Quantum hardware adapter for :mod:`.qbench` -- OFF by default, never run in CI.

This adapter runs only when the user sets the environment variable ``OES_QBENCH_IBM_TOKEN`` (and optionally
``OES_QBENCH_IBM_INSTANCE``) **and** the plan contains a condition with ``"target": "ibm"`` and a ``"backend"``
name. The default plan has no such condition. It needs the optional extra ``ibm`` (``qiskit-ibm-runtime``).

Status: **UNTESTED against real hardware.** The author and CI have never executed it on a QPU, and no hardware
result exists in this repository. The token is read from the environment only and is never written to any
output. Calibration snapshots are not yet recorded; a hardware run should add them before it is used as evidence.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ENV_TOKEN = "OES_QBENCH_IBM_TOKEN"
ENV_INSTANCE = "OES_QBENCH_IBM_INSTANCE"


class HardwareNotEnabled(RuntimeError):
    """Raised when a plan asks for IBM hardware but the user has not provided a token."""


def ibm_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get(ENV_TOKEN, "").strip())


def require_ibm_enabled(env: Mapping[str, str] | None = None) -> None:
    if not ibm_enabled(env):
        raise HardwareNotEnabled(
            f"the plan has an IBM hardware condition but {ENV_TOKEN} is not set; hardware runs are opt-in only "
            "(remove the 'ibm' condition to run on local simulators)"
        )


def get_backend(name: str, env: Mapping[str, str] | None = None) -> Any:  # pragma: no cover - needs a token
    """Return an IBM backend via ``QiskitRuntimeService`` (requires ``qiskit-ibm-runtime``)."""
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
    """Submit one transpiled circuit with ``SamplerV2`` and return measurement counts (blocks until done)."""
    from qiskit_ibm_runtime import SamplerV2

    result = SamplerV2(mode=backend).run([transpiled], shots=shots).result()
    data = result[0].data
    register = next(iter(data.keys()))
    return dict(getattr(data, register).get_counts())
