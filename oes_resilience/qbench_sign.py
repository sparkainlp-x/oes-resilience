# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Optional detached SSH signatures for qbench evidence passports (``ssh-keygen -Y``).

A passport lists the SHA-256 of every artifact, so a signature over the passport file covers those digests
transitively. A valid signature shows only that the holder of a given private key signed these exact bytes. It
does not show when the run happened, that the listed artifacts are authentic measurements, or that the method is
valid. For public, timestamped signing see Sigstore (``cosign sign-blob``); both are documented in
``docs/qbench.md``. This module wraps OpenSSH's ``ssh-keygen`` (OpenSSH 8.1+) and adds no dependency.

Author: Jean-François Brisson (ORCID 0009-0000-9778-5374), Spark AI NLP.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import EXIT_FAILURE, EXIT_OK, EXIT_USAGE

NAMESPACE = "oes-resilience-qbench-passport"
SIGN_COMMANDS = frozenset({"sign-passport", "verify-signature"})


def _ssh_keygen() -> str:
    exe = shutil.which("ssh-keygen")
    if exe is None:
        raise FileNotFoundError("ssh-keygen (OpenSSH 8.1+) is required for passport signatures but was not found")
    return exe


def signature_path(path: Path) -> Path:
    """Return the detached signature path ``<file>.sig`` that ``ssh-keygen -Y sign`` writes."""
    path = Path(path)
    return path.with_name(path.name + ".sig")


def sign_file(path: Path, key: Path, namespace: str = NAMESPACE) -> Path:
    """Write a detached SSH signature ``<path>.sig`` for ``path``.

    Parameters
    ----------
    path : Path
        File to sign (normally ``<prefix>_passport.json``).
    key : Path
        Private SSH key (for example an ed25519 key). A passphrase prompt, if any, is handled by ssh-keygen.
    namespace : str
        Signature namespace; verification must use the same one.

    Returns
    -------
    Path
        The signature file.

    Raises
    ------
    FileNotFoundError
        If ``path``, ``key`` or ``ssh-keygen`` is missing.
    RuntimeError
        If ssh-keygen fails.
    """
    path, key = Path(path), Path(key)
    for p in (path, key):
        if not p.is_file():
            raise FileNotFoundError(f"not a file: {p}")
    sig = signature_path(path)
    sig.unlink(missing_ok=True)  # ssh-keygen refuses to overwrite an existing signature
    proc = subprocess.run([_ssh_keygen(), "-Y", "sign", "-f", str(key), "-n", namespace, str(path)],
                          capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0 or not sig.is_file():
        raise RuntimeError(f"ssh-keygen sign failed: {proc.stderr.strip()}")
    return sig


def verify_signature(path: Path, allowed_signers: Path, identity: str, signature: Path | None = None,
                     namespace: str = NAMESPACE) -> tuple[bool, str]:
    """Verify a detached SSH signature.

    Parameters
    ----------
    path : Path
        Signed file.
    allowed_signers : Path
        OpenSSH ``allowed_signers`` file mapping identities to public keys.
    identity : str
        Identity that must have signed (for example an e-mail address listed in ``allowed_signers``).
    signature : Path, optional
        Signature file; defaults to ``<path>.sig``.
    namespace : str
        Must match the namespace used to sign.

    Returns
    -------
    (bool, str)
        Whether the signature is valid, and ssh-keygen's message.
    """
    path, allowed_signers = Path(path), Path(allowed_signers)
    signature = signature_path(path) if signature is None else Path(signature)
    for p in (path, allowed_signers, signature):
        if not p.is_file():
            raise FileNotFoundError(f"not a file: {p}")
    with path.open("rb") as handle:
        proc = subprocess.run([_ssh_keygen(), "-Y", "verify", "-f", str(allowed_signers), "-I", identity,
                               "-n", namespace, "-s", str(signature)], stdin=handle, capture_output=True,
                              timeout=60, check=False)
    message = (proc.stdout + proc.stderr).decode("utf-8", "replace").strip()
    return proc.returncode == 0, message


def add_sign_parsers(ops: Any) -> None:
    """Register ``qbench sign-passport`` and ``qbench verify-signature``."""
    sign = ops.add_parser("sign-passport", help="optional: detached SSH signature of a passport (ssh-keygen -Y)")
    sign.add_argument("passport", type=Path)
    sign.add_argument("--key", type=Path, required=True, help="private SSH key, e.g. ~/.ssh/id_ed25519")
    verify = ops.add_parser("verify-signature", help="verify a detached SSH signature of a passport")
    verify.add_argument("passport", type=Path)
    verify.add_argument("--allowed-signers", type=Path, required=True, help="OpenSSH allowed_signers file")
    verify.add_argument("--identity", required=True, help="signer identity listed in allowed_signers")
    verify.add_argument("--signature", type=Path, help="signature file (default: <passport>.sig)")


def cmd_sign(args: argparse.Namespace) -> int:
    """Handle the signing subcommands; return a process exit code."""
    try:
        if args.qbench_command == "sign-passport":
            sig = sign_file(args.passport, args.key)
            print(f"signature: {sig}\nA signature shows key possession only, not provenance or validity.")
            return EXIT_OK
        ok, message = verify_signature(args.passport, args.allowed_signers, args.identity, args.signature)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(message, file=sys.stdout if ok else sys.stderr)
    return EXIT_OK if ok else EXIT_FAILURE
