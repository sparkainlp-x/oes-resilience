#!/usr/bin/env bash
# Reproduce the committed detector scorecard examples/compare_scorecard.csv.
#
#   bash scripts/reproduce_compare.sh            # uses `python` on PATH
#   PYTHON=.venv/bin/python bash scripts/reproduce_compare.sh
#
# Needs the package importable (pip install . or run from the repo root) and
# NumPy. Synthetic data only; no network access.
# Exit codes: 0 match, 1 hash mismatch, other = the compare command's own code.
set -euo pipefail

EXPECTED_SHA256="b907d55a2de73e64e4e677bd7b9645500b502af939bc250f9346ef406fdf844c"
# Reference runtime: about 70 s wall clock on one 8-core Linux box
# (Python 3.13, NumPy 2.5). Machine-dependent; for orientation only.

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/.."
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

start=$(date +%s)
"$PYTHON" -m oes_resilience compare --quiet --verify --out-dir "$OUT" --prefix compare
elapsed=$(( $(date +%s) - start ))

actual=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$OUT/compare_scorecard.csv")
echo "expected sha256: $EXPECTED_SHA256"
echo "computed sha256: $actual"
echo "runtime: ${elapsed} s"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
  echo "MISMATCH: rebuilt compare_scorecard.csv differs from the expected hash" >&2
  exit 1
fi
cmp "$OUT/compare_scorecard.csv" examples/compare_scorecard.csv
echo "OK: compare_scorecard.csv reproduced byte for byte"
