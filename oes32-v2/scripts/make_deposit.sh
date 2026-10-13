#!/usr/bin/env bash
# Build a Zenodo-ready archive of oes32-v2/ from the committed tree (no upload).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out="${1:-oes32-v2-deposit.zip}"
git archive --format=zip --prefix=oes32-v2/ -o "$out" HEAD:oes32-v2
sha256sum "$out"
