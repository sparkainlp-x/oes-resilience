#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/reports/smap_msl_results"

printf '%s\n' '== SMAP/MSL result checksum verification =='
cd "$RESULTS"
sha256sum -c SHA256SUMS

printf '%s\n' '== Recorded run provenance =='
python - <<'PY'
import json
from pathlib import Path

path = Path('smap_msl_run_metadata.json')
metadata = json.loads(path.read_text(encoding='utf-8'))
for key in ('project', 'version', 'python', 'numpy', 'platform', 'created_utc', 'elapsed_seconds'):
    print(f'{key}: {metadata.get(key)}')
print('argv:', ' '.join(metadata.get('argv', [])))
PY

printf '%s\n' '== Verification complete =='
