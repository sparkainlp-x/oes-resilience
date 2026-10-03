#!/usr/bin/env python3
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Download the public NASA SMAP/MSL dataset into a cache OUTSIDE the repository and verify it.

Usage: python scripts/fetch_smap_msl.py [CACHE_DIR]   (default: ~/.cache/oes-resilience/smap_msl)

Every extracted file is checked against reports/smap_msl_data.sha256; any mismatch exits with code 2.
The raw data is never committed. Cite Hundman et al. (2018), doi:10.1145/3219819.3219845.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oes_resilience.core import main  # noqa: E402

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".cache" / "oes-resilience" / "smap_msl")
    sys.exit(main(["smap-msl", "fetch", "--data-dir", target, *sys.argv[2:]]))
