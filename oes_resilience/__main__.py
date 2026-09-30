# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""``python -m oes_resilience``."""

import sys

from .core import main

if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
