"""Allow `python -m amde` when the package is not installed."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
