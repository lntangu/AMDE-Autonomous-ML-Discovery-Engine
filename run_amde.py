"""Convenience launcher for source checkouts.

Usage:
    python run_amde.py doctor
    python run_amde.py smoke
    python run_amde.py once --generations 3
    python run_amde.py continuous --max-hours 8
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
