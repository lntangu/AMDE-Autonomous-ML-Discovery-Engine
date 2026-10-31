"""Test package bootstrap.

Adds the source checkout to ``sys.path`` so the stdlib ``unittest`` suite can
run without installing the package. No third-party test runner is required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

for _path in (ROOT, SRC, TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Spawned evaluator workers inherit the parent environment and sys.path. Keep
# both consistent even when the suite is launched from another directory.
_existing = os.environ.get("PYTHONPATH", "")
_parts = [str(SRC)]
if _existing:
    _parts.append(_existing)
os.environ["PYTHONPATH"] = os.pathsep.join(_parts)
