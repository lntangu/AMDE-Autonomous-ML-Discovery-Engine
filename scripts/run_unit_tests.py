"""Run AMDE unit tests with a deterministic source-tree import path."""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

# Keep test-only numerical thread pools bounded. The tests validate correctness,
# not peak CPU throughput; unconstrained BLAS/OpenMP pools can oversubscribe small
# CI hosts and make spawned-process tests appear to hang.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    started = time.monotonic()
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(ROOT / "tests" / "unit"),
        pattern="test_*.py",
        top_level_dir=str(ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.monotonic() - started
    print(
        "UNIT_GATE "
        f"tests={result.testsRun} "
        f"failures={len(result.failures)} "
        f"errors={len(result.errors)} "
        f"skipped={len(result.skipped)} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
