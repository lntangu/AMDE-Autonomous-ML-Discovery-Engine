#!/usr/bin/env python3
"""Wrapper script to execute the AMDE 4.0 real-world benchmark suite."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde.examples.real_world_demo import main

if __name__ == "__main__":
    main()
