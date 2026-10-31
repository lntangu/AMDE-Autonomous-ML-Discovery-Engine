#!/usr/bin/env python3
"""Run a bounded BFS-versus-ideal-Grover maze query-model demonstration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.quantum.maze import run_maze_grover_demo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--max-statevector-mb", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_maze_grover_demo(device=args.device, max_qubits=args.max_qubits,
                                      max_statevector_mb=args.max_statevector_mb)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if result.get("status") == "ok" else 1
    except Exception as exc:
        print(f"maze demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
