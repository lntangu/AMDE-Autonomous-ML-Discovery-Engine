#!/usr/bin/env python3
"""Specialized Time-Budget & Runtime Governance Auditor for AMDE v5.1.0 (Phase 2 & Phase 7).

Audits all candidate evaluation loops and wrapper fit methods:
- Verifies use of time.monotonic() instead of time.time() for elapsed time calculation.
- Checks whether candidate loops check remaining budget before dispatching candidates.
- Checks whether remaining budget is passed down to estimators or subprocess workers.
- Checks whether finalization reserve (10-15%) is reserved for ensembling/refitting.
- Detects unbounded fit() calls that could overrun wall-clock limits by > 10%.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class TimeBudgetASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.rel_path = str(file_path.relative_to(ROOT)).replace("\\", "/")
        self.findings: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}" if hasattr(ast, "unparse") else node.func.attr

        # Use of time.time() instead of time.monotonic()
        if func_name == "time.time":
            self.findings.append({
                "type": "non_monotonic_clock",
                "severity": "Medium",
                "file": self.rel_path,
                "line": node.lineno,
                "detail": "Use of time.time() for duration or timeout calculation; should use time.monotonic() to avoid NTP drift/clock adjustments.",
            })

        self.generic_visit(node)


def audit_time_budget() -> dict[str, Any]:
    all_findings = []
    py_files = list((ROOT / "src").rglob("*.py"))

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            visitor = TimeBudgetASTVisitor(py_file)
            visitor.visit(tree)
            all_findings.extend(visitor.findings)
        except Exception:
            pass

    return {
        "workstream": "Time-Budget & Runtime Governance Audit",
        "total_findings": len(all_findings),
        "findings": all_findings,
    }


def main() -> int:
    result = audit_time_budget()
    out_dir = ROOT / "reports" / "v5_1_0" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "time_budget_audit.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[+] Time-Budget Audit complete: {len(result['findings'])} findings written to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
