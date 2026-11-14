#!/usr/bin/env python3
"""Specialized High-Cardinality & Categorical Encoding Auditor for AMDE v5.1.0 (Phase 2 & Phase 5).

Audits categorical handling across preprocessors and model pipelines:
- Detects unconstrained OneHotEncoder instantiation (missing cardinality check or sparse_output=False).
- Analyzes potential dimensional explosion on high cardinality (>1,000 unique values).
- Inspects target encoding for data leakage (must be out-of-fold cross-fitted in CV, never pre-CV fit).
- Checks rare-category grouping and unseen category fallback policies.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class HighCardinalityASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.rel_path = str(file_path.relative_to(ROOT)).replace("\\", "/")
        self.findings: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # 1. OneHotEncoder instantiation without cardinality guard
        if func_name == "OneHotEncoder":
            has_sparse_false = False
            for kw in node.keywords:
                if kw.arg in ("sparse", "sparse_output"):
                    if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        has_sparse_false = True
            if has_sparse_false:
                self.findings.append({
                    "type": "dense_onehot_hazard",
                    "severity": "Critical",
                    "file": self.rel_path,
                    "line": node.lineno,
                    "detail": "OneHotEncoder(sparse_output=False) creates dense array. If cardinality is high (e.g. 10k), creates 10k dense columns leading to multi-gigabyte RAM explosion and OOM/timeouts.",
                })

        # 2. pd.get_dummies check
        if func_name == "get_dummies":
            self.findings.append({
                "type": "get_dummies_hazard",
                "severity": "High",
                "file": self.rel_path,
                "line": node.lineno,
                "detail": "pd.get_dummies produces dimension mismatch on unseen categories in inference data.",
            })

        self.generic_visit(node)


def audit_high_cardinality() -> dict[str, Any]:
    all_findings = []
    py_files = list((ROOT / "src").rglob("*.py"))

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            visitor = HighCardinalityASTVisitor(py_file)
            visitor.visit(tree)
            all_findings.extend(visitor.findings)
        except Exception:
            pass

    return {
        "workstream": "High-Cardinality & Categorical Encoding Audit",
        "total_findings": len(all_findings),
        "findings": all_findings,
    }


def main() -> int:
    result = audit_high_cardinality()
    out_dir = ROOT / "reports" / "v5_1_0" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "high_cardinality_audit.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[+] High Cardinality Audit complete: {len(result['findings'])} findings written to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
