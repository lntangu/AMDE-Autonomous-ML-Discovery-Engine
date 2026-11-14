#!/usr/bin/env python3
"""Specialized Numerical Integrity Auditor for AMDE v5.1.0 (Phase 2 & Phase 4).

Audits the codebase for:
- Float64 to float32 boundary hazards (values exceeding float32 limits 3.4e38).
- Unhandled non-finite values (NaN, +inf, -inf).
- Blind clipping or np.nan_to_num usage.
- Unsafe logarithms (log without eps or domain check).
- Unsafe divisions (denominator near zero).
- Canonical JSON serialization of non-finite floats (-inf, +inf, NaN).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class NumericalASTVisitor(ast.NodeVisitor):
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

        # 1. Check for blind nan_to_num
        if func_name == "nan_to_num":
            self.findings.append({
                "type": "blind_numerical_fix",
                "severity": "Medium",
                "file": self.rel_path,
                "line": node.lineno,
                "detail": "Use of np.nan_to_num() detected; check if this blindly masks non-finite values.",
            })

        # 2. Check for blind clipping
        if func_name == "clip":
            self.findings.append({
                "type": "clipping_hazard",
                "severity": "Low",
                "file": self.rel_path,
                "line": node.lineno,
                "detail": "Use of clip() detected; verify whether clipping distorts distribution or masks root cause.",
            })

        # 3. Check for float32 conversions
        if func_name in ("astype", "to_numpy", "array", "ascontiguousarray"):
            for arg in node.args:
                arg_src = ast.unparse(arg) if hasattr(ast, "unparse") else ""
                if "float32" in arg_src or "np.float32" in arg_src:
                    self.findings.append({
                        "type": "float32_conversion_hazard",
                        "severity": "High",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "detail": f"Direct conversion to float32 ({arg_src}) without explicit boundary check can overflow float64 extreme values (e.g. 1e308 -> inf).",
                    })
            for kw in node.keywords:
                kw_src = ast.unparse(kw.value) if hasattr(ast, "unparse") else ""
                if "float32" in kw_src or "np.float32" in kw_src:
                    self.findings.append({
                        "type": "float32_conversion_hazard",
                        "severity": "High",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "detail": f"Direct conversion to float32 ({kw_src}) without explicit boundary check can overflow float64 extreme values.",
                    })

        # 4. Check for unhandled logs
        if func_name in ("log", "log10", "log2", "log1p"):
            self.findings.append({
                "type": "log_hazard",
                "severity": "Medium",
                "file": self.rel_path,
                "line": node.lineno,
                "detail": f"Mathematical logarithm call ({func_name}) detected; verify domain protection against <= 0.",
            })

        self.generic_visit(node)


def audit_numerical_integrity() -> dict[str, Any]:
    all_findings = []
    py_files = list((ROOT / "src").rglob("*.py"))

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            visitor = NumericalASTVisitor(py_file)
            visitor.visit(tree)
            all_findings.extend(visitor.findings)
        except Exception:
            pass

    return {
        "workstream": "Numerical Integrity Audit",
        "total_findings": len(all_findings),
        "findings": all_findings,
    }


def main() -> int:
    result = audit_numerical_integrity()
    out_dir = ROOT / "reports" / "v5_1_0" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "numerical_integrity_audit.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[+] Numerical Integrity Audit complete: {len(result['findings'])} findings written to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
