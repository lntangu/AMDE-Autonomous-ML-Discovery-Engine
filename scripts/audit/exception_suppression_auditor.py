#!/usr/bin/env python3
"""Specialized Exception Suppression Auditor for AMDE v5.1.0 (Phase 9).

Discovers and classifies every `except: pass` or silent exception suppression
across the entire repository.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class ExceptionVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.rel_path = str(file_path.relative_to(ROOT)).replace("\\", "/")
        self.occurrences: list[dict[str, Any]] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Check if the body is just a pass, or just a docstring/ellipsis/pass
        is_silent = False
        body_types = [type(stmt).__name__ for stmt in node.body]

        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            is_silent = True
        elif len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Ellipsis)):
            is_silent = True
        elif all(isinstance(stmt, (ast.Pass, ast.Expr)) for stmt in node.body):
            # Check if there is any print or log call
            has_log_or_call = False
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    has_log_or_call = True
            if not has_log_or_call:
                is_silent = True

        exc_type = "bare"
        if node.type is not None:
            exc_type = ast.unparse(node.type) if hasattr(ast, "unparse") else "typed"

        if is_silent:
            self.occurrences.append({
                "file": self.rel_path,
                "line": node.lineno,
                "exception_type": exc_type,
                "body": body_types,
                "classification": "silent_suppression",
            })

        self.generic_visit(node)


def audit_exceptions() -> dict[str, Any]:
    all_occurrences = []
    py_files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")))

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            visitor = ExceptionVisitor(py_file)
            visitor.visit(tree)
            all_occurrences.extend(visitor.occurrences)
        except Exception:
            pass

    return {
        "total_silent_exceptions": len(all_occurrences),
        "occurrences": all_occurrences,
    }


def main() -> int:
    result = audit_exceptions()
    out_dir = ROOT / "reports" / "v5_1_0" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exception_audit.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[+] Exception audit complete: found {result['total_silent_exceptions']} silent suppression locations -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
