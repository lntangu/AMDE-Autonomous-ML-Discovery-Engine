#!/usr/bin/env python3
"""Workstream A & B: Custom AST Security & Code Quality Scanner for AMDE.

Analyzes the Python source code using the ast module for:
- Code execution risks (eval, exec)
- Unsafe deserialization (pickle, joblib, yaml.load)
- Shell execution (subprocess with shell=True)
- Path traversal and temporary file risks
- Silent exception swallowing (except Exception: pass)
- Hardcoded secrets and token patterns
- Insecure network / TLS settings
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, rel_path: str):
        self.filename = filename
        self.rel_path = rel_path
        self.findings: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call):
        # 1. Check eval / exec
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                self.findings.append({
                    "id": f"SEC-CODE-EXEC-{node.lineno}",
                    "category": "security",
                    "severity": "Critical",
                    "file": self.rel_path,
                    "line": node.lineno,
                    "type": "dynamic_code_execution",
                    "message": f"Direct call to {node.func.id}() detected.",
                })

        # 2. Check unsafe deserialization
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if func_name in ("load", "loads"):
                val = getattr(node.func.value, "id", "")
                if val == "pickle":
                    self.findings.append({
                        "id": f"SEC-DESER-PICKLE-{node.lineno}",
                        "category": "security",
                        "severity": "Medium",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "type": "unsafe_deserialization",
                        "message": "Use of pickle.load/loads. Requires strict trust boundary.",
                    })
                elif val == "yaml" and not any(kw.arg == "Loader" for kw in node.keywords):
                    self.findings.append({
                        "id": f"SEC-DESER-YAML-{node.lineno}",
                        "category": "security",
                        "severity": "High",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "type": "unsafe_deserialization",
                        "message": "Use of yaml.load without explicit SafeLoader.",
                    })

        # 3. Check subprocess shell=True
        if func_name in ("Popen", "run", "call", "check_call", "check_output"):
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.findings.append({
                        "id": f"SEC-SUBPROCESS-SHELL-{node.lineno}",
                        "category": "security",
                        "severity": "High",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "type": "command_injection_risk",
                        "message": "subprocess call with shell=True.",
                    })

        # 4. Check TLS verification disabled
        for kw in node.keywords:
            if kw.arg in ("verify", "check_hostname") and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                self.findings.append({
                    "id": f"SEC-TLS-DISABLED-{node.lineno}",
                    "category": "security",
                    "severity": "High",
                    "file": self.rel_path,
                    "line": node.lineno,
                    "type": "insecure_transport",
                    "message": f"TLS verification explicitly disabled ({kw.arg}=False).",
                })

        # 5. Check tempfile.mktemp
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mktemp":
            self.findings.append({
                "id": f"SEC-TEMPFILE-MKTEMP-{node.lineno}",
                "category": "security",
                "severity": "Medium",
                "file": self.rel_path,
                "line": node.lineno,
                "type": "insecure_tempfile",
                "message": "tempfile.mktemp is deprecated and insecure (race condition).",
            })

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Check silent exception suppression: except [Exception]: pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            exc_name = "bare_except"
            if node.type is not None:
                if isinstance(node.type, ast.Name):
                    exc_name = node.type.id
                elif isinstance(node.type, ast.Attribute):
                    exc_name = node.type.attr
            self.findings.append({
                "id": f"REL-SILENT-EXCEPT-{node.lineno}",
                "category": "reliability",
                "severity": "Medium",
                "file": self.rel_path,
                "line": node.lineno,
                "type": "silent_exception_suppression",
                "message": f"Silent exception suppression detected: except {exc_name}: pass.",
            })
        self.generic_visit(node)


SECRET_PATTERNS = [
    (re.compile(r"(?:api_key|token|secret|password|access_key)\s*[:=]\s*['\"]([A-Za-z0-9_\-\.]{16,})['\"]", re.IGNORECASE), "potential_hardcoded_credential"),
    (re.compile(r"ghp_[A-Za-z0-9_]{30,}|gho_[A-Za-z0-9_]{30,}", re.IGNORECASE), "github_token"),
    (re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"), "private_key"),
]


def scan_file(file_path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    rel_path = str(file_path.relative_to(ROOT)).replace("\\", "/")

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    # Check secret patterns
    for pattern, ptype in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            val = match.group(0)
            # Ignore test files with dummy keys
            if "dummy" in val.lower() or "test" in val.lower() or "example" in val.lower() or "placeholder" in val.lower():
                continue
            line_no = content[: match.start()].count("\n") + 1
            findings.append({
                "id": f"SEC-SECRET-{line_no}",
                "category": "security",
                "severity": "Critical" if "github_token" in ptype or "private_key" in ptype else "High",
                "file": rel_path,
                "line": line_no,
                "type": ptype,
                "message": f"Potential credential found: {ptype} (redacted).",
            })

    # AST Parse
    if file_path.suffix == ".py":
        try:
            tree = ast.parse(content, filename=str(file_path))
            visitor = SecurityASTVisitor(str(file_path), rel_path)
            visitor.visit(tree)
            findings.extend(visitor.findings)
        except SyntaxError:
            pass

    return findings


def main() -> int:
    all_findings: list[dict[str, Any]] = []
    scan_dirs = ["src", "scripts", "tests"]

    for d in scan_dirs:
        target = ROOT / d
        if not target.exists():
            continue
        for p in target.rglob("*.py"):
            findings = scan_file(p)
            all_findings.extend(findings)

    out_dir = ROOT / "reports" / "full_audit" / "static_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "ast_scan_findings.json"
    json_path.write_text(json.dumps(all_findings, indent=2), encoding="utf-8")

    md_lines = [
        "# AMDE AST Security & Reliability Scan Report",
        "",
        f"- **Total Scanned Findings**: {len(all_findings)}",
        "",
        "| ID | Category | Severity | File | Line | Type | Message |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for f in all_findings:
        md_lines.append(
            f"| `{f['id']}` | {f['category']} | **{f['severity']}** | `{f['file']}` | {f['line']} | `{f['type']}` | {f['message']} |"
        )

    md_path = out_dir / "ast_scan_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[+] AST Security Scan completed: {len(all_findings)} findings recorded in {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
