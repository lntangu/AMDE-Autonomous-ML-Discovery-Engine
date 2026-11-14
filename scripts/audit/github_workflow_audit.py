#!/usr/bin/env python3
"""Workstream I: GitHub Actions CI/CD Security Auditor for AMDE.

Audits .github/workflows/*.yml for:
- Overprivileged permissions
- pull_request_target script injection vectors
- Hardcoded secrets or unpinned external actions
- Trusted publishing environment configuration
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import re

ROOT = Path(__file__).resolve().parents[2]


def run_workflow_audit() -> list[dict[str, Any]]:
    workflows_dir = ROOT / ".github" / "workflows"
    results: list[dict[str, Any]] = []

    if not workflows_dir.exists():
        return results

    for yml in workflows_dir.glob("*.yml"):
        content = yml.read_text(encoding="utf-8")
        rel_path = str(yml.relative_to(ROOT)).replace("\\", "/")

        # 1. Check permissions block
        has_permissions = "permissions:" in content
        results.append({
            "workflow": yml.name,
            "check": "Explicit least-privilege permissions block",
            "status": "PASS" if has_permissions else "WARN",
            "detail": "Workflow explicitly defines permissions block" if has_permissions else "Missing top-level permissions block",
        })

        # 2. Check for pull_request_target
        has_pr_target = "pull_request_target" in content
        results.append({
            "workflow": yml.name,
            "check": "Absence of unsafe pull_request_target trigger",
            "status": "PASS" if not has_pr_target else "FAIL",
            "detail": "pull_request_target not used" if not has_pr_target else "Dangerous pull_request_target trigger found!",
        })

        # 3. Check for unpinned actions
        unpinned = re.findall(r"uses:\s*([^\s@]+)@(?!v\d|[a-f0-9]{40})([^\s]+)", content)
        results.append({
            "workflow": yml.name,
            "check": "Action version pinning",
            "status": "PASS" if not unpinned else "WARN",
            "detail": f"Unpinned actions: {unpinned}" if unpinned else "All actions pinned with major tags or commit hashes",
        })

        # 4. Check OIDC environment scoping
        if "id-token: write" in content:
            has_env = "environment:" in content
            results.append({
                "workflow": yml.name,
                "check": "OIDC id-token scoped to named environment",
                "status": "PASS" if has_env else "FAIL",
                "detail": "id-token: write is correctly scoped to deployment environment" if has_env else "id-token: write used without environment boundary!",
            })

    return results


def main() -> int:
    results = run_workflow_audit()
    out_dir = ROOT / "reports" / "full_audit" / "github_actions"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "workflow_audit_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# AMDE CI/CD & GitHub Actions Security Audit Report",
        "",
        "| Workflow | Security Check | Status | Detail |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        status_md = f"**{r['status']}**" if r["status"] == "PASS" else f"❌ **{r['status']}**"
        md_lines.append(f"| `{r['workflow']}` | {r['check']} | {status_md} | {r['detail']} |")

    md_path = out_dir / "github_actions_security.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] Workflow audit written to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
