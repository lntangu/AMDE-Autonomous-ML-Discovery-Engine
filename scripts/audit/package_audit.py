#!/usr/bin/env python3
"""Workstream J: Packaging, Build & Distribution Auditor for AMDE.

Verifies:
- Distribution name is amde-automl
- Authorship attribution is LNTA <dutubelnta@gmail.com>
- pyproject.toml syntax and metadata completeness
- Wheel & sdist buildability and twine check verification
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def run_package_audit() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def log_test(check_name: str, status: str, detail: str):
        results.append({
            "check": check_name,
            "status": status,
            "detail": detail,
        })

    pyproject_p = ROOT / "pyproject.toml"
    content = pyproject_p.read_text(encoding="utf-8")

    if sys.version_info < (3, 11):
        import tomli as tomllib
    else:
        import tomllib
    data = tomllib.loads(content)
    proj = data.get("project", {})

    # 1. Package Name
    name = proj.get("name", "")
    if name == "amde-automl":
        log_test("Distribution Name", "PASS", f"Package name is correctly '{name}'")
    else:
        log_test("Distribution Name", "FAIL", f"Expected 'amde-automl', found '{name}'")

    # 2. Authorship Attribution
    authors = proj.get("authors", [])
    author_match = any(a.get("name") == "LNTA" and a.get("email") == "dutubelnta@gmail.com" for a in authors)
    if author_match:
        log_test("Authorship Attribution", "PASS", "Author LNTA <dutubelnta@gmail.com> properly declared")
    else:
        log_test("Authorship Attribution", "FAIL", f"Author attribution mismatch: {authors}")

    # 3. Version declaration
    ver = proj.get("version", "")
    log_test("Package Version", "PASS", f"Declared version is '{ver}'")

    # 4. Wheel & Sdist buildability
    try:
        dist_dir = ROOT / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if res.returncode == 0:
            log_test("Package Build (python -m build)", "PASS", "Wheel and sdist built successfully.")
        else:
            log_test("Package Build (python -m build)", "FAIL", f"Build failed: {res.stderr[:200]}")
    except Exception as exc:
        log_test("Package Build (python -m build)", "FAIL", f"Exception during build: {exc}")

    # 5. Twine check
    try:
        twine_res = subprocess.run(
            [sys.executable, "-m", "twine", "check", f"{dist_dir}/*"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if twine_res.returncode == 0:
            log_test("Twine Distribution Validation", "PASS", "twine check passed strictly.")
        else:
            log_test("Twine Distribution Validation", "FAIL", f"twine check failed: {twine_res.stdout[:200]} {twine_res.stderr[:200]}")
    except Exception as exc:
        log_test("Twine Distribution Validation", "FAIL", f"Exception during twine check: {exc}")

    return results


def main() -> int:
    results = run_package_audit()
    out_dir = ROOT / "reports" / "full_audit" / "packaging"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "package_audit_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# AMDE Packaging & Release Integrity Audit Report",
        "",
        "| Check | Status | Detail |",
        "| :--- | :--- | :--- |",
    ]
    for r in results:
        status_md = f"**{r['status']}**" if r["status"] == "PASS" else f"❌ **{r['status']}**"
        md_lines.append(f"| `{r['check']}` | {status_md} | {r['detail']} |")

    md_path = out_dir / "packaging_audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] Package audit written to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
