#!/usr/bin/env python3
"""Workstream C: Dependency & Software Supply-Chain Auditor for AMDE.

Inspects pyproject.toml, installed packages, optional extras, and python version compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_dependency_audit() -> dict[str, Any]:
    pyproject_path = ROOT / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")

    dependencies: list[str] = []
    optional_deps: dict[str, list[str]] = {}
    requires_python = ""

    if sys.version_info < (3, 11):
        import tomli as tomllib
    else:
        import tomllib
    data = tomllib.loads(content)

    project_meta = data.get("project", {})
    name = project_meta.get("name", "amde-automl")
    version = project_meta.get("version", "5.0.0")
    requires_python = project_meta.get("requires-python", ">=3.10")
    dependencies = project_meta.get("dependencies", [])
    optional_deps = project_meta.get("optional-dependencies", {})

    # Check installed version of core dependencies
    installed_packages: dict[str, str] = {}
    import importlib.metadata
    for dep in dependencies:
        pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
        try:
            installed_packages[pkg_name] = importlib.metadata.version(pkg_name)
        except Exception:
            installed_packages[pkg_name] = "NOT_INSTALLED"

    return {
        "project_name": name,
        "version": version,
        "requires_python": requires_python,
        "core_dependencies_count": len(dependencies),
        "core_dependencies": dependencies,
        "installed_core_versions": installed_packages,
        "optional_dependency_groups": {k: len(v) for k, v in optional_deps.items()},
        "optional_dependencies": optional_deps,
    }


def main() -> int:
    report = run_dependency_audit()
    out_dir = ROOT / "reports" / "full_audit" / "supply_chain"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dependency_inventory.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# AMDE Software Supply-Chain & Dependency Audit Report",
        "",
        f"- **Distribution Name**: `{report['project_name']}`",
        f"- **Version**: `{report['version']}`",
        f"- **Requires Python**: `{report['requires_python']}`",
        f"- **Core Dependencies**: {report['core_dependencies_count']}",
        "",
        "## Core Dependencies & Installed Status",
        "| Package Declaration | Installed Version | Status |",
        "| :--- | :--- | :--- |",
    ]
    for dep in report["core_dependencies"]:
        pkg = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
        inst = report["installed_core_versions"].get(pkg, "UNKNOWN")
        status = "INSTALLED" if inst != "NOT_INSTALLED" else "MISSING"
        md_lines.append(f"| `{dep}` | `{inst}` | **{status}** |")

    md_lines.extend([
        "",
        "## Optional Dependency Groups",
        "| Group | Package Count |",
        "| :--- | :--- |",
    ])
    for grp, count in report["optional_dependency_groups"].items():
        md_lines.append(f"| `{grp}` | {count} |")

    md_path = out_dir / "dependency_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] Dependency audit written to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
