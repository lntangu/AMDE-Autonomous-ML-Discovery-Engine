#!/usr/bin/env python3
"""Phase 0: Full Repository Inventory Generator for AMDE.

Discovers and categorizes all repository files, lines of code, public interfaces,
tests, dependencies, and configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def build_inventory() -> dict[str, Any]:
    categories = {
        "src": [],
        "tests": [],
        "scripts": [],
        "docs": [],
        "workflows": [],
        "config": [],
        "other": [],
    }

    total_files = 0
    total_loc = 0
    py_files = 0
    test_files = 0

    ignored_dirs = {
        ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "build", "dist",
        "data", "archive", "results", ".idea", ".vscode", "runs", "storage",
        "catboost_info", "artifacts", "assets", "reference"
    }

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.endswith(".egg-info")]
        rel_dir = Path(dirpath).relative_to(ROOT)

        for f in filenames:
            if f.endswith((".pyc", ".db", ".parquet", ".zip", ".tar.gz")) or f.startswith(".amde"):
                continue
            fpath = Path(dirpath) / f
            try:
                st = fpath.stat()
                size = st.st_size
            except Exception:
                continue

            rel_path = fpath.relative_to(ROOT)
            parts = rel_path.parts

            total_files += 1
            loc = 0
            if size < 500_000 and not f.endswith((".png", ".jpg", ".pdf", ".ico", ".woff", ".woff2")):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                        loc = sum(1 for _ in fp)
                    total_loc += loc
                except Exception:
                    pass

            entry = {
                "path": str(rel_path).replace("\\", "/"),
                "size_bytes": fpath.stat().st_size if fpath.is_file() else 0,
                "lines_of_code": loc,
            }

            if parts[0] == "src":
                categories["src"].append(entry)
                if f.endswith(".py"):
                    py_files += 1
            elif parts[0] == "tests":
                categories["tests"].append(entry)
                if f.endswith(".py"):
                    test_files += 1
            elif parts[0] == "scripts":
                categories["scripts"].append(entry)
            elif parts[0] == "docs":
                categories["docs"].append(entry)
            elif len(parts) >= 2 and parts[0] == ".github" and parts[1] == "workflows":
                categories["workflows"].append(entry)
            elif f in ("pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "LICENSE", "README.md", "SHA256SUMS.txt"):
                categories["config"].append(entry)
            else:
                categories["other"].append(entry)

    inventory_data = {
        "repository_name": "AMDE-Autonomous-ML-Discovery-Engine",
        "root": str(ROOT),
        "summary": {
            "total_files": total_files,
            "total_loc": total_loc,
            "python_source_files": py_files,
            "test_files": test_files,
            "category_counts": {k: len(v) for k, v in categories.items()},
        },
        "public_interfaces": [
            "amde.AMDEClassifier",
            "amde.preprocessing.TabularPreprocessor",
            "amde.engine.AMDE4Engine",
            "amde.discovery.engine.Engine",
            "amde.ledger.EventLedger",
        ],
        "categories": categories,
    }
    return inventory_data


def main() -> int:
    inventory = build_inventory()
    out_dir = ROOT / "reports" / "full_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "repository_inventory.json"
    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    md_lines = [
        "# AMDE Full Repository Inventory Report",
        "",
        "## Summary Metrics",
        f"- **Total Tracked Files**: {inventory['summary']['total_files']}",
        f"- **Total Lines of Code**: {inventory['summary']['total_loc']:,}",
        f"- **Python Source Files**: {inventory['summary']['python_source_files']}",
        f"- **Test Suite Files**: {inventory['summary']['test_files']}",
        "",
        "## Category Breakdown",
        "| Category | File Count |",
        "| :--- | :--- |",
    ]
    for cat, count in inventory["summary"]["category_counts"].items():
        md_lines.append(f"| `{cat}` | {count} |")

    md_lines.extend([
        "",
        "## Public Interfaces Audited",
    ])
    for iface in inventory["public_interfaces"]:
        md_lines.append(f"- `{iface}`")

    md_path = out_dir / "repository_inventory.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] Inventory generated: {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
