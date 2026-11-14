#!/usr/bin/env python3
"""Phase 0: Baseline Artifacts Generator for AMDE v5.1.0."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
out_dir = ROOT / "reports" / "v5_1_0" / "baseline"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. git_state.json
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
remotes = subprocess.check_output(["git", "remote", "-v"], cwd=ROOT, text=True).strip()
tags = subprocess.check_output(["git", "tag", "-l"], cwd=ROOT, text=True).strip().splitlines()

git_state = {
    "commit": commit,
    "branch": branch,
    "remotes": remotes,
    "tags": tags,
    "clean_working_tree": len(status) == 0,
    "status": status,
}
(out_dir / "git_state.json").write_text(json.dumps(git_state, indent=2), encoding="utf-8")

# 2. environment.json
packages = {}
for pkg in ["numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "flaml", "pytest", "joblib"]:
    try:
        mod = __import__(pkg.replace("-", "_"))
        packages[pkg] = getattr(mod, "__version__", "installed")
    except Exception as e:
        packages[pkg] = f"not installed ({e})"

env_info = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "packages": packages,
}
(out_dir / "environment.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

# 3. repository_inventory.md & repository_inventory.json
from scripts.audit.repository_inventory import build_inventory
inv = build_inventory()
(out_dir / "repository_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")

md_lines = [
    "# AMDE v5.1.0 Baseline Repository Inventory Report",
    "",
    "## Summary Metrics",
    f"- **Baseline Commit**: `{commit}`",
    f"- **Branch**: `{branch}`",
    f"- **Total Tracked Files**: {inv['summary']['total_files']}",
    f"- **Total Lines of Code**: {inv['summary']['total_loc']:,}",
    f"- **Python Source Files**: {inv['summary']['python_source_files']}",
    f"- **Test Suite Files**: {inv['summary']['test_files']}",
    "",
    "## Category Breakdown",
    "| Category | File Count |",
    "| :--- | :--- |",
]
for cat, count in inv["summary"]["category_counts"].items():
    md_lines.append(f"| `{cat}` | {count} |")

md_lines.extend(["", "## Public Interfaces Audited"])
for iface in inv["public_interfaces"]:
    md_lines.append(f"- `{iface}`")

(out_dir / "repository_inventory.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"[+] Baseline artifacts generated in {out_dir}")
