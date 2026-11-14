#!/usr/bin/env python3
"""Master Full Audit Runner for AMDE.

Orchestrates all specialized audit workstreams, executes each scanner,
captures logs and return codes, and aggregates results.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


AUDIT_SCRIPTS = [
    ("Repository Inventory", "scripts/audit/repository_inventory.py"),
    ("AST Security & Code Quality", "scripts/audit/custom_ast_security_scan.py"),
    ("DataFrame & Preprocessing Integrity", "scripts/audit/dataframe_audit.py"),
    ("ML Correctness & Data Leakage", "scripts/audit/ml_leakage_audit.py"),
    ("Supply-Chain & Dependency Audit", "scripts/audit/dependency_audit.py"),
    ("Concurrency & State Isolation", "scripts/audit/concurrency_audit.py"),
    ("GitHub Actions Security", "scripts/audit/github_workflow_audit.py"),
    ("Packaging & Distribution Integrity", "scripts/audit/package_audit.py"),
    ("Defect Register Consolidation", "scripts/audit/finding_deduplicator.py"),
]


def main() -> int:
    started = time.monotonic()
    print("=" * 70)
    print("STARTING AMDE FULL-REPOSITORY FORENSIC AUDIT")
    print("=" * 70)

    summary: list[dict[str, Any]] = []

    for name, rel_script in AUDIT_SCRIPTS:
        script_path = ROOT / rel_script
        print(f"\n[*] Running {name} ({rel_script})...")
        t0 = time.monotonic()
        try:
            res = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            elapsed = time.monotonic() - t0
            status = "PASS" if res.returncode == 0 else "FAIL"
            print(f"[{status}] {name} in {elapsed:.2f}s")
            summary.append({
                "workstream": name,
                "script": rel_script,
                "status": status,
                "exit_code": res.returncode,
                "elapsed_seconds": round(elapsed, 2),
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip(),
            })
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"[FAIL] {name} encountered error: {exc}")
            summary.append({
                "workstream": name,
                "script": rel_script,
                "status": "FAIL",
                "exit_code": -1,
                "elapsed_seconds": round(elapsed, 2),
                "stdout": "",
                "stderr": str(exc),
            })

    total_elapsed = time.monotonic() - started
    out_dir = ROOT / "reports" / "full_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "audit_execution_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"AUDIT RUN COMPLETE in {total_elapsed:.2f}s")
    print(f"Summary written to {summary_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
