#!/usr/bin/env python3
"""Run AMDE release gates with unit and integration suites in parallel.

Both suites write to independent logs and execute in independent processes.
Integration test cases are each isolated by run_integration_tests.py. The
combined command is the recommended pre-release gate for Windows and Linux.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def terminate_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_preflight(python: str, *, skip_manifest: bool) -> list[dict[str, Any]]:
    checks: list[tuple[str, list[str]]] = [
        ("compileall", [python, "-m", "compileall", "-q", "src", "scripts", "tests"]),
    ]
    if not skip_manifest:
        checks.append(("sha256_manifest", [python, "scripts/verify_manifest.py"]))
    results = []
    for name, command in checks:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        results.append({"name": name, "returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr})
        print(f"[{name}] {'PASS' if completed.returncode == 0 else 'FAIL'}", flush=True)
        if completed.stdout.strip():
            print(completed.stdout.rstrip(), flush=True)
        if completed.stderr.strip():
            print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
        if completed.returncode != 0:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable to test")
    parser.add_argument("--integration-timeout", type=float, default=180.0,
                        help="timeout per isolated integration test (seconds)")
    parser.add_argument("--total-timeout", type=float, default=900.0,
                        help="overall timeout for concurrent suites (seconds)")
    parser.add_argument("--sequential", action="store_true",
                        help="run suites sequentially instead of concurrently")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="skip source manifest checks while developing; not for release")
    args = parser.parse_args()
    if args.integration_timeout <= 0 or args.total_timeout <= 0:
        parser.error("timeouts must be > 0")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "reports" / "test_runs" / f"full_suite_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"AMDE full test suite; output={output}", flush=True)

    preflight = run_preflight(args.python, skip_manifest=args.skip_manifest)
    preflight_ok = all(item["returncode"] == 0 for item in preflight)
    suite_results: dict[str, Any] = {}
    if preflight_ok:
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        src = str(ROOT / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        commands = {
            "unit": [args.python, "scripts/run_unit_tests.py"],
            "integration": [args.python, "scripts/run_integration_tests.py",
                            "--timeout", str(args.integration_timeout)],
        }
        started = time.monotonic()
        active: dict[str, tuple[subprocess.Popen[Any], Any, Path]] = {}
        for name, command in commands.items():
            log_path = output / f"{name}.log"
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log_handle, stderr=subprocess.STDOUT, text=True, **kwargs)
            active[name] = (process, log_handle, log_path)
            print(f"[{name}] started pid={process.pid}", flush=True)
            if args.sequential:
                deadline = time.monotonic() + args.total_timeout
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.25)
                timed_out = process.poll() is None
                if timed_out:
                    terminate_tree(process)
                log_handle.close()
                suite_results[name] = {"returncode": process.returncode, "timed_out": timed_out,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "log": str(log_path.relative_to(ROOT))}
                active.pop(name)
                print(f"[{name}] {'TIMEOUT' if timed_out else 'PASS' if process.returncode == 0 else 'FAIL'}",
                      flush=True)
                if timed_out or process.returncode != 0:
                    break
        if not args.sequential or not suite_results or all(x.get("returncode") == 0 and not x.get("timed_out") for x in suite_results.values()):
            deadline = time.monotonic() + max(0.0, args.total_timeout - (time.monotonic() - started))
            while active and time.monotonic() < deadline:
                for name, (process, _, _) in list(active.items()):
                    if process.poll() is not None:
                        log_handle = active[name][1]
                        log_handle.close()
                        suite_results[name] = {"returncode": process.returncode, "timed_out": False,
                                               "elapsed_seconds": time.monotonic() - started,
                                               "log": str(active[name][2].relative_to(ROOT))}
                        active.pop(name)
                        print(f"[{name}] {'PASS' if process.returncode == 0 else 'FAIL'}", flush=True)
                if active:
                    time.sleep(0.25)
            for name, (process, handle, log_path) in list(active.items()):
                terminate_tree(process)
                handle.close()
                suite_results[name] = {"returncode": process.returncode, "timed_out": True,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "log": str(log_path.relative_to(ROOT))}
                print(f"[{name}] TIMEOUT", flush=True)

    for name, result in suite_results.items():
        log_path = ROOT / result["log"]
        content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        tail = content[-12000:]
        try:
            print(f"\n===== {name.upper()} LOG TAIL =====\n{tail}", flush=True)
        except UnicodeEncodeError:
            print(f"\n===== {name.upper()} LOG TAIL =====\n{tail.encode('ascii', errors='backslashreplace').decode('ascii')}", flush=True)
        result["log_tail"] = tail

    passed = preflight_ok and len(suite_results) == 2 and all(
        result.get("returncode") == 0 and not result.get("timed_out")
        for result in suite_results.values()
    )
    report = {
        "schema_version": "amde-full-test-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": args.python,
        "parallel": not args.sequential,
        "preflight": preflight,
        "suites": suite_results,
        "overall": "PASS" if passed else "FAIL",
        "note": "Unit and integration suites ran in separate processes. Test success does not imply CUDA smoke, long-run stability or scientific generalization.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# AMDE Full Test Suite", "", f"- Overall: **{report['overall']}**",
             f"- Parallel: `{report['parallel']}`", f"- Python: `{args.python}`", "", "## Results", ""]
    for name, result in suite_results.items():
        lines.append(f"- {name}: {'PASS' if result['returncode'] == 0 and not result['timed_out'] else 'FAIL'}; "
                     f"exit={result['returncode']}; timeout={result['timed_out']}; log=`{result['log']}`")
    lines += ["", report["note"], ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFULL_TEST_GATE {report['overall']} report={output / 'REPORT.md'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
