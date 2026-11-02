"""Run each integration case in its own bounded subprocess.

This protects the full release gate from an intermittent spawned-worker/test-runner
hang: every case has a timeout, output is retained in the console, and timeout
cleanup attempts to terminate the worker process tree.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import tempfile
from typing import Sequence

TESTS: tuple[str, ...] = (
    "tests.integration.test_amde2_runtime.AMDE2LifecycleIntegrationTests.test_campaign_start_status_stop_resume_is_campaign_scoped",
    "tests.integration.test_amde2_runtime.AMDE2LifecycleIntegrationTests.test_guardian_recovers_after_manager_process_crash",
    "tests.integration.test_engine.EngineIntegrationTests.test_cli_once_status_report_and_inspect",
    "tests.integration.test_engine.EngineIntegrationTests.test_empty_project_status_and_report_fail_cleanly",
    "tests.integration.test_engine.EngineIntegrationTests.test_explicit_challenge_is_aggregate_only_and_single_use",
    "tests.integration.test_engine.EngineIntegrationTests.test_search_reproducibility_is_independent_of_run_id_and_guards_protocol_hash",
    "tests.integration.test_engine.EngineIntegrationTests.test_stop_file_stops_then_resume_completes",
    "tests.integration.test_engine.EngineIntegrationTests.test_storage_budget_pauses_after_durable_checkpoint",
    "tests.integration.test_quantum_research.QuantumCampaignIntegrationTests.test_finite_quantum_campaign_records_all_methods_and_is_terminal",
    "tests.integration.test_quantum_research.QuantumCampaignIntegrationTests.test_hybrid_supervisor_starts_both_workers_and_stops_cooperatively",
    "tests.integration.test_quantum_research.QuantumCampaignIntegrationTests.test_hybrid_supervisor_treats_stopped_worker_as_cooperative_terminal_not_crash",
    "tests.integration.test_quantum_research.QuantumCampaignIntegrationTests.test_detached_hybrid_control_start_status_and_cooperative_stop",
    "tests.integration.test_external_benchmarks.ExternalBenchmarkIntegrationTests.test_external_csv_cli_writes_hashes_and_fails_closed_on_demo_scope",
    "tests.integration.test_research_lab.ResearchLabIntegrationTests.test_research_lab_register_evaluate_ablation_and_archive_commands",
)

# Print bounded progress so slow native Windows cases do not look silently hung.
PROGRESS_INTERVAL_SECONDS = 10.0


def _terminate_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
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


def _terminate_remaining_group(group_id: int) -> None:
    """Clean up grandchildren left in a POSIX test session after its leader exits."""
    if os.name == "nt":
        return  # Native Windows process-tree cleanup is performed on timeout.
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _command(test_id: str) -> list[str]:
    return [sys.executable, "-m", "unittest", test_id, "-v"]


def run_tests(timeout_seconds: float, start_index: int = 1, end_index: int | None = None) -> int:
    env = os.environ.copy()
    # Integration tests compare behavior, not numerical throughput. Keep child
    # numerical thread pools bounded to reduce oversubscription while several
    # isolated workers and subprocesses are active.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    existing = env.get("PYTHONPATH", "")
    src = os.path.abspath("src")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    failures: list[str] = []
    started_all = time.monotonic()
    end_index = len(TESTS) if end_index is None else int(end_index)
    if start_index < 1 or end_index < start_index or end_index > len(TESTS):
        raise ValueError(f"test range must satisfy 1 <= start <= end <= {len(TESTS)}")
    selected = TESTS[start_index - 1:end_index]

    for index, test_id in enumerate(selected, start_index):
        print(f"[{index}/{len(TESTS)}] RUN {test_id} (timeout={timeout_seconds:.0f}s)", flush=True)
        started = time.monotonic()
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        # Write child output to a file, not PIPE. A test's spawned grandchildren can
        # inherit stdout/stderr; communicate() would then wait for EOF even after the
        # test process exits, which made the release gate itself appear hung.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as log_file:
            try:
                process = subprocess.Popen(
                    _command(test_id),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    **kwargs,
                )
            except OSError as exc:
                failures.append(f"{test_id}: could not launch: {exc}")
                print(f"FAIL to launch: {exc}", flush=True)
                continue

            deadline = time.monotonic() + max(1.0, timeout_seconds)
            next_progress = started + PROGRESS_INTERVAL_SECONDS
            timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if now >= next_progress:
                        print(
                            f"  ... still running ({now - started:.0f}s elapsed; "
                            f"timeout at {timeout_seconds:.0f}s)",
                            flush=True,
                        )
                        next_progress = now + PROGRESS_INTERVAL_SECONDS
            if timed_out:
                _terminate_tree(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

            # A test can exit while a grandchild remains in its isolated POSIX
            # session. Reap that session before allowing the next integration case.
            if not timed_out and os.name != "nt":
                _terminate_remaining_group(process.pid)
            elapsed = time.monotonic() - started
            log_file.flush()
            log_file.seek(0)
            output = log_file.read()
            print(output or "", end="" if output and output.endswith("\n") else "\n", flush=True)
            if timed_out:
                failures.append(f"{test_id}: TIMEOUT after {timeout_seconds:.1f}s; process tree termination requested")
                print(f"FAIL TIMEOUT {test_id}", flush=True)
            elif process.returncode == 0:
                print(f"PASS {test_id} ({elapsed:.2f}s)", flush=True)
            else:
                failures.append(f"{test_id}: exit={process.returncode}")
                print(f"FAIL {test_id} exit={process.returncode} ({elapsed:.2f}s)", flush=True)

    elapsed_all = time.monotonic() - started_all
    print(f"INTEGRATION_GATE range={start_index}-{end_index} total={len(selected)} failed={len(failures)} elapsed={elapsed_all:.2f}s", flush=True)
    for item in failures:
        print(f" - {item}", flush=True)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0, help="per-test timeout in seconds (default: 60)")
    parser.add_argument("--start-index", type=int, default=1, help="first integration case number (1-based)")
    parser.add_argument("--end-index", type=int, default=len(TESTS), help="last integration case number (1-based)")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    try:
        return run_tests(float(args.timeout), int(args.start_index), int(args.end_index))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
