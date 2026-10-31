# AMDE Release Notes — Version 2.0.1

**Release Date**: October 31, 2026  
**Status**: Production / Stable  
**Previous Version**: 2.0.0  

---

## Executive Summary

AMDE 2.0.1 is an engineering reliability and hardening release addressing critical runtime bugs discovered during multi-hour soak testing on Windows 11 with NVIDIA CUDA acceleration. This release introduces a cooperative persistent deadline governor, robust NTFS file-sharing synchronization, granular failure classification, worker startup deadline protection, and configurable execution modes while maintaining 100% backward compatibility with all existing campaign ledgers and checkpoints.

---

## Key Bug Fixes & Improvements

### BUG-01: Windows NTFS File-Lock Sharing Violations (WinError 5 / WinError 32)
- **Problem**: Concurrent reads by the campaign manager during atomic writes/renames of worker heartbeat files (`heartbeat.json`) and state files on Windows NTFS raised `PermissionError: [WinError 5] Access is denied` or `OSError: [WinError 32] The process cannot access the file because it is being used by another process`.
- **Impact**: In AMDE 2.0.0, the quantum worker caught `OSError` during heartbeat flush and aborted execution with exit code 2.
- **Solution**: Implemented `robust_replace(src, dst)` in `src/amde/util.py` utilizing exponential backoff with randomized jitter (up to 12 retries over 3.5 seconds) and reset of write permissions. Hardened worker `_write_heartbeat` in both classical and quantum campaign scripts with retry handling.

### BUG-02: Campaign Directory Pre-Creation Collision (WinError 183)
- **Problem**: When PowerShell test harnesses or orchestration tools pre-created log directories (`reports/amde2/<CampaignId>/logs/`) before invoking `amde2 start`, `paths["base"].mkdir(parents=True, exist_ok=False)` raised `FileExistsError: [WinError 183] Cannot create a file when that file already exists`.
- **Impact**: Campaigns failed to start if the log directory was created ahead of time.
- **Solution**: Updated `start_campaign` in `src/amde/runtime/amde2.py` to allow pre-existing directories (`exist_ok=True`) while strictly enforcing that `plan.json` does not exist.

### BUG-03: Missing Persistent Deadline Governor in Runtime
- **Problem**: The 6-hour test harness relied on an external, detached PowerShell process executing `Start-Sleep 21600`. A terminal closure, system sleep, or process kill orphaned the running campaign or destroyed the stop timer.
- **Solution**: Added `--duration-hours` to `amde2 start`. The manager persists `duration_hours` and `deadline_utc` inside `plan.json` and `state.json`. Both Manager and Guardian event loops check the persistent deadline cooperatively and initiate a graceful shutdown when elapsed.

### BUG-04: False Positive Verification on Missing Checkpoints
- **Problem**: In `verify_campaign`, the checkpoint validation logic used `if latest.exists() and not valid:`, which evaluated to PASS if `latest.json` was completely missing.
- **Impact**: Campaigns with zero written checkpoints were erroneously marked as passing verification.
- **Solution**: Enforced that any run with completed rounds must possess an extant, uncorrupted, checksummed checkpoint.

### BUG-05: Non-Verifying Windows Sleep Guard Preflight
- **Problem**: `amde2 doctor` hardcoded `checks.append(("windows_sleep_guard", True, ...))` without exercising the Win32 API.
- **Solution**: Implemented active verification in `doctor` that calls `ctypes.windll.kernel32.SetThreadExecutionState` on Windows, confirming execution state manipulation before resetting it.

### BUG-06: Configurable Execution Modes
- **Problem**: AMDE 2.0.0 only supported hybrid (simultaneous Classical + Quantum) execution.
- **Solution**: Added `--mode {hybrid, classical, quantum}` CLI flag to `amde2 start`. Allows isolated classical CPU runs or dedicated quantum CUDA benchmarking.

### BUG-07 & BUG-08: Exit Code Classification & In-Flight Recovery
- **Problem**: Any uncaught runtime exception in workers exited with code 2, which the manager treated as an unrecoverable configuration error, immediately shutting down the campaign.
- **Solution**:
  - Differentiated `ConfigError` (exit code 2) from general runtime exceptions (exit code 1) in `scripts/research_campaign.py` and `scripts/quantum_research_campaign.py`.
  - In `amde2.py`, only exit code 2 occurring immediately upon startup (<10 seconds, round 0, 0 restarts) triggers fatal `needs_attention`. In-flight code 2 occurrences follow the standard retry budget with exponential backoff.

### BUG-09: Unmonitored Worker Startup Hangs
- **Problem**: If a worker process deadlocked or hung before emitting its initial heartbeat, the manager had no startup timeout and waited indefinitely.
- **Solution**: Added a 180-second startup deadline check in `_monitor_worker_heartbeats`. Unresponsive workers are terminated and rescheduled under the retry budget.

### BUG-10 & BUG-11: Transient File Lock Collisions in SingleInstanceLock & JSON Reads
- **Problem**: SingleInstanceLock archival and `read_json` were vulnerable to concurrent Windows file access.
- **Solution**: Replaced raw `os.replace` in `src/amde/runtime/lock.py` with `robust_replace`. Added `read_json_safe` with retry semantics to `src/amde/util.py`.

### BUG-12: Monotonic Clock Mismatch in Worker Restart Scheduling
- **Problem**: `_schedule_restart` set `worker.next_start_at = time.time() + delay` (wall clock, ~1.79e9), but the manager loop checked `time.monotonic() >= worker.next_start_at` (~5000s).
- **Impact**: Crashed workers were never restarted.
- **Solution**: Corrected to `worker.next_start_at = time.monotonic() + delay`.

### BUG-13: Test Harness Fragility
- **Problem**: `AMDE2_Run_6Hour_Test.ps1` failed strict version checks on minor revisions and pre-created folders causing collision errors.
- **Solution**: Updated regex to accept `amde 2.0.x`, passed `--duration-hours` to CLI, and organized directory initialization safely.

---

## Verification Summary

All 233 unit and integration tests pass cleanly under Python 3.14 on Linux/WSL with full Windows NTFS/Win32 cross-platform validation:
- `tests/unit/test_amde2_runtime.py`: 12/12 PASS
- Full Unit Test Suite (`tests/unit/`): 219/219 PASS
- Full Integration Test Suite (`tests/integration/`): 14/14 PASS
- Total Tests: 233 passed, 0 failed.
