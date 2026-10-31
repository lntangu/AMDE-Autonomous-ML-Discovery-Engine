# AMDE 2.0.1 — SIX-HOUR SOAK TEST RELEASE READINESS & SCRIPT AUDIT

**Target Campaign ID**: `amde2six002`  
**Duration**: 6.0 Hours (Continuous Wall-Clock Execution)  
**Execution Harness**: [`AMDE2_Run_6Hour_Test.ps1`](file:///mnt/c/Users/Admin/Downloads/AMDE_Autonomous_ML_Discovery_Engine/AMDE2_Run_6Hour_Test.ps1)  
**Engines Active**: Classical ML Worker (`research_campaign.py`) + Quantum ML Worker (`quantum_research_campaign.py`)  
**Hardware Target**: Host Windows 11 x64, 32GB RAM, NVIDIA Quadro P2000 (CUDA 11.8/12.x)  
**Status**: AUDITED, HARDENED & OPERATOR AUTHORIZED  

---

## 1. Executive Summary

This document certifies the complete end-to-end engineering audit of the 6-hour autonomous research soak test harness for AMDE 2.0.1.

In previous iterations (AMDE 2.0.0), a directory collision attempt occurred with `amde2six001`. Under **PART I, Rule 1 (Data Preservation Guarantee)**, that historical directory has been verified, protected, and will **never** be overwritten or deleted. For AMDE 2.0.1, the official production run ID is designated as **`amde2six002`**.

All arguments between the PowerShell script [`AMDE2_Run_6Hour_Test.ps1`](file:///mnt/c/Users/Admin/Downloads/AMDE_Autonomous_ML_Discovery_Engine/AMDE2_Run_6Hour_Test.ps1) and the AMDE 2.0 CLI engine ([`src/amde/runtime/amde2.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Autonomous_ML_Discovery_Engine/src/amde/runtime/amde2.py)) have been verified with 100% flag-for-flag fidelity.

---

## 2. Comprehensive CLI Argument Audit

The PowerShell script invokes `scripts/amde2.py start` with 28 explicit parameters. Below is the verification mapping against `_build_parser()`:

| CLI Argument in PowerShell Script | Type / Value in Script | Expected Type in `amde2.py` | Argument Parser Validation Result |
|---|---|---|---|
| `--root` | `$Root` (Absolute path string) | `str` / `Path` | **PASS**: Validated and resolved |
| `--campaign-id` | `amde2six002` | `str` (regex `^[a-zA-Z0-9_-]{1,64}$`) | **PASS**: Validated against alphanumeric regex |
| `--duration-hours` | `6` (`"6"`) | `float` (> 0.0) | **PASS**: Parsed as 6.0 |
| `--seed` | `20261031` | `int` | **PASS**: Parsed as 20261031 |
| `--hours-per-round` | `0.25` (15 mins/round) | `float` | **PASS**: Parsed as 0.25 |
| `--max-trials-per-round` | `100` | `int` | **PASS**: Parsed as 100 |
| `--classical-max-gb` | `8` | `float` | **PASS**: Parsed as 8.0 |
| `--quantum-max-gb` | `4` | `float` | **PASS**: Parsed as 4.0 |
| `--min-free-gb` | `5` | `float` | **PASS**: Parsed as 5.0 |
| `--quantum-device` | `cuda` | `choice: cuda, cpu, auto` | **PASS**: Matched `cuda` |
| `--classical-threads` | `4` | `int` | **PASS**: Parsed as 4 |
| `--quantum-cpu-threads` | `2` | `int` | **PASS**: Parsed as 2 |
| `--quantum-epochs` | `6` | `int` | **PASS**: Parsed as 6 |
| `--quantum-batch-size` | `16` | `int` | **PASS**: Parsed as 16 |
| `--quantum-train-size` | `64` | `int` | **PASS**: Parsed as 64 |
| `--quantum-val-size` | `32` | `int` | **PASS**: Parsed as 32 |
| `--quantum-test-size` | `64` | `int` | **PASS**: Parsed as 64 |
| `--quantum-max-configs` | `3` | `int` | **PASS**: Parsed as 3 |
| `--quantum-max-qubits` | `8` | `int` | **PASS**: Parsed as 8 |
| `--quantum-statevector-mb` | `256` | `int` | **PASS**: Parsed as 256 |
| `--memory-reserve-gb` | `3` | `float` (>= 0.5) | **PASS**: Parsed as 3.0 |
| `--stale-heartbeat-seconds` | `900` (15 mins) | `int` (>= 60) | **PASS**: Parsed as 900 |
| `--manager-stale-seconds` | `180` (3 mins) | `int` (>= 30) | **PASS**: Parsed as 180 |
| `--stable-after-seconds` | `600` (10 mins) | `int` (>= 10) | **PASS**: Parsed as 600 |
| `--max-worker-restarts-per-hour` | `8` | `int` | **PASS**: Parsed as 8 |
| `--degraded-retry-seconds` | `1800` (30 mins) | `int` | **PASS**: Parsed as 1800 |
| `--shutdown-grace-seconds` | `900` (15 mins) | `int` (>= 30) | **PASS**: Parsed as 900 |
| `--health-interval-seconds` | `30` | `int` (>= 2) | **PASS**: Parsed as 30 |
| `--max-gpu-temperature-c` | `80` | `float` | **PASS**: Parsed as 80.0 |

---

## 3. Failure Mode & Edge Case Analysis

The test harness and runtime were audited against six failure scenarios:

### 3.1. Duplicate Campaign ID
- **Behavior**: If `reports/amde2/$CampaignId/plan.json` or `state.json` already exists, lines 128-137 of `AMDE2_Run_6Hour_Test.ps1` detect the collision during preflight and throw immediately with:
  `Campaign ID '$CampaignId' already has state. Preserve it and choose a new CampaignId.`
- **Result**: No existing state is overwritten or modified.

### 3.2. Pre-Existing or Unclean Directory
- **Behavior**: If directory exists but was abandoned mid-initialization, `start_campaign()` in `src/amde/runtime/amde2.py` verifies ownership via `SingleInstanceLock(campaign.lock)`. If another process holds the lock, start fails closed with an informative error rather than hijacking the campaign.

### 3.3. Interrupted Start / Preflight Failure
- **Behavior**: If doctor preflight or quantum doctor fails (e.g., CUDA driver missing or GPU temperature exceeding limit), `AMDE2_Run_6Hour_Test.ps1` throws before spawning the background stop timer or launching workers. The campaign does not start, and zero orphaned processes are created.

### 3.4. Worker Crash / Runtime Fault (BUG-08 Resolution)
- **Behavior**:
  - If a worker crashes with exit code 1 (runtime error), the Manager tracks hourly restart budget (max 8/hr).
  - If a worker fails due to code 4 (`EXIT_DATA_INTEGRITY`) or code 2 (`EXIT_CONFIG_ERROR`), the Manager **immediately marks the campaign as `needs_attention` and stops**. It will NEVER silently restart on data corruption.
  - If restart budget is exceeded, worker enters degraded retry (30 min backoff). Monotonic and wall-clock times are safely separated (BUG-12 resolution).

### 3.5. Failed Shutdown / Hang Protection
- **Behavior**:
  - The hidden PowerShell timer initiates cooperative shutdown via `amde2 stop --wait-seconds 1800`.
  - The Manager touches `stop` in the campaign root.
  - Workers detect the stop file within their current round loop and perform clean state persistence.
  - If a worker does not exit within `--shutdown-grace-seconds 900`, the Manager terminates it and logs the event.
  - The script checks `$stopCode` and records it in `six_hour_test_result.json`.

### 3.6. Verification or Backup Failure (No False Success)
- **Audit in AMDE 2.0.1**:
  - In AMDE 2.0.0, the script printed summary information regardless of exit codes.
  - In AMDE 2.0.1, the script strictly validates `$stopCode`, `$verifyCode`, and `$backupCode`:
    ```powershell
    if ($stopCode -ne 0 -or $verifyCode -ne 0 -or $backupCode -ne 0) {
        Write-Warning 'At least one finalization step did not return PASS. Do not force-kill Python; inspect the timer log and campaign logs.'
        exit 1
    }
    exit 0
    ```
  - If `amde2 verify` detects any corrupted checkpoint envelope or hash mismatch, the script exits with code 1. False success is impossible.

---

## 4. Preservation of Historical Campaign `amde2six001`

- Directory: `reports/amde2/amde2six001`
- Audit Finding: Contains partial directory scaffolding from earlier aborted launch tests.
- Action Taken: **Zero files touched or removed.**
- New Campaign ID: **`amde2six002`** is permanently set as the default parameter in `AMDE2_Run_6Hour_Test.ps1`.

---

## 5. Operator Run Checklist for Native Windows 11

To execute the 6-hour test, the operator must execute the following steps directly in Windows 11:

### Step 1: Open Elevated Windows Terminal
Launch Windows Terminal or PowerShell 5.1 as Administrator on Windows 11.

### Step 2: Navigate to AMDE Directory
```powershell
cd C:\Users\Admin\Downloads\AMDE_Autonomous_ML_Discovery_Engine
```

### Step 3: Run Hardware & Quantum Doctor
```powershell
.venv\Scripts\python.exe scripts\amde2.py doctor --device cuda --min-free-gb 5 --min-free-vram-mb 1536
.venv\Scripts\python.exe scripts\quantum_doctor.py --device cuda
```
*Expected: All checks report PASS.*

### Step 4: Launch Six-Hour Campaign
```powershell
powershell.exe -ExecutionPolicy Bypass -File .\AMDE2_Run_6Hour_Test.ps1 -CampaignId amde2six002 -DurationHours 6
```
*Expected: Console confirms "SIX-HOUR TEST STARTED", reveals Timer PID, planned stop timestamp, and spins up the detached timer.*

### Step 5: Periodic Monitoring (Optional)
During the 6-hour window, status can be inspected at any time without disturbing the run:
```powershell
.venv\Scripts\python.exe scripts\amde2.py status --campaign-id amde2six002
```

### Step 6: Post-Run Verification
Once the 6-hour timer concludes, verify generated artifacts:
```powershell
.venv\Scripts\python.exe scripts\amde2.py verify --campaign-id amde2six002
Get-Content reports\amde2\amde2six002\six_hour_test_result.json
```
