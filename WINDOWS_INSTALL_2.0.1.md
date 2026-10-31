# AMDE 2.0.1 — Windows Long-Running Runtime Guide

Release date: 2026-10-31  
Target Platform: Windows 11 / Python 3.10–3.14 / NVIDIA Quadro P2000 (CUDA 12.6, PyTorch 2.14.0+cu126)

---

## 1. What Changed in 2.0.1

AMDE 2.0.1 hardens the 2.0.0 long-running runtime against filesystem locking collisions, worker restart clock discrepancies, and startup race conditions:
- **Robust NTFS File Replacement**: `robust_replace` eliminates transient `Access is denied (WinError 5)` and `Sharing violation (WinError 32)` during atomic writes of heartbeats, state, and checkpoints.
- **Embedded Persistent Deadlines**: Added `--duration-hours` directly to `amde2 start`. Persisted in `plan.json` and `state.json`; cooperatively monitored by Manager and Guardian without relying on external PowerShell sleep timers.
- **Granular Execution Modes**: Run hybrid (both), classical-only, or quantum-only via `--mode {hybrid, classical, quantum}`.
- **In-Flight Failure Classification**: In-flight exit code 2 occurrences follow the hourly restart budget and backoff rather than immediately shutting down the campaign.
- **Startup Deadline Guard**: 180s startup deadline terminates and reschedules hung worker subprocesses before their first heartbeat.
- **Worker Monotonic Timing**: Restarts now accurately use monotonic clock timing (`time.monotonic()`).

---

## 2. Upgrading Safely

1. **Verify No AMDE Processes Are Running**:
   ```powershell
   Get-Process -Name python, python3 -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, Path
   ```
2. **Protect Existing Data**:
   Ensure `reports\`, `storage\`, `runs\`, and `artifacts\` are untouched.
3. **Extract Release ZIP**:
   Extract `AMDE_2.0.1_Release.zip` into your project root, overwriting changed source files (`src/`, `scripts/`, `pyproject.toml`).
4. **Install / Verify Dependencies**:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade psutil
   ```

---

## 3. Preflight Diagnostics

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path

# Check AMDE version (should report amde 2.0.1):
& $PY run_amde.py --version

# Run full AMDE 2.0 preflight doctor:
& $PY scripts\amde2.py doctor --device cuda --min-free-gb 5 --min-free-vram-mb 1536

# Run quantum acceleration doctor:
& $PY scripts\quantum_doctor.py --device cuda
```

---

## 4. Launching a 6-Hour Measured Run

```powershell
powershell -ExecutionPolicy Bypass -File .\AMDE2_Run_6Hour_Test.ps1 -CampaignId amde2six001 -DurationHours 6
```
This script:
- Verifies version `2.0.1`.
- Invokes `amde2 doctor` and `quantum_doctor`.
- Starts the campaign passing `--duration-hours 6` into the runtime.
- Attaches an auxiliary detached monitoring timer.
- Cleanly stops, verifies, and backs up the campaign upon completion.

---

## 5. Launching an Unbounded 24/7 Infinite Campaign

```powershell
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$CID = "amde2main001"

& $PY scripts\amde2.py start `
    --campaign-id $CID `
    --mode hybrid `
    --seed 20261031 `
    --hours-per-round 0.25 `
    --max-trials-per-round 100 `
    --classical-max-gb 8 `
    --quantum-max-gb 4 `
    --min-free-gb 5 `
    --quantum-device cuda `
    --classical-threads 4 `
    --quantum-cpu-threads 2 `
    --quantum-epochs 6 `
    --quantum-batch-size 16 `
    --quantum-train-size 64 `
    --quantum-val-size 32 `
    --quantum-test-size 64 `
    --quantum-max-configs 3 `
    --quantum-max-qubits 8 `
    --quantum-statevector-mb 256 `
    --memory-reserve-gb 3 `
    --stale-heartbeat-seconds 900 `
    --manager-stale-seconds 180 `
    --stable-after-seconds 600 `
    --max-worker-restarts-per-hour 8 `
    --degraded-retry-seconds 1800 `
    --shutdown-grace-seconds 900 `
    --health-interval-seconds 30 `
    --max-gpu-temperature-c 80
```

---

## 6. Campaign Management & Operations

```powershell
$PY = ".\.venv\Scripts\python.exe"
$CID = "amde2main001"

# Query Status & Persistent Deadline
& $PY scripts\amde2.py status --campaign-id $CID

# Tail Live Combined Logs
& $PY scripts\amde2.py logs --campaign-id $CID --tail 100

# Request Clean Cooperative Stop
& $PY scripts\amde2.py stop --campaign-id $CID --wait-seconds 1800

# Resume a Stopped or Paused Campaign
& $PY scripts\amde2.py resume --campaign-id $CID

# Verify Evidence and Ledger Integrity
& $PY scripts\amde2.py verify --campaign-id $CID

# Create a Cryptographically Verified Backup
& $PY scripts\amde2.py backup --campaign-id $CID
```
