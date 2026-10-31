# AMDE 2.0.1 Full Operations Guide

This guide describes operational procedures for managing autonomous research campaigns in **AMDE 2.0.1**.

---

## 1. Environment & Preflight Verification

Before starting or resuming any research campaign, execute the built-in diagnostic tools to verify system resources, runtime prerequisites, and hardware accelerators.

### 1.1 Run Runtime Doctor
```bash
# For CUDA-enabled environments:
python scripts/amde2.py doctor --device cuda --min-free-gb 5 --min-free-vram-mb 1536

# For CPU-only environments:
python scripts/amde2.py doctor --device cpu --min-free-gb 5
```

The doctor validates:
1. `project_root`: Valid directory structure and write permissions.
2. `runtime_files`: Presence of all scripts, configs, and engine binaries.
3. `disk_reserve`: Verification of disk space against the safety reserve (default: 5.0 GB).
4. `psutil_resource_monitor`: High-resolution RAM and CPU monitoring.
5. `cuda`: PyTorch CUDA initialization, compute capability, and tensor allocation smoke test.
6. `windows_sleep_guard`: Active test of Win32 sleep inhibition API (`SetThreadExecutionState`).

### 1.2 Run Quantum Acceleration Doctor
```bash
python scripts/quantum_doctor.py --device cuda
```
Verifies that quantum statevector kernels execute on the target GPU (e.g., NVIDIA Quadro P2000) with acceptable numerical precision and latency.

---

## 2. Launching Campaigns

### 2.1 The Standard 6-Hour Measured Campaign
To run a bounded 6-hour test with persistent deadline enforcement:

#### Option A: Using the Hardened PowerShell Test Harness
```powershell
powershell -ExecutionPolicy Bypass -File .\AMDE2_Run_6Hour_Test.ps1 -CampaignId amde2six001 -DurationHours 6
```
This harness:
- Performs preflight checks (version, doctor, collision checks).
- Starts the campaign passing `--duration-hours 6` directly to the runtime.
- Spawns a detached auxiliary timer to monitor the run.
- Initiates cooperative shutdown, runs verification, and triggers automatic backup upon completion.

#### Option B: Direct Python CLI Launch
```bash
python scripts/amde2.py start \
  --campaign-id amde2six001 \
  --mode hybrid \
  --duration-hours 6 \
  --seed 20261031 \
  --hours-per-round 0.25 \
  --max-trials-per-round 100 \
  --classical-max-gb 8 \
  --quantum-max-gb 4 \
  --min-free-gb 5 \
  --quantum-device cuda \
  --classical-threads 4 \
  --quantum-cpu-threads 2 \
  --quantum-epochs 6 \
  --quantum-batch-size 16 \
  --quantum-train-size 64 \
  --quantum-val-size 32 \
  --quantum-test-size 64 \
  --quantum-max-configs 3 \
  --quantum-max-qubits 8 \
  --quantum-statevector-mb 256 \
  --memory-reserve-gb 3 \
  --max-gpu-temperature-c 80
```

### 2.2 Continuous 24/7 Infinite Campaign
To run an unbounded campaign that operates indefinitely until explicitly commanded to stop:
```bash
python scripts/amde2.py start \
  --campaign-id amde2main001 \
  --mode hybrid \
  --seed 20261031 \
  --quantum-device cuda
```
*(Omit `--duration-hours` for an infinite campaign).*

### 2.3 Single-Engine Modes
To run only the classical engine or only the quantum engine:
```bash
# Classical only (CPU):
python scripts/amde2.py start --campaign-id amde_classical_01 --mode classical

# Quantum only (CUDA):
python scripts/amde2.py start --campaign-id amde_quantum_01 --mode quantum --quantum-device cuda
```

---

## 3. Monitoring & Inspection

### 3.1 Check Campaign Status
```bash
python scripts/amde2.py status --campaign-id <CampaignId>
```
Reports:
- Overall campaign state (`running`, `stopped`, `degraded`, `paused_storage`, `paused_resource`, `needs_attention`).
- Mode and Campaign Type (`infinite` vs. `timed`).
- Persistent deadline UTC and remaining time.
- Guardian & Manager PIDs, heartbeats, and process liveness.
- Disk free, RAM available, CPU %, and GPU VRAM/temperature.
- Individual worker states, current round, next round, restart counts, and last heartbeat age.

### 3.2 Tail Live Campaign Logs
```bash
# View last 100 log entries across all components:
python scripts/amde2.py logs --campaign-id <CampaignId> --tail 100

# View manager-only log:
Get-Content reports\amde2\<CampaignId>\logs\manager.log -Tail 50 -Wait

# View worker logs:
Get-Content reports\amde2\<CampaignId>\logs\worker_classical.log -Tail 50 -Wait
Get-Content reports\amde2\<CampaignId>\logs\worker_quantum.log -Tail 50 -Wait
```

### 3.3 Attach a Timer to an Already-Running Campaign
If a campaign is already executing and you wish to attach an automatic stop timer:
```powershell
powershell -ExecutionPolicy Bypass -File .\AMDE2_Attach_6Hour_Timer.ps1 -CampaignId <CampaignId> -DurationHours 6
```

---

## 4. Controlled Shutdown & Resumption

### 4.1 Request Cooperative Stop
```bash
# Requests workers to finish current in-flight trial and commit checkpoint:
python scripts/amde2.py stop --campaign-id <CampaignId> --wait-seconds 1800
```
*Note*: This command is strictly idempotent. If the campaign is already stopped, it returns exit code 0 safely.

### 4.2 Force-Stop (Emergency Only)
```bash
# Terminates the campaign process tree immediately if workers are hung:
python scripts/amde2.py force-stop --campaign-id <CampaignId>
```
*Caution*: Use only if cooperative stop times out and manual inspection indicates an unrecoverable deadlocked native thread.

### 4.3 Resume a Stopped Campaign
```bash
python scripts/amde2.py resume --campaign-id <CampaignId>
```
Resumption automatically:
- Validates the immutable plan hash against `plan.json`.
- Restores worker populations, RNG states, and SQLite ledgers.
- Skips already-evaluated trials and continues from the next pending round.

---

## 5. Verification & Backup Procedures

### 5.1 Verify Campaign Evidence Integrity
```bash
python scripts/amde2.py verify --campaign-id <CampaignId>
```
The verification suite confirms:
1. `plan.json` hash matches the persisted plan hash in `state.json`.
2. All completed rounds possess valid, uncorrupted, checksummed checkpoints.
3. Classical and Quantum SQLite ledgers pass `PRAGMA integrity_check`.
4. Artifacts referenced in state are physically present on disk.

### 5.2 Create Verified Campaign Backup
```bash
python scripts/amde2.py backup --campaign-id <CampaignId>
```
Generates a timestamped `.zip` archive inside `reports/amde2/<CampaignId>/backups/` containing:
- Atomic snapshot of SQLite databases via online backup API.
- All checkpoints, plans, and state files.
- SHA-256 cryptographic manifest for tamper detection.

### 5.3 Verify Backup Archive
```bash
python scripts/amde2.py verify-backup --backup-zip <PathToZip>
```

### 5.4 Restore Campaign from Backup
```bash
python scripts/amde2.py restore \
  --backup-zip reports/amde2/<CampaignId>/backups/<BackupFile>.zip \
  --target-campaign-id <NewCampaignId>
```
Restores the complete campaign environment cleanly into a new target campaign ID without starting any processes automatically.
