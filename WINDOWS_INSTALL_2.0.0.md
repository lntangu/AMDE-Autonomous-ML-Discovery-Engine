# AMDE 2.0.0 — Windows long-running runtime

Release date: 2026-10-30

## What changed

AMDE 2.0.0 is a major **runtime and reliability** release. It preserves the AMDE 1.3.0 classical search engine, benchmark definitions, discovery archive, research tooling and quantum-statevector experiments. It adds a new campaign manager/guardian around those research workers. This release does not claim a new ML discovery algorithm or a quantum advantage.

The 2.0 runtime provides:

- Open-ended Classical CPU and Quantum CUDA/CPU workers (`rounds=0`).
- Campaign-scoped stop markers; stopping one AMDE 2 campaign does not create project-wide legacy stop markers or stop another campaign.
- Immutable, checksummed campaign plans and atomic state files.
- Checkpoint/SQLite-ledger resume. Completed trial rows are retained; interrupted/incomplete work can be retried by the underlying worker.
- A detached guardian that monitors and restarts a crashed manager, with bounded exponential backoff.
- A manager that restarts transient worker failures, persists restart budgets, and enters a cooldown after repeated failures. Configuration/backend exit code 2 fails closed and requires operator inspection.
- Heartbeat monitoring, orphan-worker recovery, PID-identity checks against PID reuse, and cooperative shutdown before a stale process is terminated.
- Disk, RAM, VRAM and GPU-temperature observation; safe pause when configured limits are exceeded. Classical and Quantum workers receive separate CPU thread caps to prevent long-run oversubscription.
- Rotating worker logs (20 MiB per file, 5 backups), plus bounded manager/guardian log rotation.
- SQLite online backup, SHA-256 manifest, ZIP CRC and SQLite integrity verification, and guarded restore.
- Windows Task Scheduler logon startup and a Windows sleep-inhibition request while the guardian is alive.

## 1. Upgrade safely

1. Stop any AMDE 1.3.0 campaign before replacing source files. Do not overwrite source while any AMDE worker is running.
2. Create a full backup of the existing project folder, including `.venv`, `runs`, `artifacts`, `reports`, `storage`, `config` and user datasets. Keep the existing 1.3.0 evidence.
3. Extract the contents of `AMDE_2.0.0_Research_Runtime.zip` into the existing project root and allow the 2.0 source files to replace the old source files. The release archive does not contain `.venv`, `runs`, `artifacts` or personal campaign output.
4. Keep using the existing `.venv` only if it already contains the required Python packages. AMDE 2.0 requires `psutil>=5.9`; the CUDA quantum worker also needs the working CUDA-enabled PyTorch installation, NumPy and scikit-learn used by your 1.3.0 project.

If you are installing into a new folder instead, create a new virtual environment there and reinstall the same Python/PyTorch/CUDA-compatible packages. A Windows virtual environment should not be copied between folders because its launchers can contain absolute paths.

## 2. Preflight

Run PowerShell from the project root:

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path

& $PY run_amde.py --version
if ($LASTEXITCODE -ne 0) { throw "AMDE package check failed" }

& $PY scripts\amde2.py doctor --device cuda --min-free-gb 5 --min-free-vram-mb 1536
if ($LASTEXITCODE -ne 0) { throw "AMDE 2.0 preflight failed; do not start a CUDA campaign" }

& $PY scripts\quantum_doctor.py --device cuda
if ($LASTEXITCODE -ne 0) { throw "Quantum CUDA doctor failed" }
```

If you intentionally want a CPU-only campaign, use `doctor --device cpu` and start with `--quantum-device cpu`. Do not silently replace a requested CUDA device with CPU.

## 3. Start an open-ended hybrid campaign

These are conservative starting values for a machine with 32 GiB RAM and a 4 GiB Quadro P2000. They are not throughput guarantees. The campaign has no overall time limit; per-round budgets only bound individual Classical rounds.

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$CID = "amde2main001"

& $PY scripts\amde2.py start `
    --campaign-id $CID `
    --seed 20261030 `
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

if ($LASTEXITCODE -ne 0) { throw "AMDE 2.0 did not start cleanly" }
```

The guardian and manager detach from the PowerShell session. You may close that PowerShell window after `start` reports a guardian PID. The separate Classical and Quantum thread caps limit CPU oversubscription on the 6-core i7-8850H; change them only after measuring sustained throughput and thermals. Keep the laptop plugged into AC and ensure unobstructed cooling. The guardian requests that Windows not enter system sleep while the guardian process is alive. It cannot prevent power loss, forced OS updates/restarts, critical-battery shutdown, user logoff, hardware faults or disk failure.

## 4. Control commands

```powershell
$PY = ".\.venv\Scripts\python.exe"
$CID = "amde2main001"

# Live state, worker PIDs, heartbeat age, memory/disk/GPU health
& $PY scripts\amde2.py status --campaign-id $CID

# Tail guardian, manager, Classical and Quantum logs
& $PY scripts\amde2.py logs --campaign-id $CID --tail 100

# Request a cooperative stop; workers finish their current safe unit and commit state
& $PY scripts\amde2.py stop --campaign-id $CID --wait-seconds 1200

# Continue the same immutable campaign and existing ledgers/checkpoints
& $PY scripts\amde2.py resume --campaign-id $CID

# Check campaign plan, checkpoints and SQLite database integrity
& $PY scripts\amde2.py verify --campaign-id $CID
```

A cooperative stop is intentionally not an instantaneous kill. It is honored at safe boundaries: a current isolated Classical trial or Quantum model evaluation may finish first so its result can be committed. If stop exceeds the wait limit, inspect `status` and `logs`; use `force-stop` only after deciding that losing the in-flight unit is acceptable:

```powershell
& $PY scripts\amde2.py force-stop --campaign-id $CID
```

The force-stop command checks process identity before terminating a process tree. Never use broad `taskkill /IM python.exe /F`, because that can terminate unrelated Python applications.

## 5. Backup and restore

Create a consistent online backup while a campaign is running or stopped:

```powershell
& $PY scripts\amde2.py backup --campaign-id $CID
```

The ZIP includes the campaign plan/state/logs, Classical and Quantum campaign state, run checkpoints and ledgers, per-round reports, relevant artifacts, project `config/` and `data/` files, and the global discovery archive when present. SQLite databases are copied using SQLite's online backup API. Transient process locks and stop markers are intentionally excluded. Backups are additional copies; AMDE does not automatically delete old run evidence.

Verify a backup:

```powershell
$ZIP = Get-ChildItem ".\reports\amde2\$CID\backups\*.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $PY scripts\amde2.py verify-backup $ZIP.FullName
```

Restore into a clean project root (or an existing root only with deliberate overwrite). Restore verifies the archive first and does not start a worker:

```powershell
& $PY scripts\amde2.py restore "D:\AMDE_Backups\amde2main001_YYYYMMDD_HHMMSS.zip" --root "D:\AMDE_Restored"
```

The restore process rebases the immutable campaign plan to the destination project root, clears stale process identities and stop markers, and leaves the campaign stopped. After restoring the matching AMDE 2.0 source into that project root, run `doctor`, `verify` and `status`, then `resume`. `--overwrite` is an explicit destructive option; take a separate backup before using it.

## 6. Optional automatic startup after Windows logon

Register a per-campaign Task Scheduler task. This is optional; the detached guardian already keeps running while the current Windows session and machine remain up.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_amde2_autostart.ps1 `
    -CampaignId "amde2main001" `
    -PythonPath ".\.venv\Scripts\python.exe" `
    -AdoptCurrentCampaign
```

The `-AdoptCurrentCampaign` option cooperatively stops the manually launched guardian, then resumes the same campaign through Task Scheduler. The task runs at the current user's logon and is configured to restart the guardian after failure. It does not run before the user logs in and it is not a Windows service.

Remove the scheduled task (this does not stop or delete the campaign):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\uninstall_amde2_autostart.ps1 -CampaignId "amde2main001"
```

## 7. Persistent files

- `reports/amde2/<campaign-id>/plan.json`: immutable plan and SHA-256-compatible canonical hash.
- `reports/amde2/<campaign-id>/state.json`: manager state, worker state, health and heartbeat.
- `reports/amde2/<campaign-id>/guardian.json`: guardian PID, state and restart history.
- `reports/amde2/<campaign-id>/control/`: campaign-local stop markers and locks.
- `reports/amde2/<campaign-id>/logs/`: guardian/manager logs and rotating worker logs.
- `reports/campaigns/<campaign-id>_classical/`: Classical round state, leaderboard and report.
- `reports/quantum_campaigns/<campaign-id>_quantum/`: Quantum round state, SQLite ledger, leaderboard and report.
- `runs/<campaign-id>_classical_r*/`: per-round Classical SQLite ledger and checksummed checkpoints.
- `artifacts/<campaign-id>_classical_r*/`: per-round artefacts.

## 8. Operational and scientific limits

- This is an open-ended campaign manager, not an unbounded resource allocator. Each worker has disk caps; the host has a free-disk reserve; RAM and GPU thermal pressure can pause a campaign.
- `max-worker-restarts-per-hour` puts a worker into a cooldown after repeated failures; transient retries continue. A configuration/backend exit code is fail-closed and requires inspection.
- The watchdog uses heartbeats to detect a non-responsive manager/worker. It requests cooperative stop first. If a worker ignores the stop beyond the configured grace period, the manager can terminate that worker tree; the in-flight operation may need to be repeated from the last committed ledger/checkpoint.
- Checkpoints and SQLite protect completed work, but no local application can guarantee survival from disk failure, filesystem corruption, power loss during hardware writes, or a failing storage device. Keep independent backups on another physical drive/cloud location.
- The included quantum branch is a classical statevector simulation on CPU/GPU, not physical QPU execution. Synthetic benchmark results do not establish quantum advantage or universal ML improvement.
- A short smoke test and unit/integration tests do not prove 24-hour or 7-day reliability. Qualification must be performed on the exact Windows/Python/PyTorch/driver/GPU installation: start with a short pilot, then 6-hour, 24-hour and 72-hour soak runs with stop/resume and backup/restore tests.

## 9. Release build qualification (this development environment)

The package was built and tested in a Linux development container using Python 3.13, not on the target Windows 11 / Python 3.14.7 / Quadro P2000 machine. A complete 213-test unit gate passed on the pre-final 2.0 snapshot. The final runtime additions were then checked with the 7-test AMDE 2 unit module (all passed), the AMDE 2 lifecycle integration (`start → two workers → cooperative stop → resume → stop`), a separate manager-crash/guardian-recovery integration run, backup/restore with SQLite integrity and SHA-256 checks, four legacy Quantum/Hybrid integration tests, and selected legacy tests. A complete 214-test rerun and the aggregate legacy integration runner did not complete reliably in the final build session, so no final-tree full-unit or all-integration-gate PASS is claimed. Native Windows Task Scheduler, Windows process-tree recovery, CUDA P2000 thermal behavior and 24-hour soak qualification remain unverified and must be tested on the target machine before unattended production use.
