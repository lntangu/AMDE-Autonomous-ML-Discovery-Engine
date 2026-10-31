# AMDE 2.0.1 Failure Recovery & Troubleshooting Guide

This guide provides operational incident response runbooks for recovering AMDE 2.0.1 research campaigns from faults, crashes, resource constraints, and OS-level interruptions.

---

## 1. Failure Taxonomy & State Indicators

| State / Incident | Root Cause Category | Automatic Recovery Behavior | Operator Action Required |
| :--- | :--- | :--- | :--- |
| `degraded_retry_*s` | Worker crashed repeatedly | Exponential backoff restart under hourly budget | Monitor logs; check for GPU/CUDA driver instability |
| `needs_attention` | Fatal configuration or startup error | Campaign pauses safely; workers stopped | Review `last_error` in `state.json`; fix configuration |
| `paused_storage` | Free disk dropped below `min_free_gb` | Campaign pauses cooperatively; data preserved | Free disk space on host drive; resume campaign |
| `paused_resource` | Low RAM or GPU temp $\ge$ limit for 3 samples | Campaign pauses cooperatively | Cool down hardware / close external applications; resume |
| `stale_heartbeat` | Worker process hung or deadlocked | Manager sends `STOP` marker; terminates after grace | Automatic: manager restarts worker from last checkpoint |
| `manager_crash` | Manager killed by OS or power loss | Guardian detects dead manager within 30s | Automatic: guardian restarts manager; workers re-adopted |
| `os_reboot` | Host machine restarted | Process tree terminates | Verify checkpoints; run `amde2 resume` |

---

## 2. Standard Incident Runbooks

### Runbook 1: Campaign in `needs_attention` State

#### Symptom:
`amde2 status` shows `Campaign state: needs_attention` with a note such as:
`worker returned configuration error on startup (exit code 2)`.

#### Diagnosis:
1. Read the campaign log:
   ```bash
   python scripts/amde2.py logs --campaign-id <CampaignId> --tail 50
   ```
2. Read the specific worker log:
   ```powershell
   Get-Content reports\amde2\<CampaignId>\logs\worker_quantum.log -Tail 100
   ```
3. Check `state.json` for `last_error`:
   ```powershell
   Get-Content reports\amde2\<CampaignId>\state.json | ConvertFrom-Json | Select-Object -ExpandProperty workers
   ```

#### Recovery:
- **If caused by invalid parameters in plan** (e.g. `--quantum-max-qubits` exceeds RAM or statevector capacity):
  An immutable plan cannot be altered. Create a new campaign with corrected arguments.
- **If caused by CUDA driver failure or GPU device unavailable**:
  Reinstall or restart the NVIDIA graphics driver. Verify with `python scripts/amde2.py doctor --device cuda`.
  Once doctor passes, resume the campaign:
  ```bash
  python scripts/amde2.py resume --campaign-id <CampaignId>
  ```

---

### Runbook 2: Storage Pressure (`paused_storage`)

#### Symptom:
Campaign paused because available free disk on the project drive is below `min_free_gb` (e.g. 5.0 GB).

#### Recovery:
1. Verify free disk space:
   ```bash
   python scripts/amde2.py doctor
   ```
2. Delete temporary files, unused virtual environments, or old test runs outside the AMDE campaign directory.
   *(Never delete active campaign databases or checkpoints!)*
3. Resume the campaign:
   ```bash
   python scripts/amde2.py resume --campaign-id <CampaignId>
   ```

---

### Runbook 3: Recovery After Machine Reboot or Power Loss

#### Incident Scenario:
The operating system rebooted during an active campaign. Guardian, Manager, and Workers were terminated abruptly.

#### Recovery Procedure:
1. **Verify Evidence and Integrity First**:
   Run the verification command before resuming:
   ```bash
   python scripts/amde2.py verify --campaign-id <CampaignId>
   ```
   This ensures:
   - The SQLite databases are not corrupted (runs `PRAGMA integrity_check`).
   - The latest checkpoint has a valid SHA-256 checksum.
2. **Handle Stale Locks**:
   AMDE 2.0.1 automatically detects that the recorded PIDs are no longer alive or match a different process creation timestamp. It archives stale locks to `*.pid-reused.<timestamp>` safely.
3. **Resume Campaign**:
   ```bash
   python scripts/amde2.py resume --campaign-id <CampaignId>
   ```
   The engine reads the last valid checkpoint, recovers RNG states, skips completed trials, and resumes execution seamlessly.

---

### Runbook 4: Resolving Worker Stalls (Stale Heartbeats)

#### Symptom:
A worker thread is hung (e.g., in a native C/CUDA extension call or infinite loop).

#### Built-in Automatic Mitigation:
1. Manager detects that `heartbeat_age > stale_heartbeat_seconds` (default: 900s).
2. Manager places a cooperative `STOP` marker in the worker's control directory.
3. If the worker does not exit within `shutdown_grace_seconds` (default: 900s), the manager terminates the process tree using `psutil`.
4. The manager restarts the worker from its last verified checkpoint.

#### Manual Intervention (if required):
If the manager itself was terminated during a stall:
```bash
# Request clean stop:
python scripts/amde2.py stop --campaign-id <CampaignId> --wait-seconds 60

# If still unresponsive, invoke force-stop:
python scripts/amde2.py force-stop --campaign-id <CampaignId>

# Resume cleanly:
python scripts/amde2.py resume --campaign-id <CampaignId>
```

---

### Runbook 5: Checkpoint Corruption Fallback

#### Symptom:
`verify` reports that `latest.json` is missing or has a mismatched SHA-256 checksum due to abrupt power failure during disk write.

#### Built-in Generational Redundancy:
The checkpoint manager maintains generational backups:
- `checkpoint.json.1`
- `checkpoint.json.2`
- `checkpoint.json.3`

#### Recovery Steps:
1. Locate the run directory: `reports/campaigns/<CampaignId>_classical/runs/<RunId>/`
2. Inspect the latest valid backup:
   ```bash
   ls -la reports/campaigns/<CampaignId>_classical/runs/<RunId>/checkpoints/
   ```
3. Copy the most recent valid generational checkpoint to replace `checkpoint.json` and update `latest.json`:
   ```powershell
   Copy-Item checkpoints\checkpoint.json.1 -Destination checkpoints\checkpoint.json -Force
   ```
4. Re-run verification:
   ```bash
   python scripts/amde2.py verify --campaign-id <CampaignId>
   ```
5. Resume the campaign.

---

### Runbook 6: Windows NTFS File-Lock Collisions (WinError 5 / WinError 32)

#### Background:
On Windows NTFS filesystems, `os.replace` fails with `Access is denied (WinError 5)` or `Sharing violation (WinError 32)` if an indexing service, antivirus scanner, or concurrent process holds a read handle.

#### Built-in Fix in AMDE 2.0.1:
All file updates employ `robust_replace(src, dst)`:
- Exponential backoff starting at 2ms with randomized jitter.
- Up to 12 retries spanning 3.5 seconds.
- Destination permissions verified and reset (`stat.S_IWRITE`) before replacement.

#### Operator Action:
Ensure the AMDE project directory (`AMDE_Autonomous_ML_Discovery_Engine`) is added to Windows Defender / Antivirus exclusion list to prevent background scanners from causing lock latency.
