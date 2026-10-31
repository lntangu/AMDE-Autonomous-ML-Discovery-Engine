# AMDE 2.0.0 long-running operations

This file is retained as a short pointer for existing AMDE users. Use the full [Windows AMDE 2.0.0 installation and operations guide](../WINDOWS_INSTALL_2.0.0.md) for deployment and exact commands.

## Essential commands

```powershell
$PY = ".\.venv\Scripts\python.exe"
$CID = "amde2main001"

# Preflight
& $PY scripts\amde2.py doctor --device cuda

# Start a new open-ended campaign (full recommended args in WINDOWS_INSTALL_2.0.0.md)
& $PY scripts\amde2.py start --campaign-id $CID --quantum-device cuda

# Operate it
& $PY scripts\amde2.py status --campaign-id $CID
& $PY scripts\amde2.py logs --campaign-id $CID --tail 100
& $PY scripts\amde2.py stop --campaign-id $CID --wait-seconds 1200
& $PY scripts\amde2.py resume --campaign-id $CID
& $PY scripts\amde2.py verify --campaign-id $CID
& $PY scripts\amde2.py backup --campaign-id $CID
```

`rounds=0` makes both research workers open-ended. The guardian is detached; on Windows it inhibits system sleep while active. Cooperative stop waits for the current safe unit to finish and commit. Resume reuses the immutable plan, completed ledger rows and checkpoints. Each campaign uses its own stop markers. No old campaign data is automatically deleted.

## Reliability boundary

A long-running design is not the same as a proven 24/7 system. Run short, 6-hour, 24-hour and 72-hour soak tests on the target Windows/Python/CUDA machine. Verify stop/resume and backup/restore at every stage. Keep independent backups and run on AC power with adequate cooling. AMDE cannot prevent power loss, disk failure, OS restart, user logoff or hardware faults.
