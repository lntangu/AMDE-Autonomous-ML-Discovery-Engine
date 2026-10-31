# AMDE 2.0.0 — Verification Results

Date: 2026-10-30 development build  
Platform: Linux 6.18.57, CPython 3.13.5, CPU-only test environment

## Final release archive checks

| Check | Result |
|---|---|
| ZIP CRC / extraction | PASS — no compressed-data errors |
| Clean-room source manifest | PASS — 201/201 SHA-256 entries |
| Python compile | PASS — `python -m compileall -q src scripts tests` |
| CLI version | PASS — `amde 2.0.0` |
| CLI command help | PASS — start/resume/status/logs/stop/force-stop/verify/backup/restore/doctor |

## Runtime and test checks

| Check | Result |
|---|---|
| AMDE 2 unit tests | PASS — 7/7 |
| Entire unit suite | PASS — 214 tests, 0 failures, 0 errors, 0 skipped; 29.74 s using `scripts/run_unit_tests.py` |
| AMDE 2 lifecycle integration | PASS — 2/2 individually (campaign start/status/stop/resume; Guardian recovery after Manager crash) |
| Legacy Quantum/Hybrid integration | PASS — 4/4 individually, as recorded in the release manifest |
| Backup / verify / restore | PASS — SHA-256, ZIP CRC, SQLite integrity and root rebasing |
| CPU runtime doctor | PASS |
| Quantum statevector doctor on CPU | PASS — finite four-qubit statevector; norm approximately one |
| Aggregate full integration gate | NOT COMPLETED reliably; no overall integration PASS claimed |

## Target machine qualification

| Check | Result |
|---|---|
| Windows 11 / Python 3.14.7 process-tree behavior | NOT RUN for AMDE 2.0 |
| Quadro P2000 / CUDA-enabled PyTorch 2.14.0+cu126 | NOT RUN for AMDE 2.0 |
| GPU temperature / VRAM pressure under sustained load | NOT RUN |
| Windows Task Scheduler logon/restart | NOT RUN |
| 6-hour / 24-hour / 72-hour soak | NOT RUN |

## Reproduction commands

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY scriptserify_manifest.py
& $PY -m compileall -q src scripts tests
& $PY scriptsun_unit_tests.py
& $PY scriptsun_integration_tests.py --timeout 90
& $PY scriptsmde2.py doctor --device cuda
& $PY scripts\quantum_doctor.py --device cuda
```

Run the integration command in a supervised session and preserve its full log. The test runner isolates each case and bounds each case's execution time.

## Interpretation

These results validate the software paths listed above in the development environment. They do not certify the target Windows/GPU system or continuous operation for 24 hours. Software reliability and scientific validity are separate gates; no claim of general ML superiority, causal understanding or physical quantum advantage is made.
