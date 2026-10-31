# AMDE 2.0.0 — Release Notes

Release date: 2026-10-30  
Release type: major runtime / operational reliability release

## Executive summary

AMDE 2.0.0 keeps the AMDE 1.3.0 research and discovery engine and replaces the former hybrid campaign supervisor with a campaign-scoped long-running runtime. The release is aimed at open-ended campaigns, controlled stop/resume, crash recovery, checkpoint/ledger durability, resource-aware pausing, and verifiable backup/restore.

This is **not** a new ML architecture-search algorithm, a claim that AMDE discovers the best training algorithm, or evidence of quantum advantage. The Quantum branch is classical CPU/CUDA statevector simulation.

## Major additions

- **Detached Guardian + Campaign Manager:** the Guardian supervises the Manager; the Manager supervises separate Classical CPU and Quantum CPU/CUDA workers.
- **Campaign-isolated controls:** each campaign has its own immutable plan, state, locks and stop markers. AMDE 2 commands do not create the old project-global `runs/STOP`, `runs/QUANTUM_STOP` or `runs/HYBRID_STOP` files.
- **Safe stop and resume:** workers receive campaign-specific cooperative stop requests; committed Classical checkpoints and Quantum SQLite trial rows are reused on resume. `force-stop` is an explicit emergency command, not the normal stop path.
- **Crash recovery:** heartbeat watchdogs, PID identity checks, orphan-worker inspection, bounded exponential backoff and persisted restart budgets. Invalid configuration/backend exit codes fail closed for operator inspection.
- **Resource governor:** free-disk and available-RAM checks, GPU VRAM/temperature telemetry, conservative resource reserves and separate Classical/Quantum CPU thread caps. Safety-limit breaches pause rather than silently switching a requested CUDA campaign to CPU.
- **Bounded logs:** rotating worker, Manager and Guardian logs.
- **Evidence-preserving backups:** SQLite online backup, SHA-256 manifest, ZIP CRC validation, SQLite integrity checks and guarded restore with destination-root rebasing. Runtime locks and stale PIDs are not restored.
- **Windows operations:** Guardian requests system-sleep inhibition while active; optional Task Scheduler integration restarts the Guardian at the current user's logon and after task failure.
- **Diagnostics:** runtime `doctor`, campaign `verify`, backup verification and structured status/log commands.

## Compatibility and migration

- Stop all AMDE 1.3.0 campaigns before replacing source files.
- Back up the complete existing project, including `.venv`, `runs`, `artifacts`, `reports`, `storage`, `config` and local datasets.
- Extract the release archive over the existing project root. The archive intentionally excludes `.venv`, user campaign output, `runs/`, and generated artefacts.
- Reuse the existing Python environment only if its dependencies work. AMDE 2.0 requires `psutil`; CUDA workloads require the matching CUDA-enabled PyTorch and the project's numerical/ML dependencies.
- Existing 1.3.0 worker checkpoints are not silently adopted into an AMDE 2 campaign. Preserve them as historical evidence and start a fresh 2.0 campaign ID.

## Qualification boundary

The development-container release checks include the 214-test unit gate, targeted AMDE 2 lifecycle tests, backup/restore verification, CPU runtime diagnostics, manifest validation and compilation. The full aggregate legacy integration gate did not complete reliably in the final build session; therefore no aggregate integration-suite PASS is claimed.

Native Windows 11 process handling, Python 3.14.7, NVIDIA Quadro P2000 CUDA/thermal behavior, Task Scheduler restart behavior and 6/24/72-hour soak tests remain unverified on the target computer. The software supports open-ended operation; **unattended 24/7 reliability is not certified until the target machine passes staged soak tests**.

See `INDEPENDENT_AUDIT_AND_RELEASE_QUALIFICATION.md`, `VERIFICATION_RESULTS.md`, `TEST_MATRIX.md`, and `../WINDOWS_INSTALL_2.0.0.md`.
