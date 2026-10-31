# AMDE 2.0.0 — Independent Audit and Release Qualification

## Executive assessment

**Assessment: suitable for a controlled target-machine pilot; not yet certified for unattended 24/7 use.** AMDE 2.0.0 materially upgrades process supervision and data recovery while retaining the AMDE 1.3.0 research engine. Its open-ended loop is a design capability, not proof of multi-day reliability.

### What the release changes

1. A detached Guardian supervises the campaign Manager.
2. The Manager owns an immutable, hash-checked plan and supervises separate Classical and Quantum workers.
3. Campaign-scoped stop markers isolate one campaign from other campaigns and legacy global stop files.
4. Checkpoint and SQLite ledger persistence allow explicit resume after cooperative stop or supported process crashes.
5. Heartbeat monitoring, process identity checks, orphan cleanup and bounded restart/backoff address ordinary process failure.
6. Resource monitoring pauses a campaign at configured safety limits rather than silently relaxing the plan.
7. Online SQLite backup, SHA-256/ZIP validation and guarded restore preserve research evidence.

## Architecture and evidence interpretation

The Classical worker performs the existing AMDE candidate search on CPU. The Quantum worker performs separate synthetic QML experiments using an exact statevector simulator on CPU or CUDA. The two branches are orchestrated together but are not a jointly trained model or a single coupled discovery algorithm.

A successful process-lifecycle test proves that the tested process path starts, records state, stops or recovers under the tested conditions. It does not prove that AMDE discovers a superior algorithm, generalizes to independent real-world data, or exhibits a quantum advantage.

## Software verification (development container)

| Gate | Result | Qualification |
|---|---|---|
| Clean ZIP extraction | PASS | Release archive extracted into a new directory; `src/amde/storage` and runtime package present |
| SHA-256 source manifest | PASS — 201/201 | Verified from the clean extraction |
| Python compilation | PASS | `compileall` for `src`, `scripts`, `tests` |
| AMDE 2 runtime unit module | PASS — 7/7 | Final runtime-specific tests |
| Full unit suite | PASS — 214/214 | `scripts/run_unit_tests.py`; 0 failures/errors/skips, about 30 seconds |
| AMDE 2 lifecycle integration | PASS — 2/2 individually | Start/status/stop/resume and Manager-crash/Guardian-recovery; isolated fake workers |
| Legacy Quantum/Hybrid integration | PASS — 4/4 individually | Recorded release checks; not a full aggregate integration-suite pass |
| Backup/restore | PASS | ZIP CRC, SHA-256 manifest, SQLite integrity and destination-root rebasing tested |
| CPU runtime doctor | PASS | Development-container CPU path |
| Quantum doctor on CPU | PASS | Four-qubit statevector finite and normalized; no physical QPU |
| Complete aggregate integration suite | NOT COMPLETED | Do not infer a full integration-gate PASS |
| Native Windows / CUDA P2000 | NOT RUN | Requires target computer |
| Task Scheduler autostart/restart | NOT RUN | Requires target Windows account/session |
| 6 h / 24 h / 72 h soak | NOT RUN | Required before unattended operation |

The full unit gate was run through the repository's `scripts/run_unit_tests.py` runner with output redirected to a file so spawned evaluation workers could not keep an interactive output pipe open. The separate AMDE 2 lifecycle integration cases passed individually. The aggregate runner did not complete reliably in this build session, so its result is explicitly unqualified.

## Operational failure model

### Covered to a bounded extent

- Python Manager or worker process termination.
- Stale Manager/worker heartbeat.
- Orphaned worker process after supported Manager crash.
- Transient worker exits with bounded restart budgets and backoff.
- Cooperative stop and resume from committed records.
- Disk/RAM/VRAM/temperature pressure causing a safe pause.
- Corrupt or modified backup archive rejected by verification.

### Not guaranteed

- Disk hardware failure, filesystem corruption, sudden power loss during device writes, forced OS update/reboot, user logoff, laptop overheating or hardware failure.
- A CUDA driver hang that prevents the process from being terminated within OS limits.
- A successful scientific discovery or forward progress on every round.
- Restart before an interactive Windows logon: the optional Task Scheduler task is configured for the current user's logon, not as a pre-logon Windows service.

## Required target-machine qualification

Before an unattended run:

1. Extract AMDE 2.0 into the backed-up project and run `doctor --device cuda` plus `quantum_doctor.py --device cuda`.
2. Run a short campaign; check `status`, both worker logs, `verify`, backup and restore.
3. Test `stop` and `resume`; verify both worker states and SQLite integrity.
4. Run a 6-hour supervised soak; inspect resource/thermal history and storage growth.
5. Repeat at 24 hours, then 72 hours. Introduce one controlled Manager termination during a supervised test and verify Guardian recovery without duplicate workers.
6. Register Task Scheduler only after the manual Guardian lifecycle has passed. Reboot/logon behavior must be tested separately.

Keep independent backups on another physical drive or trusted remote storage. A backup stored on the same failing disk is not a disaster-recovery strategy.

## Final conclusion

AMDE 2.0.0 is a major runtime/lifecycle upgrade over 1.3.0 and is ready for staged, supervised testing. It is **not honestly describable as 24/7-certified yet**. The target Windows/P2000 environment and sustained-soak tests are the remaining release-qualification gates.
