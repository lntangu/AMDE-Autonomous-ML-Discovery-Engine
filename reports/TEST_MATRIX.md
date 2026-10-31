# AMDE 2.0.0 — Test Matrix and Qualification Boundary

## Current 2.0.0 release

| Area | Test | Result | Boundary |
|---|---|---|---|
| Packaging | Clean-room ZIP extraction | PASS | Release tree checked after extraction |
| Packaging | SHA-256 source manifest | 201/201 PASS | Does not hash the manifest file itself |
| Static | Compile `src`, `scripts`, `tests` | PASS | Development Python 3.13.5 |
| CLI | `run_amde.py --version` | PASS — 2.0.0 | Clean extraction |
| Runtime | AMDE 2 unit module | 7/7 PASS | Linux dev container |
| Unit | Entire unit suite | 214/214 PASS | `scripts/run_unit_tests.py`; CPU-only environment |
| Runtime integration | Campaign lifecycle | PASS | Start → two workers → cooperative stop → resume → stop; fake workers |
| Runtime integration | Manager crash recovery | PASS | Guardian orphan cleanup and Manager restart; fake workers |
| Quantum/Hybrid | Legacy integration subset | 4/4 PASS | Individually invoked, not aggregate suite |
| Persistence | Backup and restore | PASS | ZIP CRC, SHA-256, SQLite integrity and root rebasing |
| Doctor | CPU runtime preflight | PASS | Development container |
| Quantum doctor | CPU statevector | PASS | Four-qubit normalization/finite values |
| Integration | Full aggregate legacy suite | NOT COMPLETED | No aggregate PASS claimed |
| Target hardware | Windows 11 / Python 3.14.7 | NOT RUN | User's PC required |
| Target hardware | Quadro P2000 CUDA | NOT RUN | Driver, PyTorch build, VRAM/thermal behavior required |
| Operations | Task Scheduler startup/recovery | NOT RUN | Target Windows logon session required |
| Soak | 6 h / 24 h / 72 h | NOT RUN | Required before unattended operation |

## Interpretation

A PASS applies only to the specific environment and test path shown. The Linux CPU tests do not validate native Windows process semantics, target CUDA, thermal stability, Task Scheduler or multi-day recovery. A finite synthetic benchmark pass is not evidence of a newly discovered ML algorithm or quantum advantage.

## Historical release evidence

Version-specific records for older releases are preserved in:

- `VERIFICATION_RESULTS_1.2.0.md`
- `VERIFICATION_RESULTS_1.2.1.md`
- `VERIFICATION_RESULTS_1.2.2.md`
- `VERIFICATION_RESULTS_1.3.0.md`
- `RELEASE_NOTES_1.2.0.md`, `RELEASE_NOTES_1.2.1.md`, `RELEASE_NOTES_1.2.2.md`, and `RELEASE_NOTES_1.3.0.md`

Those historical reports must not be interpreted as 2.0 target-machine or 24/7 qualification.
