# AMDE 1.1.1 — Maintenance Patch Notes

## Scope

This is a reliability patch on top of AMDE 1.1.0. It does not add a new learning algorithm, alter benchmark definitions, or claim a quantum advantage.

## Corrections

1. The campaign-seed unit test now closes its direct SQLite read connection explicitly. The SQLite connection context manager controls transaction outcome; it does not close the connection. This is relevant to Windows, where an open handle prevents deletion of the temporary `ledger.sqlite3` file.
2. `Ledger` now has idempotent close semantics and closes its connection if schema migration fails during object initialization.
3. Run finalization now attempts lock release even if cleanup raises, and log handlers are closed in a `finally` block around ledger closure.
4. Evaluation process startup closes both pipe endpoints if `Process.start()` fails. Unexpected batch cleanup also closes process handles after terminating remaining workers.
5. The optional quantum extra now declares NumPy. The classical core remains standard-library-only.

## Verification

The Linux build environment has Python 3.13.5 and CPU-only PyTorch 2.10.0. The 1.1.1 source passed compilation, the full unit suite, and all eight integration cases when executed individually. The aggregate integration runner did not complete within the outer execution window in one initial attempt; therefore the patch does not claim a successful end-to-end aggregate runner invocation.

The Windows-specific `PermissionError: [WinError 32]` has not been reproduced or directly confirmed fixed on Windows in this build environment. Run the bundled regression test suite on the target Windows machine before starting long campaigns. The Quadro P2000 CUDA smoke also remains a target-machine check.

## Evidence boundary

- Linux tests validate source behavior on the available interpreter; they do not certify Windows handle-release behavior.
- CUDA hardware is unavailable in the build environment; no new P2000 GPU qualification is claimed.
- No one-hour/24-hour/long-duration soak was run for this patch.
- Existing synthetic QML results remain software-path evidence, not evidence of quantum advantage.
