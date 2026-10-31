# AMDE 1.2.0 — Verification results

**Verification date:** 2026-10-30  
**Build environment:** CPython 3.13.5, Linux container  
**Scope:** source-level release checks; not a native Windows or Quadro P2000 qualification.

## Automated gates

| Gate | Result | Notes |
|---|---:|---|
| Unit suite | **177/177 PASS** | `scripts/run_unit_tests.py`; 0 failures, 0 errors, 0 skipped |
| Integration cases 1–4 | **4/4 PASS** | Isolated subprocess slice |
| Integration cases 5–10 | **6/6 PASS** | Isolated subprocess slice; includes stop/resume, storage pause, quantum campaign and hybrid controller regressions |
| Integration case 11 | **1/1 PASS** | External CSV CLI integration test passed in a separate direct invocation (3.06s); the final combined slice timed out at this case in the runner |
| Integration total | **11/11 PASS** | Cases 1–4 and 5–10 passed in bounded slices; case 11 passed separately |
| `compileall` | **PASS** | `src`, `scripts` and `tests` compile |
| External CSV CLI | **PASS** | End-to-end synthetic smoke, with raw scores, protocol hash and reports written |
| External generalization gate | **Correctly NOT PASSED** | Only one synthetic family; deliberately insufficient and selection-contaminated by design |

## External-evaluation smoke result

A synthetic binary-classification demonstration was run with three seeds, candidate `linear_adam`, fixed baseline `mlp_medium`, CPU backend, `max_steps=120`, and `epochs_scale=0.1`. The observed family-level balanced accuracy was 0.62500 for the candidate and 0.93333 for the baseline (paired delta -0.30833). The audit emitted `insufficient_or_selection_contaminated`, with both internal and external generalization gates false. This is a pipeline sanity check, not a research result about external data.

## Additional bounded checks

- 10,000 deterministic random-expression seeds were checked against the 256-node cap; none exceeded the cap.
- New regression tests verify that a worker exiting with campaign status `stopped` is treated as a cooperative terminal event rather than a crash/restart.
- Detached controller lifecycle test verifies `start`, `status` and marker-based cooperative `stop` without Ctrl+C.
- Final integration rerun: cases 1–4 passed (33.88s); cases 5–10 passed (about 34s); case 11 passed independently (3.06s). The combined 5–11 runner invocation hit the outer execution timeout at case 11, but no child process remained and the test itself passed when invoked directly.
- External CSV tests verify manifest/file hashing, deterministic splits, exact duplicate-feature grouping, rejection of conflicting duplicate targets, regression split support, protocol-checked partial resume, and fail-closed behavior when a one-family synthetic demo is presented as evidence.

## Not established by these checks

- Native Windows PowerShell/CMD behavior on the user's PC.
- CUDA execution on the Quadro P2000 using the new 1.2.0 tree. The user's 1.1.1 P2000 CUDA test is historical evidence for that older tree, not a 1.2.0 qualification.
- Long-duration hybrid stability (one hour, 24 hours or 24/7), thermal stability, storage exhaustion recovery or power-loss recovery.
- Better optimizer discovery, broad external generalization, universal generalization, physical quantum execution or quantum advantage.

The first Windows run must use a new 1.2.0 folder and campaign ID. Keep 1.1.1 and its campaign data intact. Follow `WINDOWS_INSTALL_1.2.0.md`; use the detached hybrid controller and do not stop its `.bat` launcher with Ctrl+C.
