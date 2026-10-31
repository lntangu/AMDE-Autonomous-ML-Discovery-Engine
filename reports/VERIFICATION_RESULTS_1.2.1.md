# AMDE 1.2.1 — Verification results

**Verification date:** 2026-10-30  
**Build environment:** CPython 3.13.5, Linux container  
**Scope:** source-level verification of the Windows PID hotfix; this is not native Windows qualification.

## Automated gates

| Gate | Result | Notes |
|---|---:|---|
| Unit suite | **179/179 PASS** | Includes two new hybrid PID regression tests; 0 failures, 0 errors, 0 skipped |
| Integration cases 1–4 | **4/4 PASS** | Bounded subprocess slice |
| Integration case 5 | **1/1 PASS** | Run separately after an outer runner invocation timed out while scheduling this case |
| Integration cases 6–11 | **6/6 PASS** | Bounded subprocess slice; includes detached `start -> status -> stop` |
| Integration total | **11/11 PASS** | Every case completed successfully; case 5 was verified independently |
| `compileall` | **PASS** | `src`, `scripts`, `tests` |
| Package version | **1.2.1** | `run_amde.py --version` |
| Wheel build | **PASS** | `pip wheel . --no-deps --no-build-isolation`; `amde-1.2.1-py3-none-any.whl` |
| SHA-256 manifest | **PASS** | Verified after extraction from final ZIP |

## Root cause and regression

The 1.2.0 detached controller used `os.kill(pid, 0)` as a PID existence probe. This is non-destructive on POSIX but not on Windows: Python documents that on Windows signal values other than the special console events are passed to `TerminateProcess`. The controller now delegates to the shared platform-aware `default_pid_alive` helper. The Windows fallback uses a typed process handle and a zero-time wait; it does not send a signal. The new unit test confirms that checking a live child process leaves it alive.

## Qualification not established

- Native Windows integration case 10 on the user's actual PC (required).
- Native Windows worker process cleanup and file-handle release after cooperative stop (required).
- CUDA/Quadro P2000 hybrid campaign under 1.2.1 (not yet run).
- One-hour/24-hour stability, power-loss recovery, scientific discovery or external generalization.

Do not start a long hybrid campaign until the updated package passes integration case 10 on the target Windows machine.
