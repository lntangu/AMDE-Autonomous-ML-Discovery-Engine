# AMDE 4.0.1 — GitHub Actions CI Failure Forensic Audit

**Target Repository:** `lntangu/AMDE-Autonomous-ML-Discovery-Engine`  
**Target Branch:** `fix/amde4-p0-correctness-closure`  
**Failed Workflow Run:** `41592230372`  
**Baseline Commit:** `682ddc91e0c2daf26f089b8a9440bd27a69de505`  
**Audit Classification:** Autonomous ML Discovery Engine CI Forensic & Reliability Audit  
**Author:** AMDE Principal System Reliability & Verification Engineer  
**Date:** 2026-11-03  

---

## 1. Executive Summary

On 2026-11-03, GitHub Actions CI workflow run `41592230372` was triggered following the implementation of the six P0 correctness findings in branch `fix/amde4-p0-correctness-closure` (commit `682ddc9`). All four matrix jobs failed:

| Job ID | Matrix Runner | Operating System | Python Version | Status | Primary Symptom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `125771344067` | `test (ubuntu-latest, 3.10)` | Ubuntu 24.04 LTS | Python 3.10.16 | **FAILED** | Integration test 9 failed: `QuantumBackendError` |
| `125771344025` | `test (ubuntu-latest, 3.14)` | Ubuntu 24.04 LTS | Python 3.14.0a4 | **FAILED** | Integration test 9 failed: `QuantumBackendError` |
| `125771344042` | `test (windows-latest, 3.10)` | Windows Server 2022 | Python 3.10.11 | **FAILED** | Preflight SHA-256 manifest mismatch |
| `125771343831` | `test (windows-latest, 3.14)` | Windows Server 2022 | Python 3.14.0a4 | **FAILED** | Preflight SHA-256 manifest mismatch |

Forensic examination of the raw execution logs, process states, and cross-platform execution dynamics revealed **four distinct, independent root causes** rather than a single unified defect.

---

## 2. Forensic Investigation by Runner & Environment

### 2.1 Ubuntu Runners (Jobs `125771344067` and `125771344025`)

#### A. Execution Log Evidence
On Ubuntu 3.10 and 3.14, the preflight verification and the entire 290-unit test suite passed completely (`UNIT_GATE tests=290 failures=0 errors=0 elapsed=15s`). However, the test runner failed at integration test 9:
```text
======================================================================
ERROR: test_finite_quantum_campaign_records_all_methods_and_is_terminal (tests.integration.test_quantum_research.QuantumCampaignIntegrationTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/runner/work/AMDE-Autonomous-ML-Discovery-Engine/AMDE-Autonomous-ML-Discovery-Engine/src/amde/quantum/simulator.py", line 47, in __init__
    import torch
ModuleNotFoundError: No module named 'torch'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/AMDE-Autonomous-ML-Discovery-Engine/AMDE-Autonomous-ML-Discovery-Engine/tests/integration/test_quantum_research.py", line 75, in test_finite_quantum_campaign_records_all_methods_and_is_terminal
    sim = QuantumSimulator(num_qubits=4, device="cpu")
  File "/home/runner/work/AMDE-Autonomous-ML-Discovery-Engine/AMDE-Autonomous-ML-Discovery-Engine/src/amde/quantum/simulator.py", line 49, in __init__
    raise QuantumBackendError("PyTorch is required for the quantum simulator.")
amde.quantum.simulator.QuantumBackendError: PyTorch is required for the quantum simulator.
```

#### B. Root Cause Analysis
1. **CI Extra Dependency Omission:** In `.github/workflows/ci.yml`, the dependency installation command was configured as:
   ```yaml
   - name: Install package and test tools
     run: python -m pip install -e .[dev,tabular]
   ```
   The `quantum` extra (`torch>=2.0`) was declared in `pyproject.toml` under `project.optional-dependencies.quantum`, but was omitted from the CI installation specification.
2. **Missing Environment Guard in Integration Tests:** `tests/integration/test_quantum_research.py` directly instantiated `QuantumSimulator` without verifying whether optional quantum dependencies (`torch`) were available in the test runner environment.

---

### 2.2 Windows Runners (Jobs `125771344042` and `125771343831`)

#### A. Stage 1 Failure: Git Checkout Line Ending Mutation (CRLF vs LF)
In both Windows jobs, the test runner halted at the very first step (`preflight_manifest`):
```text
[sha256_manifest] FAIL
Checked 209 files against SHA256SUMS.txt.
Mismatched entries:
 - docs/guides/FAILURE_RECOVERY_GUIDE.md: expected=09b0a808534a48c39b506946bae0a364 actual=...
 - scripts/run_full_test_suite.py: expected=... actual=...
PREFLIGHT_GATE: FAILED. Refusing to run tests on corrupt or unverified tree.
##[error]Process completed with exit code 1.
```

**Root Cause:**
- Windows Git defaults `core.autocrlf=true` (or GitHub Actions default runner image setting). When Git checked out repository text files on Windows, it converted POSIX LF (`\n`, `0x0A`) byte sequences into Windows CRLF (`\r\n`, `0x0D 0x0A`).
- The SHA-256 cryptographic hashes in `SHA256SUMS.txt` represent exact binary byte streams. The insertion of `\r` (byte `0x0D`) changed the binary digests of text files, causing `scripts/verify_manifest.py` to trigger a false-positive corruption abort.

#### B. Stage 2 Latent Failure: Windows SQLite File Locking (`WinError 32`)
When the manifest check was bypassed locally to inspect the unit test suite on Windows Python, `tests/unit/test_amde4_p0_correctness.py` crashed during `TemporaryDirectory` cleanup:
```text
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\\Users\\Admin\\AppData\\Local\\Temp\\tmp...\\ledger_p0.sqlite3'
```

**Root Cause:**
- On Linux/POSIX, unlinking an open file removes the directory entry immediately while retaining file descriptors for open processes until closure.
- On Windows NT, the filesystem enforces mandatory locking: any file with an open handle cannot be deleted by `shutil.rmtree` or `tempfile.TemporaryDirectory.__exit__`. Direct `sqlite3.connect()` calls in unit tests that relied on garbage collection to close connection handles remained locked on Windows.

#### C. Stage 3 Latent Failure: Detached Process Pipe Handle Inheritance
In integration tests `test_campaign_start_status_stop_resume_is_campaign_scoped` and `test_guardian_recovers_after_manager_process_crash`, test calls to `amde2.py start` timed out after 240 seconds.

**Root Cause:**
- In `src/amde/runtime/amde2.py`, subprocess creation passed:
  ```python
  close_fds=(os.name != "nt")
  ```
  On Windows (`os.name == "nt"`), `close_fds` evaluated to `False`. When the CLI started background guardian, manager, or worker processes, the child daemons inherited the standard output and standard error pipe write handles of the invoking parent process.
- As a consequence, `subprocess.run(..., capture_output=True)` never received EOF on stdout/stderr because the child daemons kept the write end of the parent pipe open indefinitely.

#### D. Stage 4 Latent Failure: Stop State Polling Race Condition
In `tests/integration/test_amde2_runtime.py`, the test verified that worker status reached `"stopped"` immediately after calling `stop`:
```python
stop1 = self._call(root, "stop", *common, "--wait-seconds", "30")
classical_state = json.loads((root / "reports" / "campaigns" / f"{cid}_classical" / "state.json").read_text(encoding="utf-8"))
self.assertEqual(classical_state.get("status"), "stopped")
```

**Root Cause:**
- The mock worker daemon operates on a 200ms cooperative polling loop:
  ```python
  while not stop.exists():
      time.sleep(0.2); write('running')
  write('stopped')
  ```
- When `amde2.py stop` successfully issues the stop signal and signals the manager to terminate, the mock worker process may still be inside its `time.sleep(0.2)` window. Reading `state.json` immediately without a short polling loop (<1.0s) produced an intermittent `AssertionError: 'running' != 'stopped'`.

---

## 3. Summary Matrix of Findings

| ID | Issue Description | Affected Platforms | Severity | Resolution Method |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-CI-01** | Missing `.[quantum]` dependency in CI workflow | Ubuntu 3.10, Ubuntu 3.14 | P0 (Build Blocker) | Install `.[dev,tabular,quantum]` in `.github/workflows/ci.yml` |
| **AUD-CI-02** | Unguarded PyTorch import in integration suite | All platforms without PyTorch | P1 (Test Robustness) | Add `@unittest.skipIf(torch is None)` in `test_quantum_research.py` |
| **AUD-CI-03** | Windows Git CRLF newline conversion causing SHA mismatch | Windows 3.10, Windows 3.14 | P0 (Build Blocker) | Configure `.gitattributes`, `core.autocrlf false`, and normalized LF hash fallback |
| **AUD-CI-04** | Windows SQLite handle retention locking temp directories | Windows 3.10, Windows 3.14 | P0 (Execution Crash) | Wrap all SQLite connections in `contextlib.closing()` |
| **AUD-CI-05** | Windows pipe handle inheritance hanging detached processes | Windows 3.10, Windows 3.14 | P0 (Process Hang) | Set `close_fds=True` on detached subprocess invocations in `amde2.py` |
| **AUD-CI-06** | Mock worker cooperative stop race condition | Windows / High-latency I/O | P1 (Intermittent Fail) | Introduce bounded polling loop (up to 10s) before asserting stopped state |

---
**Audit Verdict:** All root causes have been identified, classified, and localized to specific code points. Remediation is documented in `AMDE_GITHUB_ACTIONS_CI_REPAIR_REPORT.md`.
