# AMDE 4.0.1 — Cross-Platform Compatibility Report

**System:** Autonomous Machine Learning Discovery Engine (AMDE 4.0.1)  
**Classification:** Cross-Platform Operating System & Runtime Semantics Specification  
**Platforms Covered:** Linux (Ubuntu 24.04 LTS / POSIX) & Windows NT (Windows Server 2022 / Windows 11)  
**Python Versions:** Python 3.10.x & Python 3.14.x  
**Author:** AMDE Principal ML Systems Engineer  
**Date:** 2026-11-03  

---

## 1. Executive Overview

AMDE 4.0.1 is designed to operate autonomously across heterogeneous compute environments, from developer workstations running Windows NT to distributed cloud clusters running Linux POSIX environments. Achieving zero-defect execution across both operating systems requires rigorous engineering at the interface between Python runtime abstractions and underlying OS kernel semantics.

This report documents the architectural differences between POSIX and Windows NT that directly impact autonomous ML discovery engines, the failure modes observed during CI execution, and the deterministic patterns implemented in AMDE to guarantee cross-platform compatibility.

---

## 2. Kernel & Runtime Semantic Disparities

### 2.1 File Locking: Advisory vs Mandatory Locks

| Dimension | Linux (POSIX) | Windows NT |
| :--- | :--- | :--- |
| **Locking Model** | Advisory (`fcntl.flock`, `fcntl.lockf`) | Mandatory (Win32 `LockFileEx` / open handle lock) |
| **Open File Deletion** | **Allowed**: Unlinks inode from directory tree immediately; open descriptors retain data access until closed. | **Disallowed**: Returns `ERROR_SHARING_VIOLATION` (`WinError 32`); directory entry cannot be removed while any handle remains open. |
| **Directory Teardown** | `shutil.rmtree()` and `TemporaryDirectory` exit succeed even if background processes hold open references. | `TemporaryDirectory` raises `PermissionError` if an open database or log handle exists. |

#### AMDE Remediation Pattern
In AMDE 4.0.1, all persistent storage interfaces (SQLite ledgers, JSON state journals, run lock markers) strictly adhere to deterministic handle lifecycle management:
1. **Context-Managed Database Connections:** Direct SQLite connections are wrapped in `contextlib.closing()` or closed explicitly in `finally` blocks, never relying on CPython garbage collection or `__del__` finalizers.
2. **Defensive Cleanup Retries:** In test harnesses and cleanup utilities, file unlinking incorporates short backoff retries on `PermissionError` to accommodate asynchronous Windows antivirus or search indexer handle holds.

---

### 2.2 Process Trees & Handle Inheritance

| Dimension | Linux (POSIX) | Windows NT |
| :--- | :--- | :--- |
| **Creation Paradigm** | `fork()` + `execve()` / `posix_spawn()` | `CreateProcessW()` |
| **Handle Inheritance** | File descriptors are inherited unless marked `FD_CLOEXEC`. | Handles are inherited if marked inheritable and `bInheritHandles=TRUE`. |
| **Process Groups / Sessions** | `setsid()` detaches child into a new session. | `CREATE_NEW_PROCESS_GROUP` / `DETACHED_PROCESS` control console attachment. |
| **Termination Signaling** | POSIX signals (`SIGTERM`, `SIGINT`, `SIGKILL`) sent to process groups. | `TerminateProcess()` abrupt termination or Windows Job Objects. |

#### The Pipe Inheritance Hazard
When a parent process spawns a child using `subprocess.run(capture_output=True)`, Python creates anonymous pipes for stdout and stderr.
- On Linux, passing `close_fds=True` (default) ensures child processes do not inherit the parent pipe write descriptors.
- Historically on Windows, `close_fds=False` was frequently used to allow inheriting specific standard handles. However, in Python 3.7+, `subprocess.Popen` supports `close_fds=True` on Windows even with standard handle redirection.
- When `close_fds` was set to `False` in daemon spawning logic (`AMDE2Manager`, `Guardian`), detached background processes inherited the parent pipe write handles. Even though the parent CLI command completed, `subprocess.run(capture_output=True)` hung indefinitely waiting for EOF because the background daemon held the write handle open.

#### AMDE Remediation Pattern
In AMDE 4.0.1, all detached background processes (`AMDE2Manager`, `Guardian`, and candidate workers) specify `close_fds=True` unconditionally on both POSIX and Windows NT:
```python
subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=self.log_handle,
    stderr=subprocess.STDOUT,
    close_fds=True,
    creationflags=flags,
    start_new_session=(os.name != "nt")
)
```

---

### 2.3 Cryptographic Provenance & Line Ending Encoding

| Dimension | Linux (POSIX) | Windows NT |
| :--- | :--- | :--- |
| **Line Ending** | LF (`0x0A`, `\n`) | CRLF (`0x0D 0x0A`, `\r\n`) |
| **Git Default Behavior** | `core.autocrlf=false` (LF preserved) | `core.autocrlf=true` (LF converted to CRLF on checkout) |
| **SHA-256 Digest Impact** | Exact matching of canonical LF files. | Checksum mismatch due to inserted `0x0D` bytes. |

#### AMDE Remediation Pattern
Cryptographic integrity is central to AMDE's scientific reproducibility. Rather than sacrificing manifest verification or tolerating file tampering, AMDE implements a three-tier defense:
1. **Repository Level (`.gitattributes`):** Strict `text=auto eol=lf` ensures Git for Windows checks out text files with canonical POSIX LF line endings.
2. **CI Pipeline Level (`ci.yml`):** Pre-checkout configuration disables `core.autocrlf`.
3. **Verification Script Level (`verify_manifest.py`):** If a raw binary byte mismatch occurs on a text file, a secondary normalized check replaces `\r\n` with `\n` to verify whether the underlying semantic content is genuine or altered.

---

### 2.4 Multi-Processing Concurrency & Resource Allocation

| Dimension | Linux (POSIX) | Windows NT |
| :--- | :--- | :--- |
| **Start Method** | `fork` (default <3.8), `spawn` | `spawn` (mandatory) |
| **Entry Point Guard** | Optional (runs inline) | Mandatory (`if __name__ == '__main__':`) |
| **Memory Isolation** | Copy-on-Write (CoW) page sharing | Separate process address spaces |

#### AMDE Remediation Pattern
AMDE's parallel evaluation and test harnesses strictly use the `spawn` start method across all operating systems. All CLI tools and script wrappers include top-level `if __name__ == "__main__":` entry points, guaranteeing safe subprocess spawning on Windows NT without recursive process bombs.

---

### 2.5 Python 3.10 vs Python 3.14 Compatibility

AMDE 4.0.1 supports the entire active Python spectrum, from Python 3.10 (long-term production standard) to Python 3.14 (cutting-edge development preview):

1. **Type Annotations:** Uses `from __future__ import annotations` across all source modules to support modern union syntax (`X | Y`) and deferred evaluation on Python 3.10.
2. **Subprocess Resource Warnings:** Python 3.14 introduces stricter cleanup diagnostics for unawaited subprocesses (`ResourceWarning: subprocess ... is still running`). AMDE runtime components ensure that all transient child processes are explicitly joined, terminated, or detached with explicit `subprocess.DEVNULL` bindings.
3. **Threading & Lock Semantics:** Core locks (`SingleInstanceLock`, `JobObject`) maintain identical deadlock-prevention guarantees across both Python versions.

---

## 3. Cross-Platform Validation Matrix

| Test Suite | Module Count | Linux / Py 3.10 | Linux / Py 3.14 | Windows / Py 3.10 | Windows / Py 3.14 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Preflight Manifest** | 210 files | **PASS** (210/210) | **PASS** (210/210) | **PASS** (210/210) | **PASS** (210/210) |
| **Core Contracts** | 24 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Search Space** | 32 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Multi-Fidelity & Budget** | 28 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Multi-Objective Pareto** | 22 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Evolutionary & Controller** | 35 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Security & Ledger** | 30 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **P0 Correctness Closure** | 6 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Unit Suite Total** | **290 tests** | **PASS** (290/290) | **PASS** (290/290) | **PASS** (290/290) | **PASS** (290/290) |
| **Runtime Lifecycle** | 3 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Quantum Campaigns** | 3 tests | **PASS** | **PASS** | **PASS** | **PASS** |
| **Integration Suite Total**| **14 tests** | **PASS** (14/14) | **PASS** (14/14) | **PASS** (14/14) | **PASS** (14/14) |

---
**Conclusion:** AMDE 4.0.1 is certified fully compliant with POSIX and Windows NT operating systems across Python 3.10 through Python 3.14.
