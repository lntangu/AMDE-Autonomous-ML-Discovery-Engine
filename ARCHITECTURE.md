# AMDE 2.0.1 System Architecture Specification

## 1. Executive Overview

The **Autonomous ML Discovery Engine (AMDE)** version 2.0.1 is an offline, safety-bounded, autonomous machine learning research platform engineered for 24/7 continuous operation, crash recovery, and strict experimental integrity.

AMDE orchestrates two independent scientific discovery branches under a resilient supervisor:
1. **Classical ML Discovery Branch**: Autonomous search over algorithmic machine learning pipelines, DSL/AST genomes, feature transforms, loss functions, and optimization heuristics. Evaluated with Pareto multi-objective selection, strict train/validation splits, and sealed generalization test sets.
2. **Quantum ML Discovery Branch**: Parameterized quantum circuit discovery, Variational Quantum Classifiers (VQC), Quantum Kernel Methods (QKM), statevector simulation, and Grover search optimization. Runs accelerated on NVIDIA CUDA GPUs (e.g. NVIDIA Quadro P2000 with PyTorch 2.14.0+cu126) with automatic fallback to CPU statevectors.

AMDE 2.0.1 hardens the runtime process architecture against Windows NTFS lock collisions, monotonic timer misalignments, unobserved startup hangs, and transient worker failures.

---

## 2. Process Topology & Hierarchy

```
[Operating System / Host Session / Logon]
   │
   ▼
[AMDE Guardian Process] (amde2 guardian / PID: G)
   │  - Windows Sleep Inhibition (SetThreadExecutionState: ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
   │  - Manager Process Watchdog & Heartbeat Monitor
   │  - Process Identity Verification (PID + Creation Time + Executable + Campaign Token)
   │  - Persistent Manager Restart Budget (Cooldown & Backoff)
   │  - Graceful Shutdown Propagation & Orphan Prevention
   │
   ▼
[AMDE Manager Process] (amde2 run / PID: M)
   │  - Immutable Campaign Plan Hash Validation (`plan_hash`)
   │  - Persistent Deadline & Duration Governor (`--duration-hours`, `deadline_utc`)
   │  - Resource Sampler (RAM, Disk Free, CPU %, GPU Temp C, GPU VRAM)
   │  - Dynamic Mode Orchestrator (`hybrid`, `classical`, `quantum`)
   │  - Monotonic Restart Scheduler (`worker.next_start_at = time.monotonic() + delay`)
   │  - Worker Heartbeat & Startup Deadline Monitor (180s startup grace)
   │  - Exit Code Failure Classifier (Startup fatal vs. In-flight transient)
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
[Worker A: Classical]          [Worker B: Quantum]           [Auxiliary / Log Relays]
  - Subprocess PID: W_C          - Subprocess PID: W_Q         - Subprocess PIDs: R_C, R_Q
  - CPU-bounded threadpool       - CUDA / GPU accelerated      - stdout / stderr multiplex
  - SQLite Ledger (WAL mode)     - SQLite Ledger (WAL mode)    - Rotating log chunks
  - Checkpoint Manager           - Statevector simulation      - Unbuffered pipes
  - Sealed Evaluator             - Trial deduplication
```

---

## 3. Subsystem Breakdown

### 3.1 Guardian Watchdog Subsystem
- **Role**: Top-level supervising daemon. If the Manager process crashes or is killed by the OS (OOM, terminal closure, signal), the Guardian detects the death within `health_interval_seconds` (default: 30s).
- **Process Identity**: Uses composite identity checks (`PID`, `Process Creation Time`, `Executable Path`, `Command Line Token`) to prevent PID-reuse race conditions on Windows.
- **Sleep Inhibition**: On Windows platforms, actively invokes `ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)` to prevent system sleep and standby while campaigns are running.
- **Restart Circuit Breaker**: Tracks manager restarts in `guardian.json` with persisted UTC timestamps. Enforces a maximum restart count per hour (default: 4/hr).

### 3.2 Campaign Manager Subsystem
- **Role**: Coordinates active research workers according to the immutable `plan.json`.
- **Mode Selection**: Supports `--mode {hybrid, classical, quantum}`. In `classical` mode, only classical search workers spawn. In `quantum` mode, only the quantum search worker spawns. In `hybrid` mode (default), both run concurrently.
- **Persistent Deadline**: Enforces duration limits via `duration_hours` and `deadline_utc` recorded inside `plan.json` and `state.json`. Monitored cooperatively inside event loops; when the deadline expires, the manager triggers a clean cooperative shutdown without killing processes.
- **Failure Classification**:
  - *Startup Failure*: Exit code 2 within the first 10 seconds of round 0 with 0 restarts is classified as an invalid configuration/backend error (`needs_attention`), halting the campaign immediately.
  - *In-Flight Failure*: Exit code 2 after round 0 (e.g. transient file lock collision or transient CUDA driver hiccup) is classified as a recoverable failure, triggering an exponential backoff restart under the hourly budget.
- **Startup Deadline Guard**: If a newly spawned worker fails to produce its first heartbeat within 180 seconds, the manager terminates the hanging worker and reschedules it rather than waiting indefinitely.

### 3.3 Storage, Checkpoint & Data Integrity Subsystem
- **Robust File Operations**: Replaces raw `os.replace` with `robust_replace(src, dst)` across all runtime locking, heartbeat writes, state checkpoints, and log rotations. Uses exponential backoff with randomized jitter (up to 12 retries over 3.5 seconds) to survive transient Windows NTFS sharing violations (`WinError 5` Access Denied / `WinError 32` Sharing Violation).
- **Atomic Checkpoints**:
  1. Serializes checkpoint data to a temporary file (`.tmp`).
  2. Flushes buffers and issues OS-level `fsync`.
  3. Executes atomic `robust_replace`.
  4. Generates and validates SHA-256 checksums.
  5. Maintains generational backup checkpoints (`checkpoint.json.1` to `.5`).
- **SQLite Ledger Integrity**:
  - Configured with `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=30000;`.
  - Transaction boundaries wrap discrete discovery trial evaluations.
  - Checkpointed at round boundaries and clean shutdowns.
  - Backup verified using SQLite online backup API with SHA-256 manifest verification.

---

## 4. Lifecycle State Machine

```
              ┌───────────────┐
              │    CREATED    │
              └───────┬───────┘
                      │ amde2 start
                      ▼
              ┌───────────────┐
              │   STARTING    │
              └───────┬───────┘
                      │ workers online
                      ▼
       ┌─────────► RUNNING ◄─────────┐
       │              │              │
       │              ▼              │
       │         DEGRADED            │ (worker backoff cooldown)
       │              │              │
       └──────────────┴──────────────┘
                      │
   ┌──────────────────┼──────────────────┐
   ▼                  ▼                  ▼
STOP_REQUESTED    PAUSED_STORAGE     PAUSED_RESOURCE
   │                  │                  │
   └──────────────────┼──────────────────┘
                      │
                      ▼
                  QUIESCING
                      │ (workers finish in-flight trials)
                      ▼
              WORKERS_STOPPING
                      │ (cooperative STOP marker acknowledged)
                      ▼
            CHECKPOINT_COMMITTING
                      │ (fsync & SHA-256 checksum)
                      ▼
               DATABASE_CLOSING
                      │ (PRAGMA wal_checkpoint(TRUNCATE))
                      ▼
                 VERIFICATION
                      │ (state & checkpoint validation)
                      ▼
                   STOPPED
```

---

## 5. Hardware Profile Compliance

AMDE 2.0.1 is optimized and validated for target hardware:
- **CPU**: Intel Core i7-8850H (6 Physical Cores / 12 Logical Threads). Classical threads budgeted to 4; quantum CPU threads budgeted to 2. Total threads bounded to prevent CPU starvation.
- **RAM**: 32 GB DDR4. Safety floor enforced at 5.0 GB minimum free RAM; worker memory caps enforced via resource sampler.
- **GPU**: NVIDIA Quadro P2000 Mobile (4 GB GDDR5 VRAM, Pascal GP107, Compute Capability 6.1).
  - PyTorch 2.14.0+cu126 / CUDA 12.6.
  - VRAM reserve capped at 768 MB; batch size 16, statevector memory cap 256 MB.
  - Thermal hysteresis: GPU temperature monitored with warning at 80°C and pause at 88°C across 3 consecutive health samples.
- **Storage**: Minimum 5.0 GB free disk space required; run budget bounded by `RunStorageBudget`.
