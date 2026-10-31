# AMDE 2.0 runtime architecture

## Scope

AMDE 2.0 keeps the 1.3.0 research engines and wraps them in a durable process-control layer. The Classical worker remains a CPU AMDE campaign; the Quantum worker remains a statevector/QML experiment using CUDA or CPU. The two research branches remain independent campaigns; the supervisor does not make them a single co-trained model.

## Process topology

```text
Windows Task Scheduler (optional, at user logon)
  └─ Guardian (detached; sleep inhibition; manager watchdog)
       └─ Campaign Manager (immutable plan; state; health; restart policy)
            ├─ Log relay ─ Classical campaign worker (CPU)
            │                 └─ per-round engine workers / SQLite / checkpoints
            └─ Log relay ─ Quantum campaign worker (CUDA or CPU)
                              └─ quantum SQLite ledger / per-trial results
```

The manager and workers are separate processes. The guardian is a parent watchdog and can re-adopt an already-live manager after guardian restart. After a manager crash, it checks worker locks and command-line identity, requests campaign-local cooperative stop for orphan workers, and only then starts a replacement manager. It does not deliberately launch duplicate workers.

## Durable state

- `plan.json` is immutable and hashed with AMDE's canonical JSON hash (`plan_hash`). Resume rejects a modified plan.
- `state.json` is atomically replaced and includes manager heartbeat, worker state, process IDs, restart history, health and plan hash.
- `guardian.json` includes guardian/manager identity, heartbeat and persisted manager-restart epochs.
- Worker locks and heartbeats are checked against process identity, not only numeric PID, to reduce PID-reuse errors after a crash/reboot.
- Classical round checkpoints are atomic and checksummed with bounded internal backups. Trial history is held in SQLite.
- Quantum results are committed in SQLite per model/trial; incomplete trial rows are eligible for retry. The worker checkpoints SQLite WAL at round boundaries and shutdown.
- Backups use SQLite online backup, then a SHA-256 manifest and ZIP CRC check. Restore validates the complete archive before extracting and never starts workers automatically.

## Lifecycle state model

```text
created → starting → running ↔ degraded
                     │
                     ├─ stop request → stopping → stopped
                     ├─ disk cap/floor → stopping → paused_storage
                     ├─ RAM/GPU pressure → stopping → paused_resource
                     └─ unrecoverable config/backend error → stopping → needs_attention

stopped / paused_* → explicit resume → starting
needs_attention / failed → inspect evidence; explicit force-recover or a new campaign
```

A transient worker crash is restarted with exponential backoff. Restart budgets are persisted across manager restarts. After the per-hour limit, the worker enters a longer cooldown but continues bounded retries. A configuration/backend exit code is not retried indefinitely; it fails closed for operator inspection.

A worker with a stale heartbeat receives a cooperative stop marker first. If it does not exit within the configured grace interval, its process tree is terminated and the campaign restarts from the last durable checkpoint/ledger. A stop request starts the same grace clock immediately; a frozen process cannot block shutdown indefinitely.

## Resource policy

The manager samples free disk, available RAM, CPU and (when CUDA is requested) `nvidia-smi` GPU status. It pauses rather than silently changing the requested device when disk/RAM/GPU safety floors are breached. The Quantum worker also waits for its free-VRAM and thermal budget. Worker campaign caps bound generated run files; worker log relays rotate files to bound logs.

## Data and research semantics

A resumed Classical run uses the same immutable run config/checkpoint. A resumed Quantum round skips successfully committed trials and retries incomplete/failed entries. A campaign restart does not compare scores across changed protocols. Campaign plan hashes and per-round protocol hashes remain the authoritative identity of an experiment.

The quantum branch is classical GPU/CPU simulation over statevectors; no physical QPU is invoked. Runtime stability is not evidence of scientific superiority, causal understanding, general intelligence or quantum advantage.

## Failure assumptions

The runtime is designed to recover from ordinary Python process crashes, stale worker heartbeats, transient worker errors, manager crashes, stop/resume and some resource pressure. It cannot recover guaranteed data from a failed disk, filesystem damage, power loss during device-level writes, OS updates/reboots, a user logging off, or hardware overheating/failure. Use independent off-machine backups and qualify the exact target system with multi-day soak tests.
