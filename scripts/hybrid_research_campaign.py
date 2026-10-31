#!/usr/bin/env python3
"""Supervise two concurrent, crash-resumable AMDE research campaigns.

Worker A: the established AMDE classical search engine on CPU.
Worker B: VQC/quantum-kernel/statevector/Grover research on the selected
PyTorch device (normally CUDA). Each worker owns separate ledgers, locks,
checkpoints and reports. The supervisor restarts transient failures with a
bounded circuit breaker and propagates a cooperative stop to both workers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.errors import AMDEBaseError, ConfigError, LockError  # noqa: E402
from amde.runtime.lock import SingleInstanceLock, default_pid_alive  # noqa: E402
from amde.util import atomic_write_json, atomic_write_text, hash_json, read_json, sanitize_id, utc_now  # noqa: E402


@dataclass
class Worker:
    name: str
    campaign_id: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    restarts: int = 0
    consecutive_failures: int = 0
    next_start_at: float = 0.0
    last_exit_code: int | None = None
    status: str = "not_started"


def _rotate_log(path: Path, max_bytes: int = 20 * 1024 * 1024, backups: int = 3) -> None:
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    for index in range(backups, 0, -1):
        src = path.with_suffix(path.suffix + f".{index}")
        dst = path.with_suffix(path.suffix + f".{index + 1}")
        if src.exists():
            if index == backups:
                src.unlink(missing_ok=True)
            else:
                os.replace(src, dst)
    os.replace(path, path.with_suffix(path.suffix + ".1"))


def _worker_state_path(root: Path, worker: Worker) -> Path:
    if worker.name == "classical":
        return root / "reports" / "campaigns" / worker.campaign_id / "state.json"
    return root / "reports" / "quantum_campaigns" / worker.campaign_id / "state.json"


def _worker_lock_path(root: Path, worker: Worker) -> Path:
    if worker.name == "classical":
        return root / "reports" / "campaigns" / worker.campaign_id / "campaign.lock"
    return root / "reports" / "quantum_campaigns" / worker.campaign_id / "campaign.lock"


def _read_worker_state(root: Path, worker: Worker) -> dict[str, Any]:
    try:
        value = read_json(_worker_state_path(root, worker))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _check_orphan_worker(root: Path, worker: Worker) -> None:
    path = _worker_lock_path(root, worker)
    try:
        lock = read_json(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        raise LockError(f"cannot inspect prior {worker.name} worker lock {path}: {exc}") from exc
    pid = lock.get("pid") if isinstance(lock, dict) else None
    if isinstance(pid, int) and default_pid_alive(pid):
        raise LockError(
            f"prior {worker.name} campaign worker is still alive (pid={pid}); "
            "inspect its existing console/process and stop it cooperatively before starting another supervisor"
        )
    # Dead/stale worker locks are handled by the campaign's own lock manager.


def _start_worker(root: Path, worker: Worker) -> None:
    _check_orphan_worker(root, worker)
    _rotate_log(worker.log_path)
    worker.log_handle = worker.log_path.open("a", encoding="utf-8", buffering=1)
    worker.log_handle.write(f"\n[{utc_now()}] supervisor starting {worker.name}: {worker.command}\n")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    worker.process = subprocess.Popen(
        worker.command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=worker.log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )
    worker.status = "running"


def _close_worker(worker: Worker) -> None:
    if worker.log_handle is not None:
        try:
            worker.log_handle.flush()
            worker.log_handle.close()
        except Exception:
            pass
        worker.log_handle = None
    worker.process = None


def _write_supervisor_state(path: Path, plan: dict[str, Any], workers: list[Worker], status: str,
                            *, started_at: str, note: str = "") -> None:
    children = {}
    for worker in workers:
        children[worker.name] = {
            "campaign_id": worker.campaign_id,
            "pid": worker.process.pid if worker.process and worker.process.poll() is None else None,
            "status": worker.status,
            "restarts": worker.restarts,
            "consecutive_failures": worker.consecutive_failures,
            "last_exit_code": worker.last_exit_code,
            "log": str(worker.log_path),
            "campaign_state": str(_worker_state_path(path.parents[3], worker)),
        }
    atomic_write_json(path, {
        "schema_version": "amde-hybrid-supervisor-v1",
        "campaign_id": plan["campaign_id"],
        "plan": plan,
        "plan_hash": hash_json(plan),
        "status": status,
        "note": note,
        "supervisor_pid": os.getpid(),
        "started_at": started_at,
        "heartbeat_at": utc_now(),
        "updated_at": utc_now(),
        "children": children,
    })


def _propagate_stop(root: Path) -> None:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    atomic_write_text(runs / "STOP", f"hybrid supervisor stop requested at {utc_now()}\n")
    atomic_write_text(runs / "QUANTUM_STOP", f"hybrid supervisor stop requested at {utc_now()}\n")


def _clear_stop(root: Path) -> None:
    for name in ("HYBRID_STOP", "STOP", "QUANTUM_STOP"):
        (root / "runs" / name).unlink(missing_ok=True)


def _campaign_terminal_status(root: Path, worker: Worker) -> str:
    state = _read_worker_state(root, worker)
    return str(state.get("status", "unknown"))


def _build_workers(root: Path, args: argparse.Namespace, log_dir: Path) -> list[Worker]:
    py = sys.executable
    classical_id = f"{args.campaign_id}_classical"
    quantum_id = f"{args.campaign_id}_quantum"
    classical_command = [
        py, str(root / "scripts" / "research_campaign.py"),
        "--root", str(root), "--campaign-id", classical_id,
        "--rounds", "0", "--hours-per-round", str(args.hours_per_round),
        "--max-trials-per-round", str(args.max_trials_per_round),
        "--max-campaign-gb", str(args.classical_max_gb),
        "--min-free-gb", str(args.min_free_gb),
        "--seed", str(args.seed), "--device", "cpu",
    ]
    quantum_command = [
        py, str(root / "scripts" / "quantum_research_campaign.py"),
        "--root", str(root), "--campaign-id", quantum_id,
        "--rounds", "0", "--seed", str(args.seed),
        "--device", args.quantum_device,
        "--epochs", str(args.quantum_epochs),
        "--batch-size", str(args.quantum_batch_size),
        "--train-size", str(args.quantum_train_size),
        "--val-size", str(args.quantum_val_size),
        "--test-size", str(args.quantum_test_size),
        "--max-configs", str(args.quantum_max_configs),
        "--max-qubits", str(args.quantum_max_qubits),
        "--max-statevector-mb", str(args.quantum_statevector_mb),
        "--max-campaign-gb", str(args.quantum_max_gb),
        "--min-free-gb", str(args.min_free_gb),
    ]
    return [
        Worker("classical", classical_id, classical_command, log_dir / "classical_worker.log"),
        Worker("quantum", quantum_id, quantum_command, log_dir / "quantum_worker.log"),
    ]


def _shutdown_workers(root: Path, workers: list[Worker], grace_seconds: int) -> None:
    """Request safe stops, wait boundedly, then reap every direct child."""
    active = [worker for worker in workers if worker.process is not None and worker.process.poll() is None]
    if active:
        _propagate_stop(root)
        deadline = time.monotonic() + max(1, int(grace_seconds))
        while time.monotonic() < deadline:
            if not any(worker.process is not None and worker.process.poll() is None for worker in active):
                break
            time.sleep(0.5)
    for worker in workers:
        process = worker.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except OSError:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
        if process is not None:
            worker.last_exit_code = process.poll()
        if worker.status not in {"resource_paused", "circuit_breaker"}:
            worker.status = "stopped"
        _close_worker(worker)


def run_supervisor(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else ROOT
    campaign_id = sanitize_id(args.campaign_id, fallback="amde_hybrid")
    if len(campaign_id) > 32:
        raise ConfigError("hybrid campaign id must be 32 characters or fewer")
    if args.hours_per_round <= 0 or args.max_trials_per_round <= 0:
        raise ConfigError("classical round hours and trials must be positive")
    if args.classical_max_gb <= 0 or args.quantum_max_gb <= 0 or args.min_free_gb < 1:
        raise ConfigError("storage caps must be positive and minimum free disk must be >= 1 GiB")
    if args.max_restarts < 0 or args.quantum_epochs < 1 or args.quantum_max_configs < 3:
        raise ConfigError("restart/quantum sweep settings are invalid")
    if args.quantum_device not in {"cuda", "cpu", "auto"}:
        raise ConfigError("quantum-device must be cuda, cpu or auto")

    plan = {
        "protocol": "amde-hybrid-classical-quantum-supervisor-v1",
        "campaign_id": campaign_id,
        "seed": int(args.seed),
        "hours_per_round": float(args.hours_per_round),
        "max_trials_per_round": int(args.max_trials_per_round),
        "classical_max_gb": float(args.classical_max_gb),
        "quantum_max_gb": float(args.quantum_max_gb),
        "min_free_gb": float(args.min_free_gb),
        "quantum_device": args.quantum_device,
        "quantum_epochs": int(args.quantum_epochs),
        "quantum_batch_size": int(args.quantum_batch_size),
        "quantum_train_size": int(args.quantum_train_size),
        "quantum_val_size": int(args.quantum_val_size),
        "quantum_test_size": int(args.quantum_test_size),
        "quantum_max_configs": int(args.quantum_max_configs),
        "quantum_max_qubits": int(args.quantum_max_qubits),
        "quantum_statevector_mb": int(args.quantum_statevector_mb),
        "max_restarts": int(args.max_restarts),
    }
    plan_hash = hash_json(plan)
    supervisor_dir = root / "reports" / "hybrid_campaigns" / campaign_id
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    state_path = supervisor_dir / "supervisor_state.json"
    log_dir = supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = SingleInstanceLock(supervisor_dir / "supervisor.lock", run_id=campaign_id, stale_seconds=300)
    lock.acquire()
    workers: list[Worker] = []
    started_at = utc_now()
    terminal_status = "stopped"
    terminal_note = "cooperative stop requested"
    normal_shutdown = False
    try:
        workers = _build_workers(root, args, log_dir)
        if state_path.exists():
            old_state = read_json(state_path)
            if old_state.get("plan_hash") != plan_hash:
                raise ConfigError("existing hybrid supervisor has a different immutable plan; use a new campaign id or the original arguments")

        # Check for live child PIDs before clearing any prior stop markers.
        for worker in workers:
            _check_orphan_worker(root, worker)
        if any((root / "runs" / item).exists() for item in ("HYBRID_STOP", "STOP", "QUANTUM_STOP")):
            if not args.clear_stop:
                raise ConfigError("a hybrid/AMDE stop marker exists; inspect prior runs and pass --clear-stop explicitly to resume")
            _clear_stop(root)
        print(f"AMDE hybrid supervisor: {campaign_id}", flush=True)
        print("Classical AMDE campaign: CPU; Quantum research campaign: " + args.quantum_device, flush=True)
        print(f"Supervisor state: {state_path}", flush=True)
        for worker in workers:
            _start_worker(root, worker)
        _write_supervisor_state(state_path, plan, workers, "running", started_at=started_at)

        while True:
            if (root / "runs" / "HYBRID_STOP").exists():
                print("HYBRID_STOP detected; propagating cooperative stop to both workers", flush=True)
                _shutdown_workers(root, workers, args.shutdown_grace_seconds)
                normal_shutdown = True
                _write_supervisor_state(
                    state_path, plan, workers, terminal_status,
                    started_at=started_at, note=terminal_note,
                )
                return 0 if terminal_status == "stopped" else 1

            for worker in workers:
                if worker.process is None:
                    if time.monotonic() >= worker.next_start_at:
                        try:
                            _start_worker(root, worker)
                            print(f"restarted {worker.name} worker pid={worker.process.pid}", flush=True)
                        except LockError as exc:
                            worker.consecutive_failures += 1
                            worker.last_exit_code = 20
                            worker.status = "blocked_by_existing_worker"
                            worker.next_start_at = time.monotonic() + min(300, 5 * 2 ** min(worker.consecutive_failures - 1, 6))
                            print(f"{worker.name} worker not started: {exc}", file=sys.stderr, flush=True)
                            if worker.consecutive_failures > args.max_restarts:
                                _propagate_stop(root)
                                _write_supervisor_state(state_path, plan, workers, "circuit_breaker", started_at=started_at, note=str(exc))
                                return 1
                    continue
                code = worker.process.poll()
                if code is None:
                    continue
                worker.last_exit_code = int(code)
                worker.status = "exited"
                terminal = _campaign_terminal_status(root, worker)
                _close_worker(worker)
                print(f"{worker.name} campaign exited code={code}, campaign_state={terminal}", flush=True)
                if terminal == "stopped":
                    # A worker that completed a requested cooperative stop is
                    # not a crash and must not be restarted by the supervisor.
                    terminal_status = "stopped"
                    terminal_note = f"{worker.name} worker completed a cooperative stop"
                    _propagate_stop(root)
                    atomic_write_text(root / "runs" / "HYBRID_STOP", f"cooperative stop completed by {worker.name}\n")
                    break
                if terminal in {"storage_cap_reached", "low_disk_space", "stop_requested", "paused"}:
                    terminal_status = "resource_paused"
                    terminal_note = f"{worker.name} worker reached {terminal}"
                    _propagate_stop(root)
                    for other in workers:
                        if other.process is not None and other.process.poll() is None:
                            other.status = "stopping_for_resource_limit"
                    _write_supervisor_state(state_path, plan, workers, terminal_status, started_at=started_at,
                                            note=terminal_note)
                    # Let the normal stop branch perform the bounded shutdown.
                    atomic_write_text(root / "runs" / "HYBRID_STOP", f"resource pause from {worker.name}: {terminal}\n")
                    break
                if code == 2 or terminal in {"failed", "partial", "interrupted"}:
                    worker.consecutive_failures += 1
                elif code != 0:
                    worker.consecutive_failures += 1
                else:
                    # A clean exit without a stop or resource condition is
                    # unexpected for an infinite campaign; supervise restart.
                    worker.consecutive_failures += 1
                if worker.consecutive_failures > args.max_restarts:
                    terminal_status = "circuit_breaker"
                    terminal_note = f"{worker.name} exited repeatedly (last code {code}, state {terminal})"
                    _propagate_stop(root)
                    _write_supervisor_state(state_path, plan, workers, terminal_status, started_at=started_at,
                                            note=terminal_note)
                    atomic_write_text(root / "runs" / "HYBRID_STOP", "hybrid restart circuit breaker reached\n")
                    break
                delay = min(300, 5 * (2 ** max(0, worker.consecutive_failures - 1)))
                worker.restarts += 1
                worker.next_start_at = time.monotonic() + delay
                worker.status = f"backoff_{delay}s"
                print(f"{worker.name} worker restart scheduled in {delay}s ({worker.consecutive_failures}/{args.max_restarts})", flush=True)

            if (root / "runs" / "HYBRID_STOP").exists():
                if terminal_status == "circuit_breaker":
                    _shutdown_workers(root, workers, args.shutdown_grace_seconds)
                    normal_shutdown = True
                    _write_supervisor_state(state_path, plan, workers, terminal_status,
                                            started_at=started_at, note=terminal_note)
                    return 1
                continue
            _write_supervisor_state(state_path, plan, workers, "running", started_at=started_at)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Ctrl+C received; requesting cooperative stop", file=sys.stderr, flush=True)
        terminal_status = "stopped"
        terminal_note = "Ctrl+C"
        _shutdown_workers(root, workers, args.shutdown_grace_seconds)
        normal_shutdown = True
        _write_supervisor_state(state_path, plan, workers, terminal_status, started_at=started_at, note=terminal_note)
        return 130
    finally:
        try:
            # Never orphan a CPU/GPU child if the supervisor itself raises unexpectedly.
            if not normal_shutdown and any(w.process is not None and w.process.poll() is None for w in workers):
                try:
                    _shutdown_workers(root, workers, min(args.shutdown_grace_seconds, 30))
                    _write_supervisor_state(
                        state_path, plan, workers, "supervisor_error", started_at=started_at,
                        note="supervisor exited unexpectedly; child workers were stopped and reaped",
                    )
                except Exception:
                    for worker in workers:
                        process = worker.process
                        if process is not None and process.poll() is None:
                            try:
                                process.kill()
                                process.wait(timeout=5)
                            except Exception:
                                pass
                        _close_worker(worker)
            else:
                for worker in workers:
                    if worker.process is not None and worker.process.poll() is not None:
                        _close_worker(worker)
        finally:
            lock.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours-per-round", type=float, default=1.0)
    parser.add_argument("--max-trials-per-round", type=int, default=400)
    parser.add_argument("--classical-max-gb", type=float, default=12.0)
    parser.add_argument("--quantum-max-gb", type=float, default=10.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--quantum-device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--quantum-epochs", type=int, default=12)
    parser.add_argument("--quantum-batch-size", type=int, default=32)
    parser.add_argument("--quantum-train-size", type=int, default=96)
    parser.add_argument("--quantum-val-size", type=int, default=48)
    parser.add_argument("--quantum-test-size", type=int, default=96)
    parser.add_argument("--quantum-max-configs", type=int, default=5)
    parser.add_argument("--quantum-max-qubits", type=int, default=10)
    parser.add_argument("--quantum-statevector-mb", type=int, default=256)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=120)
    parser.add_argument("--clear-stop", action="store_true", help="Explicitly remove hybrid/global/quantum stop markers before starting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_supervisor(args)
    except (AMDEBaseError, OSError, ValueError) as exc:
        print(f"hybrid supervisor error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
