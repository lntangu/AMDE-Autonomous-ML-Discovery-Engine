"""AMDE 2.0 long-running campaign runtime.

The research engines remain separate, immutable-plan workers. This module owns
campaign-scoped lifecycle control, a guardian process, health telemetry,
cooperative stop/resume, bounded crash recovery and evidence snapshots.
It intentionally never deletes experiment data automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .lock import SingleInstanceLock, default_pid_alive
from .job import WindowsJobObject, terminate_process_tree, validate_process_identity
from .resources import sample_process_tree
from ..errors import (
    ConfigError,
    DataIntegrityError,
    EXIT_CONFIG_ERROR,
    EXIT_DATA_INTEGRITY,
    EXIT_RESOURCE_EXHAUSTION,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    FAILURE_CATEGORY_CONFIG,
    FAILURE_CATEGORY_DATA_INTEGRITY,
    FAILURE_CATEGORY_RESOURCE,
    FAILURE_CATEGORY_RUNTIME,
    FAILURE_CATEGORY_UNKNOWN,
)
from ..util import (
    CorruptJSONError,
    hash_json,
    read_json_critical,
    read_json_safe,
    robust_replace,
)

SCHEMA = "amde-2-runtime-v1"
PLAN_SCHEMA = "amde-2-plan-v1"
TERMINAL = {"stopped", "paused_storage", "paused_resource", "needs_attention", "failed", "manager_error"}
WORKER_STOP_STATES = {"stopped", "stop_requested", "interrupted"}
RESOURCE_STATES = {"storage_cap_reached", "low_disk_space", "paused", "resource_paused"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_hash(value: Any) -> str:
    # Match the exact canonical hash implementation used by checkpoints/manifests.
    return hash_json(value)


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        robust_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path, default: Any = None) -> Any:
    return read_json_safe(path, default)


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError, OverflowError):
        return False
    if value <= 0:
        return False
    try:
        import psutil  # type: ignore
        process = psutil.Process(value)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        return process.is_running()
    except ImportError:
        return default_pid_alive(value)
    except psutil.NoSuchProcess:
        return False
    except psutil.ZombieProcess:
        return False
    except psutil.AccessDenied:
        # Never assume an access-denied process is dead and steal its lock.
        return True
    except Exception:
        return default_pid_alive(value)


def _pid_matches(pid: Any, root: Path, campaign_token: str, required_script: str) -> bool:
    """Validate process identity, guarding against PID reuse after reboot/crash."""
    if not _pid_alive(pid):
        return False
    try:
        import psutil  # type: ignore
    except ImportError:
        # Without command-line inspection, stay conservative and never steal a live PID.
        return True
    try:
        process = psutil.Process(int(pid))
        command = " ".join(process.cmdline()).casefold().replace("\\", "/")
        root_token = str(root.resolve()).casefold().replace("\\", "/")
        return root_token in command and campaign_token.casefold() in command and required_script.casefold() in command
    except psutil.NoSuchProcess:
        return False
    except psutil.ZombieProcess:
        # A zombie has exited and cannot be doing work. It is safe to consider
        # it absent for orphan-recovery and duplicate-start decisions.
        return False
    except psutil.AccessDenied:
        return True
    except (OSError, ValueError, TypeError):
        return False


def _archive_reused_pid_lock(lock_path: Path, root: Path, campaign_token: str, required_script: str) -> None:
    data = _read_json(lock_path, {})
    pid = data.get("pid") if isinstance(data, dict) else None
    if _pid_alive(pid) and not _pid_matches(pid, root, campaign_token, required_script):
        archived = lock_path.with_name(f"{lock_path.name}.pid-reused.{int(time.time())}")
        robust_replace(lock_path, archived)


def _path_age_seconds(path: Path, field: str = "heartbeat_at") -> float | None:
    data = _read_json(path, {})
    stamp = data.get(field) if isinstance(data, dict) else None
    if not isinstance(stamp, str):
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except ValueError:
        return None


def _rotate(path: Path, max_bytes: int = 20 * 1024 * 1024, backups: int = 5) -> None:
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    for index in range(backups, 0, -1):
        source = path.with_name(path.name + f".{index}")
        target = path.with_name(path.name + f".{index + 1}")
        if source.exists():
            if index == backups:
                source.unlink(missing_ok=True)
            else:
                robust_replace(source, target)
    robust_replace(path, path.with_name(path.name + ".1"))


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _paths(root: Path, campaign_id: str) -> dict[str, Path]:
    base = root / "reports" / "amde2" / campaign_id
    control = base / "control"
    return {
        "base": base,
        "plan": base / "plan.json",
        "state": base / "state.json",
        "control": control,
        "stop": control / "STOP",
        "classical_stop": control / "classical.STOP",
        "quantum_stop": control / "quantum.STOP",
        "manager_lock": control / "manager.lock",
        "guardian_lock": control / "guardian.lock",
        "guardian": control / "guardian.json",
        "manager_log": base / "logs" / "manager.log",
        "guardian_log": base / "logs" / "guardian.log",
        "events": base / "logs" / "events.jsonl",
    }


def _validate_id(value: str) -> str:
    value = str(value or "").strip()
    if not SAFE_ID.fullmatch(value):
        raise ValueError("campaign ID must be 1-32 ASCII letters/digits/underscore/hyphen and start with a letter or digit")
    reserved = {"STOP", "CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if value.upper() in reserved:
        raise ValueError(f"campaign ID is reserved by Windows: {value}")
    return value


def _make_plan(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    mode = str(getattr(args, "mode", "hybrid") or "hybrid").lower()
    if mode not in {"hybrid", "classical", "quantum"}:
        raise ValueError("mode must be hybrid, classical or quantum")
    dur_arg = getattr(args, "duration_hours", None)
    duration_hours = float(dur_arg) if dur_arg is not None and float(dur_arg) > 0 else None
    campaign_type = "timed" if duration_hours is not None else "infinite"
    deadline_utc = None
    if duration_hours is not None:
        deadline_dt = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        deadline_utc = deadline_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    d_policy_arg = getattr(args, "deadline_policy", "cooperative")
    deadline_policy = str(d_policy_arg or "cooperative").lower()
    if deadline_policy not in {"cooperative", "hard"}:
        raise ValueError(f"deadline_policy must be 'cooperative' or 'hard', got '{deadline_policy}'")

    qprofile = getattr(args, "quantum_profile", None)
    q_epochs = int(args.quantum_epochs)
    q_batch_size = int(args.quantum_batch_size)
    q_train_size = int(args.quantum_train_size)
    q_val_size = int(args.quantum_val_size)
    q_test_size = int(args.quantum_test_size)
    q_max_configs = int(args.quantum_max_configs)
    q_max_qubits = int(args.quantum_max_qubits)
    if qprofile:
        from ..quantum.profiles import PROFILE_CATALOG
        prof_key = str(qprofile).upper()
        if prof_key in PROFILE_CATALOG and prof_key != "BASELINE":
            prof = PROFILE_CATALOG[prof_key]
            q_epochs = prof.epochs
            q_batch_size = prof.batch_size
            q_train_size = prof.train_size
            q_val_size = prof.val_size
            q_test_size = prof.test_size
            q_max_configs = prof.max_configs
            q_max_qubits = prof.n_qubits

    values = {
        "schema": PLAN_SCHEMA,
        "campaign_id": _validate_id(args.campaign_id),
        "root": str(root.resolve()),
        "mode": mode,
        "campaign_type": campaign_type,
        "duration_hours": duration_hours,
        "deadline_utc": deadline_utc,
        "deadline_policy": deadline_policy,
        "seed": int(args.seed),
        "hours_per_round": float(args.hours_per_round),
        "max_trials_per_round": int(args.max_trials_per_round),
        "classical_max_gb": float(args.classical_max_gb),
        "quantum_max_gb": float(args.quantum_max_gb),
        "min_free_gb": float(args.min_free_gb),
        "quantum_device": str(args.quantum_device),
        "quantum_profile": str(qprofile).lower() if qprofile else "baseline",
        "classical_threads": int(args.classical_threads),
        "quantum_cpu_threads": int(args.quantum_cpu_threads),
        "quantum_epochs": q_epochs,
        "quantum_batch_size": q_batch_size,
        "quantum_train_size": q_train_size,
        "quantum_val_size": q_val_size,
        "quantum_test_size": q_test_size,
        "quantum_max_configs": q_max_configs,
        "quantum_max_qubits": q_max_qubits,
        "quantum_statevector_mb": int(args.quantum_statevector_mb),
        "memory_reserve_gb": float(args.memory_reserve_gb),
        "stale_heartbeat_seconds": int(args.stale_heartbeat_seconds),
        "manager_stale_seconds": int(args.manager_stale_seconds),
        "stable_after_seconds": int(args.stable_after_seconds),
        "max_worker_restarts_per_hour": int(args.max_worker_restarts_per_hour),
        "degraded_retry_seconds": int(args.degraded_retry_seconds),
        "shutdown_grace_seconds": int(args.shutdown_grace_seconds),
        "health_interval_seconds": int(args.health_interval_seconds),
        "max_gpu_temperature_c": float(args.max_gpu_temperature_c),
    }
    if values["hours_per_round"] <= 0 or values["max_trials_per_round"] <= 0:
        raise ValueError("hours-per-round and max-trials-per-round must be positive")
    if min(values["classical_max_gb"], values["quantum_max_gb"], values["min_free_gb"]) <= 0:
        raise ValueError("campaign storage limits must be positive")
    if values["quantum_device"] not in {"cuda", "cpu", "auto"}:
        raise ValueError("quantum-device must be cuda, cpu or auto")
    if not 1 <= values["classical_threads"] <= 12 or not 1 <= values["quantum_cpu_threads"] <= 8:
        raise ValueError("classical threads must be 1-12 and quantum CPU threads 1-8")
    if values["quantum_max_configs"] < 3 or values["quantum_max_qubits"] < 8:
        raise ValueError("quantum-max-configs must be >=3 and quantum-max-qubits >=8")
    if values["quantum_epochs"] < 1 or min(values["quantum_train_size"], values["quantum_val_size"], values["quantum_test_size"]) < 8:
        raise ValueError("quantum epochs and split sizes are invalid")
    if values["memory_reserve_gb"] < 0.5 or values["health_interval_seconds"] < 2:
        raise ValueError("memory reserve must be >=0.5GB and health interval >=2s")
    if values["stale_heartbeat_seconds"] < 10 or values["manager_stale_seconds"] < 10 or values["shutdown_grace_seconds"] < 1.0:
        raise ValueError("heartbeat thresholds must be >=10s and shutdown grace >=1.0s")
    if not 60.0 <= values["max_gpu_temperature_c"] <= 100.0:
        raise ValueError("max GPU temperature must be between 60 and 100 Celsius")
    if values["max_worker_restarts_per_hour"] < 1 or values["degraded_retry_seconds"] < 60:
        raise ValueError("restart budget and degraded retry interval are invalid")
    values["plan_hash"] = _json_hash(values)
    return values


def _read_plan(paths: dict[str, Path]) -> dict[str, Any]:
    try:
        plan = read_json_critical(paths["plan"], expected_type=dict)
    except Exception as exc:
        raise ConfigError(f"campaign plan is missing or invalid: {paths['plan']} ({exc})") from exc
    if plan.get("schema") != PLAN_SCHEMA:
        raise ConfigError(f"campaign plan has invalid schema: {paths['plan']}")
    expected = plan.get("plan_hash")
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if expected != _json_hash(payload):
        raise DataIntegrityError(f"campaign plan checksum mismatch in {paths['plan']}; preserve files and inspect before resuming")
    return plan


def _worker_paths(root: Path, campaign_id: str, name: str) -> dict[str, Path]:
    if name == "classical":
        child_id = f"{campaign_id}_classical"
        base = root / "reports" / "campaigns" / child_id
        return {"id": Path(child_id), "state": base / "state.json", "lock": base / "campaign.lock"}
    child_id = f"{campaign_id}_quantum"
    base = root / "reports" / "quantum_campaigns" / child_id
    return {"id": Path(child_id), "state": base / "state.json", "lock": base / "campaign.lock"}


@dataclass
class Worker:
    name: str
    campaign_id: str
    state_path: Path
    lock_path: Path
    stop_path: Path
    log_path: Path
    command: list[str]
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    status: str = "not_started"
    restarts: int = 0
    consecutive_failures: int = 0
    failure_times: list[float] | None = None
    next_start_at: float = 0.0
    started_at_monotonic: float = 0.0
    last_exit_code: int | None = None
    last_error: str | None = None
    stale_stop_requested_at: float | None = None
    stale_warning_written: bool = False

    def __post_init__(self) -> None:
        if self.failure_times is None:
            self.failure_times = []


class AMDE2Manager:
    def __init__(self, root: Path, campaign_id: str, plan: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.campaign_id = _validate_id(campaign_id)
        self.plan = plan
        self.paths = _paths(self.root, self.campaign_id)
        self.workers: list[Worker] = []
        self.lock = SingleInstanceLock(self.paths["manager_lock"], run_id=self.campaign_id, stale_seconds=120)
        self.job = WindowsJobObject(name=f"Local\\AMDE_{self.campaign_id}")
        self.job.create(kill_on_close=False)
        self.state: dict[str, Any] = {}
        self.target_status: str | None = None
        self.target_note = ""
        self.shutdown_started: float | None = None
        self.last_health_at = 0.0
        self.low_memory_samples = 0
        self.high_gpu_temperature_samples = 0
        self._interrupted = False

    def _log(self, message: str, *, level: str = "INFO", **fields: Any) -> None:
        stamp = utc_now()
        line = f"[{stamp}] {level} {message}"
        _rotate(self.paths["manager_log"])
        self.paths["manager_log"].parent.mkdir(parents=True, exist_ok=True)
        with self.paths["manager_log"].open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
        _append_event(self.paths["events"], message, level=level, **fields)
        print(line, flush=True)

    def _build_workers(self) -> list[Worker]:
        py = sys.executable
        script_classical = self.root / "scripts" / "research_campaign.py"
        script_quantum = self.root / "scripts" / "quantum_research_campaign.py"
        script_relay = self.root / "scripts" / "amde2_logrelay.py"
        if not script_classical.is_file() or not script_quantum.is_file() or not script_relay.is_file():
            raise FileNotFoundError("AMDE 2.0 worker scripts/log relay are missing; extract the complete release archive")
        classical_id = f"{self.campaign_id}_classical"
        quantum_id = f"{self.campaign_id}_quantum"
        classical_command = [
            py, "-u", str(script_classical), "--root", str(self.root), "--campaign-id", classical_id,
            "--rounds", "0", "--hours-per-round", str(self.plan["hours_per_round"]),
            "--max-trials-per-round", str(self.plan["max_trials_per_round"]),
            "--max-campaign-gb", str(self.plan["classical_max_gb"]),
            "--min-free-gb", str(self.plan["min_free_gb"]), "--seed", str(self.plan["seed"]),
            "--device", "cpu", "--campaign-stop-file", str(self.paths["classical_stop"]), "--isolated-stop",
        ]
        quantum_command = [
            py, "-u", str(script_quantum), "--root", str(self.root), "--campaign-id", quantum_id,
            "--rounds", "0", "--seed", str(self.plan["seed"]), "--device", str(self.plan["quantum_device"]),
            "--epochs", str(self.plan["quantum_epochs"]), "--batch-size", str(self.plan["quantum_batch_size"]),
            "--train-size", str(self.plan["quantum_train_size"]), "--val-size", str(self.plan["quantum_val_size"]),
            "--test-size", str(self.plan["quantum_test_size"]), "--max-configs", str(self.plan["quantum_max_configs"]),
            "--max-qubits", str(self.plan["quantum_max_qubits"]), "--max-statevector-mb", str(self.plan["quantum_statevector_mb"]),
            "--max-campaign-gb", str(self.plan["quantum_max_gb"]), "--min-free-gb", str(self.plan["min_free_gb"]),
            "--campaign-stop-file", str(self.paths["quantum_stop"]), "--isolated-stop",
        ]
        if self.plan.get("quantum_profile"):
            quantum_command.extend(["--profile", str(self.plan["quantum_profile"])])
        cp = _worker_paths(self.root, self.campaign_id, "classical")
        qp = _worker_paths(self.root, self.campaign_id, "quantum")
        log_dir = self.paths["base"] / "logs"
        mode = str(self.plan.get("mode", "hybrid") or "hybrid").lower()
        workers: list[Worker] = []
        if mode in {"hybrid", "classical"}:
            workers.append(Worker("classical", classical_id, cp["state"], cp["lock"], self.paths["classical_stop"], log_dir / "classical.log", classical_command))
        if mode in {"hybrid", "quantum"}:
            workers.append(Worker("quantum", quantum_id, qp["state"], qp["lock"], self.paths["quantum_stop"], log_dir / "quantum.log", quantum_command))
        if not workers:
            raise ValueError(f"no workers configured for mode: {mode}")
        return workers

    def _check_existing_worker(self, worker: Worker) -> None:
        lock_data = _read_json(worker.lock_path, {})
        pid = lock_data.get("pid") if isinstance(lock_data, dict) else None
        script_token = "research_campaign.py" if worker.name == "classical" else "quantum_research_campaign.py"
        if _pid_matches(pid, self.root, worker.campaign_id, script_token):
            raise RuntimeError(
                f"refusing duplicate {worker.name} worker: PID {pid} matches this campaign and is alive. Stop the existing campaign before resume."
            )
        if _pid_alive(pid):
            # PID was reused by an unrelated process. Archive only this stale AMDE lock.
            archived = worker.lock_path.with_name(f"{worker.lock_path.name}.pid-reused.{int(time.time())}")
            robust_replace(worker.lock_path, archived)

    def _start_worker(self, worker: Worker) -> None:
        self._check_existing_worker(worker)
        _rotate(worker.log_path)
        worker.log_path.parent.mkdir(parents=True, exist_ok=True)
        relay_script = self.root / "scripts" / "amde2_logrelay.py"
        if not relay_script.is_file():
            raise FileNotFoundError(f"worker log relay is missing: {relay_script}")
        relay_command = [sys.executable, "-u", str(relay_script), "--log-path", str(worker.log_path), "--", *worker.command]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        thread_limit = int(self.plan["classical_threads"] if worker.name == "classical" else self.plan["quantum_cpu_threads"])
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[variable] = str(thread_limit)
        try:
            worker.process = subprocess.Popen(
                relay_command,
                cwd=self.root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=(os.name != "nt"),
                creationflags=creationflags,
                start_new_session=(os.name != "nt"),
            )
        except BaseException:
            self._close_worker_log(worker)
            raise
        worker.status = "running"
        if self.job and self.job.is_active() and worker.process:
            self.job.assign_process(worker.process.pid)
        worker.started_at_monotonic = time.monotonic()
        worker.next_start_at = 0.0
        self._log(f"worker started: {worker.name}", pid=worker.process.pid, campaign_id=worker.campaign_id)

    @staticmethod
    def _close_worker_log(worker: Worker) -> None:
        if worker.log_handle is not None:
            try:
                worker.log_handle.flush()
                worker.log_handle.close()
            except Exception:
                pass
        worker.log_handle = None

    def _worker_state(self, worker: Worker) -> dict[str, Any]:
        data = _read_json(worker.state_path, {})
        return data if isinstance(data, dict) else {}

    def _worker_heartbeat(self, worker: Worker, state: dict[str, Any] | None = None) -> str | None:
        state = state or self._worker_state(worker)
        if worker.name == "classical":
            for record in reversed(state.get("rounds", []) if isinstance(state.get("rounds"), list) else []):
                if record.get("status") == "running" and record.get("run_id"):
                    heartbeat_path = self.root / "runs" / str(record["run_id"]) / "heartbeat.json"
                    hb = _read_json(heartbeat_path, {})
                    if isinstance(hb, dict) and isinstance(hb.get("updated_at"), str):
                        return hb["updated_at"]
        stamp = state.get("heartbeat_at", state.get("updated_at"))
        return stamp if isinstance(stamp, str) else None

    def _worker_status_payload(self, worker: Worker) -> dict[str, Any]:
        state = self._worker_state(worker)
        alive = worker.process is not None and worker.process.poll() is None
        lock_data = _read_json(worker.lock_path, {})
        actual_worker_pid = lock_data.get("pid") if _pid_alive(lock_data.get("pid")) else None
        target_pid = actual_worker_pid if actual_worker_pid is not None else (worker.process.pid if alive and worker.process else None)
        telemetry = sample_process_tree(target_pid) if target_pid else None
        return {
            "campaign_id": worker.campaign_id,
            "pid": target_pid,
            "relay_pid": worker.process.pid if alive and worker.process else None,
            "alive": bool(alive),
            "status": worker.status,
            "campaign_status": state.get("status", "not_created"),
            "heartbeat_at": self._worker_heartbeat(worker, state),
            "current_round": state.get("current_round", state.get("next_round")),
            "next_round": state.get("next_round"),
            "last_round_status": state.get("last_round_status"),
            "restarts": worker.restarts,
            "consecutive_failures": worker.consecutive_failures,
            "last_exit_code": worker.last_exit_code,
            "last_error": worker.last_error,
            "failure_times_epoch": list(worker.failure_times or []),
            "telemetry": telemetry,
            "state_path": str(worker.state_path),
            "log_path": str(worker.log_path),
        }

    def _health_snapshot(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.root)
        health: dict[str, Any] = {
            "at": utc_now(),
            "disk_total_gb": round(disk.total / 1024**3, 3),
            "disk_free_gb": round(disk.free / 1024**3, 3),
            "memory_available_gb": None,
            "memory_total_gb": None,
            "cpu_percent": None,
            "gpu": None,
        }
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            health["memory_available_gb"] = round(vm.available / 1024**3, 3)
            health["memory_total_gb"] = round(vm.total / 1024**3, 3)
            health["cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        if self.plan["quantum_device"] == "cuda":
            try:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                cmd = ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False, creationflags=flags)
                if result.returncode == 0 and result.stdout.strip():
                    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
                    if len(parts) >= 5:
                        def number_or_none(value: str) -> float | None:
                            try:
                                return float(value)
                            except (TypeError, ValueError):
                                return None
                        health["gpu"] = {"name": parts[0], "temperature_c": number_or_none(parts[1]),
                                         "memory_total_mb": number_or_none(parts[2]),
                                         "memory_free_mb": number_or_none(parts[3]),
                                         "utilization_percent": number_or_none(parts[4])}
            except Exception as exc:
                health["gpu"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        return health

    def _write_state(self, status: str | None = None, note: str | None = None) -> None:
        previous = self.state
        if status is not None:
            previous["status"] = status
        if note is not None:
            previous["note"] = note
        guardian = _read_json(self.paths["guardian"], {})
        previous.update({
            "schema_version": SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_hash": self.plan["plan_hash"],
            "manager_pid": os.getpid(),
            "guardian_pid": guardian.get("pid") if isinstance(guardian, dict) else None,
            "heartbeat_at": utc_now(),
            "updated_at": utc_now(),
            "workers": {worker.name: self._worker_status_payload(worker) for worker in self.workers},
            "health": previous.get("health"),
        })
        _atomic_json(self.paths["state"], previous)

    def _request_worker_stops(self, reason: str) -> None:
        for worker in self.workers:
            worker.stop_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(worker.stop_path, {"requested_at": utc_now(), "reason": reason, "campaign_id": self.campaign_id})
            if worker.process is not None and worker.process.poll() is None:
                worker.status = "stopping"
                if worker.stale_stop_requested_at is None:
                    # Start the cooperative grace clock for any intentional or
                    # safety shutdown. A frozen worker cannot hold stop forever.
                    worker.stale_stop_requested_at = time.monotonic()
        self._log("campaign-scoped stop propagated to workers", reason=reason)

    def _begin_shutdown(self, target_status: str, note: str) -> None:
        if self.target_status is not None:
            # needs_attention and storage/resource pause take precedence over an ordinary user stop.
            priority = {"stopped": 1, "paused_resource": 2, "paused_storage": 3, "needs_attention": 4, "failed": 5}
            if priority.get(target_status, 0) <= priority.get(self.target_status, 0):
                return
        self.target_status = target_status
        self.target_note = note
        self.shutdown_started = self.shutdown_started or time.monotonic()
        self._request_worker_stops(note)
        self._write_state("stopping", note)

    def _resource_check(self) -> None:
        now = time.monotonic()
        if now - self.last_health_at < self.plan["health_interval_seconds"]:
            return
        self.last_health_at = now
        health = self._health_snapshot()
        self.state["health"] = health
        disk_free = float(health["disk_free_gb"])
        if disk_free < float(self.plan["min_free_gb"]):
            self._begin_shutdown("paused_storage", f"free disk {disk_free:.2f} GiB below reserve {self.plan['min_free_gb']:.2f} GiB")
            return
        available = health.get("memory_available_gb")
        if isinstance(available, (int, float)) and available < float(self.plan["memory_reserve_gb"]):
            self.low_memory_samples += 1
        else:
            self.low_memory_samples = 0
        if self.low_memory_samples >= 3:
            self._begin_shutdown("paused_resource", f"available RAM remained below {self.plan['memory_reserve_gb']:.2f} GiB for three health samples")
            return
        gpu = health.get("gpu")
        gpu_temp = gpu.get("temperature_c") if isinstance(gpu, dict) else None
        if isinstance(gpu_temp, (int, float)) and gpu_temp >= float(self.plan["max_gpu_temperature_c"]):
            self.high_gpu_temperature_samples += 1
        else:
            self.high_gpu_temperature_samples = 0
        if self.high_gpu_temperature_samples >= 3:
            self._begin_shutdown("paused_resource", f"GPU temperature remained >= {self.plan['max_gpu_temperature_c']:.1f} C for three health samples")
            return
        _atomic_json(self.paths["base"] / "health.json", health)

    def _schedule_restart(self, worker: Worker, code: int, state_status: str) -> None:
        # Persist wall-clock epochs so the one-hour restart budget survives a
        # manager/guardian restart. Monotonic timestamps cannot be persisted
        # across process or OS restarts.
        now = time.time()
        worker.failure_times = [t for t in (worker.failure_times or []) if now - t <= 3600.0]
        worker.failure_times.append(now)
        worker.consecutive_failures += 1
        worker.restarts += 1
        failures_last_hour = len(worker.failure_times)
        if failures_last_hour >= int(self.plan["max_worker_restarts_per_hour"]):
            delay = int(self.plan["degraded_retry_seconds"])
            worker.status = f"degraded_retry_{delay}s"
            self._log("worker entered degraded retry cooldown", level="WARNING", worker=worker.name, failures_last_hour=failures_last_hour, retry_in_seconds=delay, exit_code=code, campaign_status=state_status)
        else:
            delay = min(900, 10 * (2 ** min(worker.consecutive_failures - 1, 6)))
            worker.status = f"backoff_{delay}s"
            self._log("worker restart scheduled", level="WARNING", worker=worker.name, failures_last_hour=failures_last_hour, retry_in_seconds=delay, exit_code=code, campaign_status=state_status)
        worker.next_start_at = time.monotonic() + delay

    def _terminate_unresponsive_worker(self, worker: Worker) -> None:
        process = worker.process
        if process is None or process.poll() is not None:
            return
        pid = process.pid
        grace_val = float(self.plan.get("shutdown_grace_seconds", 900))
        self._log(
            "worker did not exit after its cooperative stop window; executing verified forced termination",
            level="WARNING",
            worker=worker.name,
            pid=pid,
        )
        _append_event(
            self.paths["events"],
            "forced_shutdown_timeout_reached",
            worker=worker.name,
            pid=pid,
            grace_seconds=grace_val,
        )
        _append_event(
            self.paths["events"],
            "forced_termination_requested",
            worker=worker.name,
            pid=pid,
        )
        term_res = terminate_process_tree(
            pid,
            exit_code=137,
            job=self.job,
            expected_tokens=["python", worker.name, "research_campaign"],
            timeout=5.0,
        )
        _append_event(
            self.paths["events"],
            "forced_termination_completed",
            worker=worker.name,
            pid=pid,
            **term_res,
        )
        _append_event(
            self.paths["events"],
            "process_tree_verified_exited",
            worker=worker.name,
            verified=term_res.get("success", False),
            pids_terminated=term_res.get("pids_terminated", []),
        )
        worker.last_error = f"unresponsive worker terminated after forced shutdown timeout ({term_res.get('mechanism')})"
        worker.stop_path.unlink(missing_ok=True)

    def _monitor_worker_heartbeats(self) -> None:
        now = time.monotonic()
        stale_limit = float(self.plan.get("stale_heartbeat_seconds", 900))
        grace = float(self.plan.get("shutdown_grace_seconds", 900))
        for worker in self.workers:
            if worker.process is None or worker.process.poll() is not None:
                continue
            if worker.stale_stop_requested_at is not None:
                if now - worker.stale_stop_requested_at >= grace:
                    self._terminate_unresponsive_worker(worker)
                continue
            state = self._worker_state(worker)
            heartbeat = self._worker_heartbeat(worker, state)
            if not isinstance(heartbeat, str):
                startup_limit = min(float(stale_limit), 180.0)
                if now - worker.started_at_monotonic > startup_limit:
                    worker.stale_stop_requested_at = now
                    worker.status = "startup_timeout_stopping"
                    _atomic_json(worker.stop_path, {
                        "requested_at": utc_now(),
                        "reason": f"worker exceeded startup deadline of {startup_limit:.0f}s without first heartbeat",
                        "campaign_id": self.campaign_id,
                    })
                    self._log("worker exceeded startup deadline without first heartbeat; requested cooperative stop",
                              level="WARNING", worker=worker.name, pid=worker.process.pid)
                continue  # Worker is still initializing within deadline.
            try:
                stamp = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                age = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
            except ValueError:
                continue
            if age > stale_limit:
                worker.stale_stop_requested_at = now
                worker.status = "stale_heartbeat_stopping"
                _atomic_json(worker.stop_path, {"requested_at": utc_now(), "reason": f"heartbeat stale for {age:.0f}s", "campaign_id": self.campaign_id})
                self._log("worker heartbeat is stale; requested cooperative stop before checkpoint recovery",
                          level="WARNING", worker=worker.name, heartbeat_age_seconds=round(age, 1), pid=worker.process.pid)

    def _handle_worker_exit(self, worker: Worker) -> None:
        process = worker.process
        if process is None:
            return
        code = process.poll()
        if code is None:
            return
        worker.last_exit_code = int(code)
        duration = max(0.0, time.monotonic() - worker.started_at_monotonic)
        state = self._worker_state(worker)
        campaign_status = str(state.get("status", "unknown"))
        worker.process = None
        self._close_worker_log(worker)
        self._log("worker process exited", worker=worker.name, exit_code=code, campaign_status=campaign_status, uptime_seconds=round(duration, 1))
        if duration >= int(self.plan["stable_after_seconds"]):
            worker.consecutive_failures = 0
            worker.failure_times = []
        stale_recovery = worker.stale_stop_requested_at is not None
        if stale_recovery:
            worker.stale_stop_requested_at = None
            worker.stop_path.unlink(missing_ok=True)
        if self.target_status is not None or (worker.stop_path.exists() and not stale_recovery) or self.paths["stop"].exists():
            worker.status = "stopped" if campaign_status in WORKER_STOP_STATES or code == 0 else "stopped_after_stop_request"
            return
        if stale_recovery:
            self._schedule_restart(worker, int(code), f"stale_heartbeat_recovery:{campaign_status}")
            return

        failure_category = str(state.get("failure_category", "")).lower()

        # 1. Critical Data Integrity Failure: Fatal. NEVER restart silently.
        if failure_category == FAILURE_CATEGORY_DATA_INTEGRITY or code == EXIT_DATA_INTEGRITY:
            detail = state.get("error") or "corrupted ledger, checkpoint, or schema violation"
            worker.last_error = f"FATAL data integrity failure in {worker.name} worker (exit code {code}): {detail}"
            self._log(worker.last_error, level="ERROR", worker=worker.name)
            self._begin_shutdown("needs_attention", worker.last_error)
            worker.status = "needs_attention"
            return

        # 2. Configuration Error: Fatal. Do NOT retry bad configuration.
        if failure_category == FAILURE_CATEGORY_CONFIG or code == EXIT_CONFIG_ERROR:
            worker.last_error = f"{worker.name} worker reported configuration error (exit code {code}; state={campaign_status})"
            self._log(worker.last_error, level="ERROR", worker=worker.name)
            self._begin_shutdown("needs_attention", worker.last_error)
            worker.status = "needs_attention"
            return

        # 3. Resource Exhaustion / Resource Pause:
        if (
            failure_category == FAILURE_CATEGORY_RESOURCE
            or code == EXIT_RESOURCE_EXHAUSTION
            or campaign_status in RESOURCE_STATES
        ):
            target = "paused_storage" if campaign_status in {"storage_cap_reached", "low_disk_space"} else "paused_resource"
            self._begin_shutdown(target, f"{worker.name} worker reported {campaign_status or 'resource_exhaustion'}")
            worker.status = "resource_paused"
            return

        # 4. Clean exit without a stop marker in an infinite campaign:
        if code == EXIT_SUCCESS and campaign_status in WORKER_STOP_STATES:
            self._schedule_restart(worker, code, campaign_status)
            return

        # 5. Standard transient runtime error or unexpected signal (exit code 1 or other):
        self._schedule_restart(worker, int(code), campaign_status)

    def run(self) -> int:
        self.paths["base"].mkdir(parents=True, exist_ok=True)
        self.paths["control"].mkdir(parents=True, exist_ok=True)
        self.paths["manager_log"].parent.mkdir(parents=True, exist_ok=True)
        _archive_reused_pid_lock(self.paths["manager_lock"], self.root, self.campaign_id, "amde2.py")
        self.lock.acquire()
        if self.job and self.job.is_active():
            self.job.assign_process(os.getpid())

        # Fail closed if state.json is physically present on disk but corrupted
        if self.paths["state"].is_file():
            try:
                old_state = read_json_critical(self.paths["state"])
            except Exception as exc:
                self.lock.release()
                raise DataIntegrityError(f"CRITICAL: state.json is corrupted on disk for campaign {self.campaign_id}: {exc}") from exc
        else:
            old_state = {}

        if isinstance(old_state, dict) and old_state.get("plan_hash") not in (None, self.plan["plan_hash"]):
            self.lock.release()
            raise ValueError("existing AMDE 2.0 state has a different immutable plan")
        self.state = old_state if isinstance(old_state, dict) else {}
        prior_manager_pid = self.state.get("manager_pid")
        prior_status = self.state.get("status")
        self.state["manager_recoveries"] = int(self.state.get("manager_recoveries", 0) or 0)
        if prior_status in {"running", "degraded", "manager_error"} and prior_manager_pid and not _pid_matches(
            prior_manager_pid, self.root, self.campaign_id, "amde2.py"
        ):
            self.state["manager_recoveries"] += 1
        self.workers = self._build_workers()
        saved_workers = old_state.get("workers", {}) if isinstance(old_state, dict) else {}
        now_epoch = time.time()
        for worker in self.workers:
            saved = saved_workers.get(worker.name, {}) if isinstance(saved_workers, dict) else {}
            if isinstance(saved, dict):
                worker.restarts = int(saved.get("restarts", 0) or 0)
                worker.consecutive_failures = int(saved.get("consecutive_failures", 0) or 0)
                persisted = saved.get("failure_times_epoch", [])
                worker.failure_times = [float(value) for value in persisted
                                         if isinstance(value, (int, float)) and 0 <= now_epoch - float(value) <= 3600.0]
                # Check if worker was in backoff cooldown or exceeded hourly restart budget across restart/reboot
                if worker.consecutive_failures > 0 and worker.failure_times:
                    last_failure = worker.failure_times[-1]
                    failures_last_hour = len(worker.failure_times)
                    if failures_last_hour >= int(self.plan["max_worker_restarts_per_hour"]):
                        delay = int(self.plan["degraded_retry_seconds"])
                    else:
                        delay = min(900, 10 * (2 ** min(worker.consecutive_failures - 1, 6)))
                    elapsed_since_failure = max(0.0, now_epoch - last_failure)
                    remaining_delay = max(0.0, delay - elapsed_since_failure)
                    if remaining_delay > 0:
                        worker.next_start_at = time.monotonic() + remaining_delay
                        worker.status = (
                            f"degraded_retry_{int(remaining_delay)}s"
                            if failures_last_hour >= int(self.plan["max_worker_restarts_per_hour"])
                            else f"backoff_{int(remaining_delay)}s"
                        )
        self._log("AMDE 2.0 manager starting", campaign_id=self.campaign_id, plan_hash=self.plan["plan_hash"])
        try:
            if self.paths["stop"].exists():
                self._write_state("stopped", "stop marker already present; explicit resume required")
                return 0
            for worker in self.workers:
                self._check_existing_worker(worker)
                if worker.stop_path.exists():
                    raise RuntimeError(f"stale worker stop marker exists: {worker.stop_path}; use the resume command")
            self.state.setdefault("created_at", utc_now())
            self.state["started_at"] = utc_now()
            self._write_state("starting", "workers are being launched")
            for worker in self.workers:
                if worker.next_start_at > time.monotonic():
                    self._log(
                        "worker starting in backoff cooldown",
                        worker=worker.name,
                        status=worker.status,
                        remaining_seconds=round(worker.next_start_at - time.monotonic(), 1),
                    )
                else:
                    self._start_worker(worker)
            self._write_state("running", f"workers launched ({len(self.workers)})")
            while True:
                self.lock.heartbeat()
                if self.paths["stop"].exists() and self.target_status is None:
                    self._begin_shutdown("stopped", "user requested cooperative stop")
                if self._interrupted and self.target_status is None:
                    self._begin_shutdown("stopped", "manager received termination signal")
                if self.target_status is None and self.plan.get("deadline_utc"):
                    try:
                        deadline_dt = datetime.fromisoformat(str(self.plan["deadline_utc"]).replace("Z", "+00:00"))
                        if datetime.now(timezone.utc) >= deadline_dt:
                            policy = str(self.plan.get("deadline_policy", "cooperative")).lower()
                            if policy == "hard":
                                self._log(
                                    f"Manager detected persistent hard deadline reached ({self.plan.get('duration_hours')}h); terminating worker tree",
                                    level="CRITICAL",
                                )
                                for worker in self.workers:
                                    if worker.process is not None and worker.process.poll() is None:
                                        self._terminate_unresponsive_worker(worker)
                                self._begin_shutdown(
                                    "stopped",
                                    f"persistent hard deadline reached ({self.plan.get('duration_hours')}h)",
                                )
                            else:
                                self._begin_shutdown(
                                    "stopped",
                                    f"persistent campaign deadline reached ({self.plan.get('duration_hours')}h)",
                                )
                    except Exception:
                        pass
                if self.target_status is None:
                    self._resource_check()
                self._monitor_worker_heartbeats()
                for worker in self.workers:
                    if worker.process is not None:
                        self._handle_worker_exit(worker)
                    elif self.target_status is None and time.monotonic() >= worker.next_start_at:
                        try:
                            self._start_worker(worker)
                        except Exception as exc:
                            worker.last_error = f"{type(exc).__name__}: {exc}"
                            worker.status = "blocked"
                            self._schedule_restart(worker, 20, "worker_lock_or_launch_error")
                all_exited = all(worker.process is None for worker in self.workers)
                if self.target_status is not None and all_exited:
                    final = self.target_status
                    self.state["ended_at"] = utc_now()
                    self._write_state(final, self.target_note)
                    self._log("AMDE 2.0 manager reached terminal state", status=final)
                    return 0 if final in {"stopped", "paused_storage", "paused_resource"} else 2
                current_status = "degraded" if any(w.status.startswith("degraded_retry") for w in self.workers) else "running"
                self._write_state(current_status, "; ".join(f"{w.name}={w.status}" for w in self.workers))
                sleep_dur = 2.0
                if self.plan.get("deadline_utc"):
                    try:
                        d_dt = datetime.fromisoformat(str(self.plan["deadline_utc"]).replace("Z", "+00:00"))
                        rem = (d_dt - datetime.now(timezone.utc)).total_seconds()
                        if rem > 0 and rem < 2.5:
                            sleep_dur = max(0.05, min(0.2, rem))
                    except Exception:
                        pass
                time.sleep(sleep_dur)
        except KeyboardInterrupt:
            self._begin_shutdown("stopped", "manager interrupted from console")
            while any(w.process is not None for w in self.workers):
                self._monitor_worker_heartbeats()
                for worker in self.workers:
                    if worker.process is not None:
                        self._handle_worker_exit(worker)
                self._write_state("stopping", self.target_note)
                time.sleep(1.0)
            self._write_state("stopped", self.target_note)
            return 0
        except BaseException as exc:
            self._log(f"manager error: {type(exc).__name__}: {exc}", level="ERROR")
            self.state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
            # If startup or manager logic fails after one worker launched, do
            # not abandon that child. Ask every started worker to checkpoint
            # and exit, then publish manager_error for an explicit operator fix.
            try:
                self._begin_shutdown("needs_attention", f"manager error: {type(exc).__name__}: {exc}")
                deadline = time.monotonic() + int(self.plan["shutdown_grace_seconds"])
                while any(worker.process is not None for worker in self.workers):
                    for worker in self.workers:
                        if worker.process is not None:
                            self._handle_worker_exit(worker)
                            if worker.process is not None and time.monotonic() >= deadline:
                                self._terminate_unresponsive_worker(worker)
                    self._write_state("stopping", self.target_note)
                    time.sleep(0.5)
                self._write_state("manager_error", str(exc))
            except Exception as shutdown_exc:
                self._log(f"manager error cleanup also failed: {type(shutdown_exc).__name__}: {shutdown_exc}", level="ERROR")
                self._write_state("manager_error", f"{exc}; cleanup error: {shutdown_exc}")
            raise
        finally:
            self.lock.release()


def _set_windows_system_awake(enabled: bool) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enabled else ES_CONTINUOUS
        previous = kernel32.SetThreadExecutionState(flags)
        return previous != 0
    except Exception:
        return False


class Guardian:
    """Parent watchdog that restarts the manager if the manager process crashes."""
    def __init__(self, root: Path, campaign_id: str) -> None:
        self.root = root.resolve()
        self.campaign_id = _validate_id(campaign_id)
        self.paths = _paths(self.root, self.campaign_id)
        self.lock = SingleInstanceLock(self.paths["guardian_lock"], run_id=self.campaign_id, stale_seconds=120)
        self.job = WindowsJobObject(name=f"Local\\AMDE_{self.campaign_id}")
        self.job.create(kill_on_close=False)
        self.child: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None
        self.restart_times: list[float] = []
        self.interrupted = False
        self.adopted_manager_pid: int | None = None
        self.manager_stale_since: float | None = None
        self.manager_stale_stop_requested = False

    def _write(self, status: str, **extra: Any) -> None:
        old = _read_json(self.paths["guardian"], {})
        data = old if isinstance(old, dict) else {}
        data.update({"schema_version": "amde2-guardian-v1", "campaign_id": self.campaign_id, "pid": os.getpid(),
                     "manager_pid": self.child.pid if self.child and self.child.poll() is None else None,
                     "status": status, "heartbeat_at": utc_now(), "updated_at": utc_now(),
                     "manager_restart_times_epoch": list(self.restart_times), **extra})
        _atomic_json(self.paths["guardian"], data)

    def _log(self, message: str, **fields: Any) -> None:
        _rotate(self.paths["guardian_log"])
        self.paths["guardian_log"].parent.mkdir(parents=True, exist_ok=True)
        suffix = " " + json.dumps(fields, ensure_ascii=False, sort_keys=True, allow_nan=False) if fields else ""
        with self.paths["guardian_log"].open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"[{utc_now()}] {message}{suffix}\n")
            handle.flush()
        print(message + suffix, flush=True)

    def _launch_manager(self) -> None:
        script = self.root / "scripts" / "amde2.py"
        cmd = [sys.executable, "-u", str(script), "run", "--root", str(self.root), "--campaign-id", self.campaign_id]
        _rotate(self.paths["manager_log"])
        self.paths["manager_log"].parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.paths["manager_log"].open("a", encoding="utf-8", buffering=1)
        self.log_handle.write(f"\n[{utc_now()}] Guardian launched manager: {cmd}\n")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.child = subprocess.Popen(
            cmd, cwd=self.root, stdin=subprocess.DEVNULL, stdout=self.log_handle, stderr=subprocess.STDOUT,
            creationflags=flags, start_new_session=(os.name != "nt"), close_fds=(os.name != "nt"),
        )
        if self.job and self.job.is_active() and self.child:
            self.job.assign_process(self.child.pid)
        self._write("monitoring", manager_started_at=utc_now(), manager_restart_count=len(self.restart_times))
        self._log(f"Guardian started manager PID={self.child.pid}")

    def _close_log(self) -> None:
        if self.log_handle is not None:
            try:
                self.log_handle.flush()
                self.log_handle.close()
            except Exception:
                pass
        self.log_handle = None

    def _manager_heartbeat_age(self) -> float | None:
        state = _read_json(self.paths["state"], {})
        if not isinstance(state, dict):
            return None
        stamp = state.get("heartbeat_at")
        if not isinstance(stamp, str):
            return None
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
        except ValueError:
            return None

    def _request_manager_stale_shutdown(self, manager_pid: int, age: float) -> None:
        if self.manager_stale_stop_requested:
            return
        self.manager_stale_stop_requested = True
        self.manager_stale_since = time.monotonic()
        # A stale-manager recovery is NOT a user stop. Do not create the
        # campaign-wide STOP marker: Guardian would interpret that as a
        # permanent operator stop rather than restart the manager. Only stop
        # this campaign's workers, then rebuild the manager after confirming
        # their processes have checkpointed and exited.
        _atomic_json(self.paths["classical_stop"], {"requested_at": utc_now(),
                    "reason": "guardian manager heartbeat watchdog", "campaign_id": self.campaign_id})
        _atomic_json(self.paths["quantum_stop"], {"requested_at": utc_now(),
                    "reason": "guardian manager heartbeat watchdog", "campaign_id": self.campaign_id})
        self._log(f"Manager heartbeat stale ({age:.0f}s, PID={manager_pid}); requested campaign-scoped cooperative shutdown")

    def _terminate_stale_manager(self, manager_pid: int) -> bool:
        """Terminate only the manager process, leaving workers to checkpoint cooperatively."""
        try:
            if self.child is not None and self.child.pid == manager_pid and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=5)
                return True
            import psutil  # type: ignore
            proc = psutil.Process(manager_pid)
            cmdline = " ".join(proc.cmdline()).casefold()
            if str(self.root).casefold() not in cmdline or self.campaign_id.casefold() not in cmdline or "amde2.py" not in cmdline:
                self._log(f"Refusing to terminate stale manager PID={manager_pid}: process identity did not match this campaign")
                return False
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                proc.kill()
            return True
        except Exception as exc:
            self._log(f"Could not safely terminate stale manager PID={manager_pid}: {type(exc).__name__}: {exc}")
            return False

    def _monitor_manager_heartbeat(self, manager_pid: int, plan: dict[str, Any]) -> bool:
        age = self._manager_heartbeat_age()
        stale_limit = int(plan["manager_stale_seconds"])
        if age is None or age <= stale_limit:
            if not self.manager_stale_stop_requested:
                self.manager_stale_since = None
            return True
        self._request_manager_stale_shutdown(manager_pid, age)
        if self.manager_stale_since is None:
            self.manager_stale_since = time.monotonic()
        if time.monotonic() - self.manager_stale_since < int(plan["shutdown_grace_seconds"]):
            return True
        self._log("Manager did not honor watchdog stop within grace period; terminating manager only and recovering workers separately")
        return self._terminate_stale_manager(manager_pid)

    def _recover_orphan_workers(self, grace_seconds: int) -> bool:
        """Stop all campaign-owned worker/relay processes before manager recovery."""
        state = _read_json(self.paths["state"], {})
        workers = state.get("workers", {}) if isinstance(state, dict) else {}
        live: list[tuple[str, int, str, str, Path]] = []
        for name in ("classical", "quantum"):
            item = workers.get(name, {}) if isinstance(workers, dict) else {}
            wp = _worker_paths(self.root, self.campaign_id, name)
            child_id = str(wp["id"])
            script_token = "research_campaign.py" if name == "classical" else "quantum_research_campaign.py"
            stop_path = self.paths["classical_stop"] if name == "classical" else self.paths["quantum_stop"]
            lock_data = _read_json(wp["lock"], {})
            worker_pid = lock_data.get("pid") if isinstance(lock_data, dict) else None
            relay_pid = item.get("pid") if isinstance(item, dict) else None
            found = False
            if _pid_matches(worker_pid, self.root, child_id, script_token):
                live.append((name, int(worker_pid), child_id, script_token, stop_path))
                found = True
            if _pid_matches(relay_pid, self.root, child_id, "amde2_logrelay.py"):
                live.append((name, int(relay_pid), child_id, "amde2_logrelay.py", stop_path))
                found = True
            if found:
                _atomic_json(stop_path, {"requested_at": utc_now(), "reason": "guardian recovering orphan after manager exit", "campaign_id": self.campaign_id})

        if live:
            self._log("Guardian found orphan workers/relays; requested campaign-scoped shutdown before recovery",
                      orphan_pids=[pid for _, pid, _, _, _ in live])
            deadline = time.monotonic() + max(1, int(grace_seconds))
            while time.monotonic() < deadline:
                remaining = [(name, pid, child_id, script, marker)
                             for name, pid, child_id, script, marker in live
                             if _pid_matches(pid, self.root, child_id, script)]
                if not remaining:
                    break
                self._write("orphan_recovery_wait", orphan_pids=[pid for _, pid, _, _, _ in remaining])
                time.sleep(1.0)
            remaining = [(name, pid, child_id, script, marker)
                         for name, pid, child_id, script, marker in live
                         if _pid_matches(pid, self.root, child_id, script)]
            if remaining:
                self._log(f"Orphan worker processes remain alive after {grace_seconds}s grace timeout; executing verified forced termination", level="WARNING")
                for name, pid, child_id, script, marker in remaining:
                    _append_event(self.paths["events"], "forced_shutdown_timeout_reached", pid=pid, target="orphan_worker", worker=name, grace_seconds=grace_seconds)
                    _append_event(self.paths["events"], "forced_termination_requested", pid=pid, target="orphan_worker", worker=name)
                    t_res = terminate_process_tree(pid, exit_code=137, expected_tokens=[script, "python"], timeout=3.0)
                    _append_event(self.paths["events"], "forced_termination_completed", pid=pid, target="orphan_worker", worker=name, **t_res)
                    _append_event(self.paths["events"], "process_tree_verified_exited", pid=pid, target="orphan_worker", worker=name, verified=t_res.get("success", False))
                still_alive = [pid for _, pid, child_id, script, _ in remaining if _pid_matches(pid, self.root, child_id, script)]
                if still_alive:
                    self._write("needs_attention", note="orphan workers failed to terminate", orphan_pids=still_alive)
                    return False

        # A manager crash may leave scoped stop markers behind. They are removed
        # only after every identified process belonging to this campaign is dead.
        if not self.paths["stop"].exists():
            self.paths["classical_stop"].unlink(missing_ok=True)
            self.paths["quantum_stop"].unlink(missing_ok=True)
        return True

    def run(self) -> int:
        plan = _read_plan(self.paths)
        if plan.get("root") != str(self.root):
            raise ValueError("campaign root differs from immutable plan; move the whole project only with the relocation procedure")
        self.paths["base"].mkdir(parents=True, exist_ok=True)
        _archive_reused_pid_lock(self.paths["guardian_lock"], self.root, self.campaign_id, "amde2.py")
        self.lock.acquire()
        awake_prevented = _set_windows_system_awake(True)
        self._log("AMDE 2.0 guardian online")
        if awake_prevented:
            self._log("Windows system sleep request is inhibited while this guardian is alive")
        try:
            prior_guardian = _read_json(self.paths["guardian"], {})
            if isinstance(prior_guardian, dict):
                now_epoch = time.time()
                self.restart_times = [float(value) for value in prior_guardian.get("manager_restart_times_epoch", [])
                                      if isinstance(value, (int, float)) and 0 <= now_epoch - float(value) <= 3600.0]
            existing_state = _read_json(self.paths["state"], {})
            existing_manager_pid = existing_state.get("manager_pid") if isinstance(existing_state, dict) else None
            if _pid_matches(existing_manager_pid, self.root, self.campaign_id, "amde2.py"):
                self.adopted_manager_pid = int(existing_manager_pid)
                self._log(f"Guardian adopted existing manager PID={self.adopted_manager_pid}")
            elif existing_state and existing_state.get("status") not in TERMINAL:
                if not self._recover_orphan_workers(int(plan["shutdown_grace_seconds"])):
                    return 2
            if self.paths["stop"].exists():
                self._write("stopped", note="cooperative stop marker present")
                self._log("Stop marker present; guardian will not start manager")
                return 0
            if isinstance(existing_state, dict) and existing_state.get("status") in {"paused_storage", "paused_resource", "needs_attention", "failed", "manager_error"}:
                terminal_status = str(existing_state.get("status"))
                self._write(terminal_status, note="safety terminal state requires an explicit operator resume")
                self._log(f"Campaign remains in safety terminal state={terminal_status}; explicit resume required")
                return 0
            if plan.get("deadline_policy") == "hard":
                self._log(f"Hard deadline watchdog active: policy=hard, deadline={plan.get('deadline_utc')}")
                _append_event(
                    self.paths["events"],
                    "hard_deadline_watchdog_started",
                    campaign_id=self.campaign_id,
                    deadline_utc=plan.get("deadline_utc"),
                    policy="hard",
                )
            while True:
                self.lock.heartbeat()
                if plan.get("deadline_utc") and not self.paths["stop"].exists():
                    try:
                        deadline_dt = datetime.fromisoformat(str(plan["deadline_utc"]).replace("Z", "+00:00"))
                        now_utc = datetime.now(timezone.utc)
                        if now_utc >= deadline_dt:
                            policy = str(plan.get("deadline_policy", "cooperative")).lower()
                            if policy == "hard":
                                self._log(
                                    f"Guardian detected persistent hard deadline reached ({plan.get('duration_hours')}h); executing operating-system process-tree termination",
                                    level="CRITICAL",
                                )
                                _append_event(
                                    self.paths["events"],
                                    "hard_deadline_reached",
                                    campaign_id=self.campaign_id,
                                    deadline_utc=plan.get("deadline_utc"),
                                    policy="hard",
                                )
                                _append_event(
                                    self.paths["events"],
                                    "hard_termination_requested",
                                    campaign_id=self.campaign_id,
                                    deadline_utc=plan.get("deadline_utc"),
                                )
                                state_data = _read_json(self.paths["state"], {})
                                mgr_pid = int(state_data.get("manager_pid")) if (isinstance(state_data, dict) and state_data.get("manager_pid")) else None
                                if mgr_pid is None or not default_pid_alive(mgr_pid):
                                    mgr_pid = self.child.pid if (self.child and self.child.poll() is None) else self.adopted_manager_pid
                                term_metrics = terminate_process_tree(
                                    mgr_pid,
                                    exit_code=137,
                                    job=self.job,
                                    expected_tokens=["amde2", "python"],
                                    timeout=3.0,
                                )
                                _append_event(
                                    self.paths["events"],
                                    "hard_termination_completed",
                                    campaign_id=self.campaign_id,
                                    **term_metrics,
                                )
                                _append_event(
                                    self.paths["events"],
                                    "process_tree_verified_exited",
                                    campaign_id=self.campaign_id,
                                    verified=term_metrics.get("success", False),
                                    pids_terminated=term_metrics.get("pids_terminated", []),
                                )
                                final_note = f"hard campaign deadline reached ({plan.get('duration_hours')}h); process tree terminated ({term_metrics.get('mechanism')})"
                                self._write(
                                    "stopped",
                                    note=final_note,
                                )
                                state_dict = _read_json(self.paths["state"], {})
                                if isinstance(state_dict, dict):
                                    state_dict["status"] = "stopped"
                                    state_dict["ended_at"] = utc_now()
                                    state_dict["note"] = final_note
                                    _atomic_json(self.paths["state"], state_dict)
                                if self.job:
                                    self.job.close()
                                return 0
                            else:
                                self._log(f"Guardian detected persistent deadline reached ({plan.get('duration_hours')}h); requesting cooperative stop")
                                self.paths["stop"].parent.mkdir(parents=True, exist_ok=True)
                                _atomic_json(self.paths["stop"], {
                                    "requested_at": utc_now(),
                                    "reason": f"persistent campaign deadline reached ({plan.get('duration_hours')}h)",
                                    "campaign_id": self.campaign_id,
                                })
                    except Exception as deadline_exc:
                        self._log(f"Error checking persistent deadline: {deadline_exc}", level="WARNING")
                if self.paths["stop"].exists():
                    adopted_manager_alive = self.child is None and self.adopted_manager_pid is not None and _pid_alive(self.adopted_manager_pid)
                    child_exited = self.child is None or self.child.poll() is not None
                    if child_exited and not adopted_manager_alive:
                        if not self._recover_orphan_workers(int(plan["shutdown_grace_seconds"])):
                            return 2
                        self._write("stopped", note="user requested stop; manager and orphan workers are confirmed exited")
                        return 0
                    self._write("stopping")
                if self.child is None and self.adopted_manager_pid is not None:
                    if _pid_alive(self.adopted_manager_pid):
                        if not self._monitor_manager_heartbeat(self.adopted_manager_pid, plan):
                            self._write("needs_attention", note="manager heartbeat stale and safe termination failed")
                            return 2
                        self._write("monitoring", manager_pid=self.adopted_manager_pid, adopted_manager=True)
                        sleep_dur = 2.0
                        if plan.get("deadline_utc"):
                            try:
                                d_dt = datetime.fromisoformat(str(plan["deadline_utc"]).replace("Z", "+00:00"))
                                rem = (d_dt - datetime.now(timezone.utc)).total_seconds()
                                if rem > 0 and rem < 2.5:
                                    sleep_dur = max(0.05, min(0.2, rem))
                            except Exception:
                                pass
                        time.sleep(sleep_dur)
                        continue
                    self._log(f"Adopted manager PID={self.adopted_manager_pid} exited")
                    self.adopted_manager_pid = None
                if self.child is None:
                    if not self._recover_orphan_workers(int(plan["shutdown_grace_seconds"])):
                        return 2
                    self._launch_manager()
                if self.child.poll() is None:
                    if not self._monitor_manager_heartbeat(self.child.pid, plan):
                        self._write("needs_attention", note="manager heartbeat stale and safe termination failed")
                        return 2
                code = self.child.poll()
                if code is None:
                    self._write("monitoring")
                    sleep_dur = 2.0
                    if plan.get("deadline_utc"):
                        try:
                            d_dt = datetime.fromisoformat(str(plan["deadline_utc"]).replace("Z", "+00:00"))
                            rem = (d_dt - datetime.now(timezone.utc)).total_seconds()
                            if rem > 0 and rem < 2.5:
                                sleep_dur = max(0.05, min(0.2, rem))
                        except Exception:
                            pass
                    time.sleep(sleep_dur)
                    continue
                self._close_log()
                state = _read_json(self.paths["state"], {})
                status = state.get("status") if isinstance(state, dict) else None
                self._log(f"Manager exited code={code}; state={status}")
                if self.paths["stop"].exists():
                    if not self._recover_orphan_workers(int(plan["shutdown_grace_seconds"])):
                        return 2
                    self._write("stopped", manager_exit_code=code, note="manager exited during stop; guardian completed orphan cleanup")
                    return 0
                if status in TERMINAL:
                    self._write(str(status), manager_exit_code=code)
                    return 0 if status in {"stopped", "paused_storage", "paused_resource"} else 2
                if int(code or 0) == 2 or status == "manager_error":
                    self._write("needs_attention", manager_exit_code=code,
                                note="manager exited with a configuration/runtime error; inspect manager.log before resume")
                    self._log("Manager returned a non-retryable error; guardian is stopping instead of entering a crash loop")
                    return 2
                now = time.time()
                self.restart_times = [stamp for stamp in self.restart_times if now - stamp <= 3600.0]
                self.restart_times.append(now)
                count = len(self.restart_times)
                delay = min(300, 5 * (2 ** min(count - 1, 6)))
                if count >= 5:
                    delay = max(delay, 900)
                self._write("manager_recovery", manager_exit_code=code, manager_restart_count=count, next_retry_seconds=delay)
                self._log(f"Unexpected manager exit; restarting in {delay}s (crash count in 1h={count})")
                self.child = None
                time.sleep(delay)
        except KeyboardInterrupt:
            self.paths["stop"].parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.paths["stop"], {"requested_at": utc_now(), "reason": "guardian interrupted"})
            if self.child and self.child.poll() is None:
                self.child.wait(timeout=None)
            self._write("stopped", note="guardian interrupted after cooperative stop")
            return 0
        finally:
            if self.child and self.child.poll() is not None:
                self._close_log()
            _set_windows_system_awake(False)
            self.lock.release()


def launch_guardian(root: Path, campaign_id: str) -> int:
    paths = _paths(root, campaign_id)
    guardian_data = _read_json(paths["guardian"], {})
    if isinstance(guardian_data, dict) and _pid_matches(guardian_data.get("pid"), root, campaign_id, "amde2.py"):
        print(f"Guardian is already alive: PID {guardian_data['pid']}")
        return 1
    script = root / "scripts" / "amde2.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [sys.executable, "-u", str(script), "guardian", "--root", str(root), "--campaign-id", campaign_id]
    paths["guardian_log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["guardian_log"].open("ab", buffering=0) as handle:
        handle.write(f"\n[{utc_now()}] Detached guardian launch: {cmd}\n".encode("utf-8", errors="replace"))
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        child = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                 close_fds=(os.name != "nt"), creationflags=flags, start_new_session=(os.name != "nt"))
    time.sleep(1.0)
    if child.poll() is not None:
        print(f"Guardian exited during startup (code={child.returncode}).")
        return 2
    print(f"AMDE 2.0 guardian started: PID={child.pid}")
    print(f"Campaign state: {paths['state']}")
    print(f"Guardian log: {paths['guardian_log']}")
    return 0


def start_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    campaign_id = _validate_id(args.campaign_id)
    plan = _make_plan(args, root)
    paths = _paths(root, campaign_id)
    if paths["plan"].exists() or paths["state"].exists():
        raise FileExistsError(f"campaign already exists: {paths['base']}; use resume, do not overwrite evidence")
    for worker_name in ("classical", "quantum"):
        worker_state = _worker_paths(root, campaign_id, worker_name)["state"]
        if worker_state.exists():
            raise FileExistsError(f"child campaign state already exists ({worker_state}); choose a new AMDE 2 campaign ID to avoid colliding with previous evidence")
    if shutil.disk_usage(root).free < int(plan["min_free_gb"] * 1024**3):
        raise RuntimeError(f"not enough free disk space: need at least {plan['min_free_gb']} GiB reserve")
    if plan["quantum_device"] == "cuda":
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; run quantum doctor and do not silently fall back to CPU")
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            if free < 1536 * 1024**2:
                raise RuntimeError(f"only {free / 1024**2:.0f} MiB VRAM is free; AMDE requires a 1536 MiB reserve")
        except ImportError as exc:
            raise RuntimeError("CUDA was requested but PyTorch is not installed in this Python environment") from exc
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["control"].mkdir(parents=True, exist_ok=True)
    paths["manager_log"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths["plan"], plan)
    _atomic_json(paths["state"], {
        "schema_version": SCHEMA, "campaign_id": campaign_id, "plan_hash": plan["plan_hash"],
        "status": "created", "created_at": utc_now(), "workers": {}, "note": "plan created; guardian starting",
        "mode": plan.get("mode", "hybrid"), "campaign_type": plan.get("campaign_type", "infinite"),
        "duration_hours": plan.get("duration_hours"), "deadline_utc": plan.get("deadline_utc"),
        "deadline_policy": plan.get("deadline_policy", "cooperative"),
    })
    _append_event(paths["events"], "campaign_created", plan_hash=plan["plan_hash"], plan=plan)
    _append_event(
        paths["events"],
        "deadline_policy_selected",
        campaign_id=campaign_id,
        deadline_policy=plan.get("deadline_policy", "cooperative"),
        deadline_utc=plan.get("deadline_utc"),
    )
    return launch_guardian(root, campaign_id)


def _start_registered_scheduled_task(campaign_id: str, paths: dict[str, Path]) -> bool:
    """Start the campaign's Windows Task Scheduler guardian when installed."""
    if os.name != "nt":
        return False
    task_name = f"AMDE2-{campaign_id}"
    command = (
        "$task = Get-ScheduledTask -TaskName '" + task_name + "' -ErrorAction SilentlyContinue; "
        "if ($null -eq $task) { exit 3 }; Start-ScheduledTask -InputObject $task; exit 0"
    )
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not query/start registered Task Scheduler entry {task_name}: {exc}") from exc
    if result.returncode == 3:
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Task Scheduler failed to start {task_name}: {result.stderr.strip() or result.stdout.strip()}")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        guardian = _read_json(paths["guardian"], {})
        if isinstance(guardian, dict) and _pid_alive(guardian.get("pid")):
            print(f"Resumed via Windows Task Scheduler: {task_name} (guardian PID={guardian['pid']})")
            return True
        time.sleep(0.5)
    raise RuntimeError(f"Task Scheduler accepted {task_name}, but no live guardian heartbeat appeared within 10 seconds; inspect Task Scheduler History and guardian.log")


def resume_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    campaign_id = _validate_id(args.campaign_id)
    paths = _paths(root, campaign_id)
    plan = _read_plan(paths)
    if plan.get("root") != str(root):
        raise ValueError("this campaign plan is pinned to another project root")
    guardian = _read_json(paths["guardian"], {})
    state = _read_json(paths["state"], {})
    if _pid_matches(guardian.get("pid") if isinstance(guardian, dict) else None, root, campaign_id, "amde2.py"):
        raise RuntimeError("guardian is still running; do not start a second copy")
    if _pid_matches(state.get("manager_pid") if isinstance(state, dict) else None, root, campaign_id, "amde2.py"):
        raise RuntimeError("manager is still running; inspect status before attempting resume")
    for name in ("classical", "quantum"):
        wp = _worker_paths(root, campaign_id, name)
        lock = _read_json(wp["lock"], {})
        expected_script = "research_campaign.py" if name == "classical" else "quantum_research_campaign.py"
        if _pid_matches(lock.get("pid") if isinstance(lock, dict) else None, root, f"{campaign_id}_{name}", expected_script):
            raise RuntimeError(f"{name} worker still alive (PID {lock.get('pid')}); refusing duplicate resume")
    if state.get("status") == "running" and not args.force_recover:
        age = _path_age_seconds(paths["state"])
        if age is not None and age < 30:
            raise RuntimeError("state looks recently active; use status first. --force-recover is only for a verified orphaned manager.")
    if isinstance(state, dict) and state.get("status") in {"needs_attention", "failed", "manager_error"} and not args.force_recover:
        raise RuntimeError("campaign is in a failed/needs_attention state; inspect logs and use --force-recover only after resolving the recorded cause, or create a new campaign with a corrected immutable plan")
    # Remove only this campaign's stop markers. Project-wide legacy markers are untouched.
    for marker in (paths["stop"], paths["classical_stop"], paths["quantum_stop"]):
        marker.unlink(missing_ok=True)
    if isinstance(state, dict):
        state["status"] = "resume_requested"
        state["resume_requested_at"] = utc_now()
        state["updated_at"] = utc_now()
        _atomic_json(paths["state"], state)
    _append_event(paths["events"], "campaign_resume_requested", force_recover=bool(args.force_recover))
    # When the optional Windows auto-start task exists, keep the guardian under
    # Task Scheduler so its configured restart-on-failure policy remains active.
    if _start_registered_scheduled_task(campaign_id, paths):
        return 0
    return launch_guardian(root, campaign_id)


def stop_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    campaign_id = _validate_id(args.campaign_id)
    paths = _paths(root, campaign_id)
    if not paths["plan"].exists():
        print("Campaign not found; no stop signal was written.")
        return 2
    paths["stop"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths["stop"], {"requested_at": utc_now(), "reason": "user requested cooperative stop", "campaign_id": campaign_id})
    print("Campaign-scoped cooperative stop requested. Workers will finish the current safe unit, commit their ledger/checkpoint, and exit.")
    deadline = time.monotonic() + max(1, int(args.wait_seconds))
    while time.monotonic() < deadline:
        state = _read_json(paths["state"], {})
        guardian = _read_json(paths["guardian"], {})
        guardian_alive = _pid_alive(guardian.get("pid") if isinstance(guardian, dict) else None)
        manager_alive = _pid_alive(state.get("manager_pid") if isinstance(state, dict) else None)
        workers = state.get("workers", {}) if isinstance(state, dict) else {}
        worker_pids_alive = any(
            _pid_alive((v or {}).get(key))
            for v in workers.values() for key in ("pid", "relay_pid")
        )
        if not guardian_alive and not manager_alive and not worker_pids_alive:
            final = state.get("status") if isinstance(state, dict) else "unknown"
            if final == "stopped":
                print("Shutdown complete: all supervised processes have exited.")
                return 0
            print(f"Processes exited, terminal state={final}. Inspect reports/amde2/{campaign_id}/logs before resume.")
            return 1
        time.sleep(1.0)
    print("Stop request remains active, but shutdown exceeded the wait limit. Do not force-kill processes yet.")
    print(f"Inspect: python scripts/amde2.py status --campaign-id {campaign_id}")
    print(f"Logs:    python scripts/amde2.py logs --campaign-id {campaign_id} --tail 100")
    print(f"After inspection only, use: python scripts/amde2.py force-stop --campaign-id {campaign_id}")
    return 2


def force_stop_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    campaign_id = _validate_id(args.campaign_id)
    paths = _paths(root, campaign_id)
    paths["stop"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths["stop"], {"requested_at": utc_now(), "reason": "explicit force-stop", "campaign_id": campaign_id})
    state = _read_json(paths["state"], {})
    guardian = _read_json(paths["guardian"], {})
    pids: set[int] = set()
    if isinstance(state, dict):
        for item in (state.get("workers") or {}).values():
            if isinstance(item, dict):
                for pid_key in ("pid", "relay_pid"):
                    try:
                        if item.get(pid_key):
                            pids.add(int(item[pid_key]))
                    except (ValueError, TypeError):
                        pass
        for key in ("manager_pid",):
            try:
                if state.get(key):
                    pids.add(int(state[key]))
            except (ValueError, TypeError):
                pass
    if isinstance(guardian, dict):
        try:
            if guardian.get("pid"):
                pids.add(int(guardian["pid"]))
        except (ValueError, TypeError):
            pass
    if not pids:
        print("No process IDs are recorded. Stop marker retained; inspect campaign state.")
        return 1
    try:
        import psutil  # type: ignore
    except ImportError:
        print("Force-stop requires psutil to verify PID identity safely. Install it with: python -m pip install psutil")
        print("The cooperative stop marker is already present; do not use taskkill on unverified PIDs.")
        return 2

    root_text = str(root).casefold()
    campaign_token = campaign_id.casefold()
    verified: dict[int, Any] = {}
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).casefold()
            # PID reuse is common on long-running Windows systems. Only touch
            # a process whose command line identifies both this checkout and
            # this exact campaign ID.
            expected_commands = ("amde2.py", "amde2_logrelay.py", "research_campaign.py", "quantum_research_campaign.py")
            if root_text in cmdline and campaign_token in cmdline and any(token in cmdline for token in expected_commands):
                verified[proc.pid] = proc
                for child in proc.children(recursive=True):
                    try:
                        child_cmd = " ".join(child.cmdline()).casefold()
                        if root_text in child_cmd and campaign_token in child_cmd and "python" in (child.name() or "").casefold():
                            verified[child.pid] = child
                    except psutil.Error:
                        continue
        except psutil.Error:
            continue
    if not verified:
        print("No live process could be positively matched to this campaign. No process was killed; stop marker retained.")
        return 1
    for proc in sorted(verified.values(), key=lambda item: item.pid, reverse=True):
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(verified.values()), timeout=8)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    print(f"Force-stop issued for {len(verified)} verified processes. An in-flight unit may be lost; verify checkpoints before resume.")
    return 0


def doctor(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("project_root", root.is_dir(), str(root)))
    required = ("scripts/research_campaign.py", "scripts/quantum_research_campaign.py",
                "scripts/amde2.py", "scripts/amde2_logrelay.py", "config/default.json")
    missing = [item for item in required if not (root / item).is_file()]
    checks.append(("runtime_files", not missing, "all required files present" if not missing else f"missing: {missing}"))
    disk = shutil.disk_usage(root)
    free_gb = disk.free / 1024**3
    checks.append(("disk_reserve", free_gb >= args.min_free_gb, f"{free_gb:.2f} GiB free; minimum {args.min_free_gb:.2f} GiB"))
    try:
        import psutil  # type: ignore
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024**3
        checks.append(("psutil_resource_monitor", True, f"{psutil.__version__}; RAM available {available_gb:.2f}/{memory.total / 1024**3:.2f} GiB"))
    except Exception as exc:
        checks.append(("psutil_resource_monitor", False, f"{type(exc).__name__}: {exc}; install psutil>=5.9"))
    if args.device == "cuda":
        try:
            import torch  # type: ignore
            available = bool(torch.cuda.is_available())
            if available:
                free, total = torch.cuda.mem_get_info()
                name = torch.cuda.get_device_name(0)
                checks.append(("cuda", True, f"{name}; PyTorch {torch.__version__}; free VRAM {free / 1024**2:.0f}/{total / 1024**2:.0f} MiB"))
                checks.append(("vram_reserve", free >= int(args.min_free_vram_mb * 1024**2), f"{free / 1024**2:.0f} MiB free; minimum {args.min_free_vram_mb:.0f} MiB"))
            else:
                checks.append(("cuda", False, f"PyTorch {torch.__version__} reports CUDA unavailable"))
        except Exception as exc:
            checks.append(("cuda", False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(("cuda", True, "not requested; CPU-only check"))
    if os.name == "nt":
        try:
            guard_on = _set_windows_system_awake(True)
            guard_off = _set_windows_system_awake(False)
            checks.append(("windows_sleep_guard", bool(guard_on and guard_off),
                           "SetThreadExecutionState API return verified and execution state reset clean"))
        except Exception as exc:
            checks.append(("windows_sleep_guard", False, f"SetThreadExecutionState check failed: {exc}"))
    else:
        checks.append(("windows_sleep_guard", True, "not applicable on this platform (POSIX)"))
    failed = [name for name, ok, _ in checks if not ok]
    print(f"AMDE 2.0 doctor | Python {sys.version.split()[0]} | platform={sys.platform}")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if failed:
        print("Doctor failed: " + ", ".join(failed))
        return 1
    print("Doctor PASS. This is a preflight, not a 24-hour reliability qualification.")
    return 0


def status_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    campaign_id = _validate_id(args.campaign_id)
    paths = _paths(root, campaign_id)
    state = _read_json(paths["state"])
    if not isinstance(state, dict):
        print("Campaign state not found.")
        return 2
    guardian = _read_json(paths["guardian"], {})
    guardian_pid = guardian.get("pid") if isinstance(guardian, dict) else None
    manager_pid = state.get("manager_pid")
    print(f"AMDE 2.0 campaign: {campaign_id}")
    plan = _read_plan(paths)
    if plan:
        d_policy = plan.get("deadline_policy", "cooperative")
        print(f"Mode: {plan.get('mode', 'hybrid')} | Campaign type: {plan.get('campaign_type', 'infinite')} | Deadline policy: {d_policy}")
        if plan.get("deadline_utc"):
            try:
                deadline_dt = datetime.fromisoformat(str(plan["deadline_utc"]).replace("Z", "+00:00"))
                rem = (deadline_dt - datetime.now(timezone.utc)).total_seconds()
                rem_str = f"{rem/3600:.2f}h remaining" if rem > 0 else "deadline elapsed"
                print(f"Deadline: {plan['deadline_utc']} (policy={d_policy}, duration={plan.get('duration_hours')}h, {rem_str})")
            except Exception:
                print(f"Deadline: {plan.get('deadline_utc')} (policy={d_policy})")
    print(f"Campaign state: {state.get('status')}")
    print(f"Guardian state: {guardian.get('status') if isinstance(guardian, dict) else 'unknown'}")
    print(f"Guardian: PID={guardian_pid} alive={_pid_alive(guardian_pid)} heartbeat={guardian.get('heartbeat_at') if isinstance(guardian, dict) else '-'}")
    print(f"Manager: PID={manager_pid} alive={_pid_alive(manager_pid)} heartbeat={state.get('heartbeat_at')}")
    print(f"Plan hash: {state.get('plan_hash')}")
    print(f"Note: {state.get('note') or '-'}")
    health = state.get("health") or _read_json(paths["base"] / "health.json", {})
    if isinstance(health, dict):
        print(f"Disk free: {health.get('disk_free_gb', 'n/a')} GiB | RAM available: {health.get('memory_available_gb', 'n/a')} GiB | CPU: {health.get('cpu_percent', 'n/a')}%")
        if health.get("gpu"):
            print(f"GPU: {health['gpu']}")
    for name, child in (state.get("workers") or {}).items():
        child = child or {}
        age = None
        stamp = child.get("heartbeat_at")
        if isinstance(stamp, str):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
            except ValueError:
                pass
        age_text = f"{age:.0f}s ago" if age is not None else "unknown"
        print(f"{name}: alive={child.get('alive')} pid={child.get('pid')} manager_status={child.get('status')} campaign_status={child.get('campaign_status')} round={child.get('current_round')} next={child.get('next_round')} restarts={child.get('restarts')} heartbeat={age_text}")
    if paths["stop"].exists():
        print("Stop marker: present (this campaign only)")
    return 0


def tail_file(path: Path, count: int) -> str:
    if not path.is_file():
        return "<not created>"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-max(1, int(count)):])
    except OSError as exc:
        return f"<cannot read: {exc}>"


def logs_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = _paths(root, _validate_id(args.campaign_id))
    for label, path in (("GUARDIAN", paths["guardian_log"]), ("MANAGER", paths["manager_log"]),
                        ("CLASSICAL", paths["base"] / "logs" / "classical.log"),
                        ("QUANTUM", paths["base"] / "logs" / "quantum.log")):
        print(f"\n===== {label} =====")
        print(tail_file(path, args.tail))
    return 0


def _sqlite_backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    source_uri = src.resolve().as_uri() + "?mode=ro"
    source = sqlite3.connect(source_uri, uri=True, timeout=30)
    target = sqlite3.connect(dst, timeout=30)
    try:
        source.backup(target, pages=256, sleep=0.05)
        result = target.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"SQLite integrity check failed for {src}: {result}")
    finally:
        target.close()
        source.close()


def backup_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    cid = _validate_id(args.campaign_id)
    paths = _paths(root, cid)
    plan = _read_plan(paths)
    base = paths["base"]
    backup_dir = base / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output).resolve() if args.output else backup_dir / f"{cid}_{stamp}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    classical_id, quantum_id = f"{cid}_classical", f"{cid}_quantum"
    roots: list[Path] = [base, root / "reports" / "campaigns" / classical_id,
                         root / "reports" / "quantum_campaigns" / quantum_id,
                         root / "config", root / "data", root / "storage" / "discovery_archive.sqlite3"]
    runs_dir = root / "runs"
    run_directories: list[Path] = []
    if runs_dir.exists():
        run_directories = [p for p in runs_dir.glob(f"{classical_id}_r*") if p.is_dir()]
        roots.extend(run_directories)
    reports_dir = root / "reports"
    for run_dir in run_directories:
        report = reports_dir / f"{run_dir.name}_report.md"
        if report.is_file():
            roots.append(report)
    artifacts_dir = root / "artifacts"
    if artifacts_dir.exists():
        roots.extend(p for p in artifacts_dir.glob(f"{classical_id}_r*") if p.is_dir())
    roots = [p for p in roots if p.exists()]
    manifest: list[dict[str, Any]] = []
    partial_output = output.with_suffix(output.suffix + ".partial")
    if partial_output.exists():
        partial_output.unlink()
    try:
        with tempfile.TemporaryDirectory(prefix="amde2-backup-") as temp_dir:
            temp = Path(temp_dir)
            with zipfile.ZipFile(partial_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
                for source_root in roots:
                    for path in [source_root, *source_root.rglob("*")]:
                        if path.is_symlink() or not path.is_file():
                            continue
                        if "backups" in path.parts and base in path.parents:
                            continue
                        if path.name == "STOP" or path.name.endswith(".STOP") or path.name.endswith(".lock") or ".lock." in path.name:
                            continue
                        relative = path.relative_to(root).as_posix()
                        try:
                            if path.name.endswith((".sqlite3", ".db")):
                                snap = temp / hashlib.sha256(relative.encode()).hexdigest()[:16] / path.name
                                _sqlite_backup(path, snap)
                                payload_path = snap
                                zip_name = relative
                            elif path.name.endswith(("-wal", "-shm")):
                                continue
                            else:
                                payload_path = path
                                zip_name = relative
                            archive.write(payload_path, zip_name)
                            digest = hashlib.sha256()
                            size = 0
                            with payload_path.open("rb") as handle:
                                for block in iter(lambda: handle.read(1024 * 1024), b""):
                                    digest.update(block)
                                    size += len(block)
                            manifest.append({"path": zip_name, "size": size, "sha256": digest.hexdigest()})
                        except Exception as exc:
                            raise RuntimeError(f"backup failed while processing {path}: {type(exc).__name__}: {exc}") from exc
                archive.writestr("BACKUP_MANIFEST.json", json.dumps({"schema": "amde2-backup-v1", "created_at": utc_now(),
                                                                      "campaign_id": cid, "plan_hash": plan["plan_hash"],
                                                                      "files": manifest}, indent=2, sort_keys=True))
        # Verify before publishing the final filename. A failed ZIP is never
        # advertised as a completed recovery point.
        with zipfile.ZipFile(partial_output, "r") as check:
            bad = check.testzip()
            if bad:
                raise RuntimeError(f"backup ZIP CRC failed for {bad}")
        robust_replace(partial_output, output)
    except BaseException:
        partial_output.unlink(missing_ok=True)
        raise
    print(f"Backup created: {output}")
    print(f"Files: {len(manifest)} | Size: {output.stat().st_size / 1024**2:.2f} MiB")
    return 0


def verify_backup(args: argparse.Namespace) -> int:
    path = Path(args.archive).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        manifest = json.loads(archive.read("BACKUP_MANIFEST.json"))
        checked = 0
        for item in manifest.get("files", []):
            digest = hashlib.sha256()
            with archive.open(item["path"], "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {item['path']}")
            if item["path"].lower().endswith((".sqlite3", ".db")):
                with tempfile.TemporaryDirectory(prefix="amde2-verify-db-") as temp_dir:
                    temp_db = Path(temp_dir) / Path(item["path"]).name
                    with archive.open(item["path"], "r") as source, temp_db.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    connection_uri = temp_db.resolve().as_uri() + "?mode=ro"
                    connection = sqlite3.connect(connection_uri, uri=True, timeout=30)
                    try:
                        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                    finally:
                        connection.close()
                    if integrity != "ok":
                        raise RuntimeError(f"SQLite integrity check failed in backup: {item['path']}: {integrity}")
            checked += 1
    print(f"Backup verification PASS: {checked} files SHA-256 verified; ZIP CRC and SQLite integrity clean.")
    return 0


def restore_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    verify_backup(argparse.Namespace(archive=str(archive_path)))
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("BACKUP_MANIFEST.json"))
        campaign_id = _validate_id(manifest.get("campaign_id"))
        paths = _paths(root, campaign_id)
        # A live manager/guardian/worker makes any restore unsafe.
        if paths["base"].exists():
            state = _read_json(paths["state"], {})
            guardian = _read_json(paths["guardian"], {})
            if _pid_matches(guardian.get("pid") if isinstance(guardian, dict) else None, root, campaign_id, "amde2.py"):
                raise RuntimeError("cannot restore while the campaign guardian is running")
            if _pid_matches(state.get("manager_pid") if isinstance(state, dict) else None, root, campaign_id, "amde2.py"):
                raise RuntimeError("cannot restore while the campaign manager is running")
        for worker_name in ("classical", "quantum"):
            wp = _worker_paths(root, campaign_id, worker_name)
            lock = _read_json(wp["lock"], {})
            expected_script = "research_campaign.py" if worker_name == "classical" else "quantum_research_campaign.py"
            if _pid_matches(lock.get("pid") if isinstance(lock, dict) else None, root,
                            f"{campaign_id}_{worker_name}", expected_script):
                raise RuntimeError(f"cannot restore while {worker_name} worker PID={lock.get('pid')} is running")
            state = _read_json(paths["state"], {})
            relay_pid = ((state.get("workers", {}).get(worker_name, {}) or {}).get("relay_pid")
                         if isinstance(state, dict) else None)
            if _pid_matches(relay_pid, root, f"{campaign_id}_{worker_name}", "amde2_logrelay.py"):
                raise RuntimeError(f"cannot restore while {worker_name} log relay PID={relay_pid} is running")
        members = [item["path"] for item in manifest.get("files", [])]
        required_bytes = sum(int(item.get("size", 0)) for item in manifest.get("files", []))
        if shutil.disk_usage(root).free < required_bytes:
            raise RuntimeError(f"insufficient free space for restore; need at least {required_bytes} bytes")
        targets: list[tuple[str, Path]] = []
        for member in members:
            relative = Path(member.replace("/", os.sep))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive path: {member}")
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"archive path escapes restore root: {member}") from exc
            targets.append((member, target))
        conflicts = [str(target) for _, target in targets if target.exists()]
        if conflicts and not args.overwrite:
            raise FileExistsError("restore would overwrite existing files; use --overwrite only after taking a separate backup. First conflicts: " + ", ".join(conflicts[:8]))
        with tempfile.TemporaryDirectory(prefix="amde2-restore-") as temp_dir:
            temp_root = Path(temp_dir)
            for member, _ in targets:
                staged = temp_root / Path(member.replace("/", os.sep))
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, staged.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

            staged_plan = temp_root / "reports" / "amde2" / campaign_id / "plan.json"
            if not staged_plan.is_file():
                raise RuntimeError("verified backup is missing its AMDE 2.0 campaign plan")
            restored_plan = json.loads(staged_plan.read_text(encoding="utf-8"))
            restored_plan["root"] = str(root)
            restored_plan["plan_hash"] = _json_hash({key: value for key, value in restored_plan.items() if key != "plan_hash"})
            _atomic_json(staged_plan, restored_plan)

            staged_state = temp_root / "reports" / "amde2" / campaign_id / "state.json"
            if staged_state.is_file():
                restored_state = json.loads(staged_state.read_text(encoding="utf-8"))
                restored_state.update({"plan_hash": restored_plan["plan_hash"], "status": "stopped",
                                       "manager_pid": None, "guardian_pid": None, "restored_at": utc_now(),
                                       "note": "restored from verified backup; resume explicitly after verification"})
                for worker in (restored_state.get("workers") or {}).values():
                    if isinstance(worker, dict):
                        worker.update({"pid": None, "relay_pid": None, "alive": False, "status": "stopped"})
                _atomic_json(staged_state, restored_state)

            staged_guardian = temp_root / "reports" / "amde2" / campaign_id / "control" / "guardian.json"
            if staged_guardian.is_file():
                guardian_state = json.loads(staged_guardian.read_text(encoding="utf-8"))
                guardian_state.update({"pid": None, "manager_pid": None, "status": "stopped",
                                       "updated_at": utc_now(), "note": "restored backup; transient guardian identity cleared"})
                _atomic_json(staged_guardian, guardian_state)

            for member, target in targets:
                staged = temp_root / Path(member.replace("/", os.sep))
                target.parent.mkdir(parents=True, exist_ok=True)
                robust_replace(staged, target)
        print(f"Restored campaign {campaign_id}: {len(targets)} verified files into {root}")
        print(f"Campaign root rebased; new plan hash={restored_plan['plan_hash']}")
        print("Transient locks and stop markers were not restored. No worker was started. Run doctor, verify, then resume explicitly.")
        return 0


def verify_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    cid = _validate_id(args.campaign_id)
    paths = _paths(root, cid)
    plan = _read_plan(paths)
    problems: list[str] = []
    checked_json = 0
    checked_sqlite = 0
    checkpoints = 0

    if not plan:
        problems.append(f"plan.json is missing or corrupted for campaign {cid}")
    else:
        checked_json += 1
        expected_hash = _json_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        if plan.get("plan_hash") != expected_hash:
            problems.append(f"plan.json plan_hash mismatch: recorded={plan.get('plan_hash')} expected={expected_hash}")

    state = _read_json(paths["state"])
    if not isinstance(state, dict):
        problems.append(f"state.json is missing or corrupted for campaign {cid}")
    else:
        checked_json += 1
        if plan and state.get("plan_hash") != plan.get("plan_hash"):
            problems.append(f"state.json plan_hash does not match plan.json")

    guardian = _read_json(paths["guardian"])
    if isinstance(guardian, dict):
        checked_json += 1

    run_dirs = list((root / "runs").glob(f"{cid}_classical_r*"))
    for run_dir in run_dirs:
        latest = run_dir / "checkpoints" / "latest.json"
        candidates = [latest, *sorted((run_dir / "checkpoints" / "backups").glob("gen_*.json"), reverse=True)]
        valid = False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                envelope = json.loads(candidate.read_text(encoding="utf-8"))
                if envelope.get("checksum") == _json_hash(envelope.get("state", {})) and envelope.get("schema_version") == "1.0.0":
                    checkpoints += 1
                    valid = True
                    break
            except Exception:
                continue
        ledger_path = run_dir / "ledger.sqlite3"
        has_trials = False
        if ledger_path.is_file():
            try:
                c = sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
                row = c.execute("SELECT COUNT(*) FROM trials WHERE status='ok'").fetchone()
                c.close()
                if row and row[0] > 0:
                    has_trials = True
            except Exception:
                pass
        if not valid and (latest.exists() or has_trials):
            problems.append(f"missing or invalid required checkpoint for run directory: {run_dir.name}")

    db_paths = [root / "reports" / "quantum_campaigns" / f"{cid}_quantum" / "quantum_ledger.sqlite3"]
    db_paths.extend(p / "ledger.sqlite3" for p in run_dirs if (p / "ledger.sqlite3").is_file())
    for db in db_paths:
        if not db.is_file():
            continue
        try:
            con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            result = con.execute("PRAGMA integrity_check").fetchone()[0]
            con.close()
            checked_sqlite += 1
            if result != "ok":
                problems.append(f"SQLite integrity failure {db}: {result}")
        except Exception as exc:
            problems.append(f"SQLite read failure {db}: {type(exc).__name__}: {exc}")
    print(f"Plan: {'valid' if plan else 'invalid'} | JSON files: {checked_json} | SQLite DBs: {checked_sqlite} | valid checkpoints found: {checkpoints}")
    if problems:
        print("Verification findings:")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print("AMDE 2.0 evidence verification PASS. This checks file/database integrity, not scientific validity.")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--campaign-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMDE 2.0 long-running autonomous ML research runtime")
    sub = parser.add_subparsers(dest="action", required=True)

    start = sub.add_parser("start", help="Create and start a new immutable hybrid campaign")
    _add_common(start)
    start.add_argument("--mode", choices=("hybrid", "classical", "quantum"), default="hybrid", help="Research execution mode (default: hybrid)")
    start.add_argument("--duration-hours", type=float, default=None, help="Campaign duration in hours; cooperatively stops at persistent deadline")
    start.add_argument("--seed", type=int, default=2026)
    start.add_argument("--hours-per-round", type=float, default=0.25)
    start.add_argument("--max-trials-per-round", type=int, default=100)
    start.add_argument("--classical-max-gb", type=float, default=12.0)
    start.add_argument("--quantum-max-gb", type=float, default=8.0)
    start.add_argument("--min-free-gb", type=float, default=10.0)
    start.add_argument("--quantum-device", choices=("cuda", "cpu", "auto"), default="cuda")
    start.add_argument("--classical-threads", type=int, default=4)
    start.add_argument("--quantum-cpu-threads", type=int, default=2)
    start.add_argument("--quantum-profile", choices=("baseline", "medium", "high", "adaptive"), default=None, help="Hardware compute profile for quantum simulation (default: baseline)")
    start.add_argument("--quantum-epochs", type=int, default=6)
    start.add_argument("--quantum-batch-size", type=int, default=16)
    start.add_argument("--quantum-train-size", type=int, default=64)
    start.add_argument("--quantum-val-size", type=int, default=32)
    start.add_argument("--quantum-test-size", type=int, default=64)
    start.add_argument("--quantum-max-configs", type=int, default=3)
    start.add_argument("--quantum-max-qubits", type=int, default=8)
    start.add_argument("--quantum-statevector-mb", type=int, default=256)
    start.add_argument("--memory-reserve-gb", type=float, default=2.0)
    start.add_argument("--stale-heartbeat-seconds", type=int, default=900)
    start.add_argument("--manager-stale-seconds", type=int, default=180)
    start.add_argument("--stable-after-seconds", type=int, default=600)
    start.add_argument("--max-worker-restarts-per-hour", type=int, default=8)
    start.add_argument("--degraded-retry-seconds", type=int, default=1800)
    start.add_argument("--shutdown-grace-seconds", type=float, default=900.0, help="Cooperative shutdown grace period in seconds before forced termination fallback (default: 900.0)")
    start.add_argument("--deadline-policy", choices=("cooperative", "hard"), default="cooperative", help="Deadline enforcement policy: cooperative (default) or hard (enforces strict process-tree termination at deadline)")
    start.add_argument("--health-interval-seconds", type=int, default=30)
    start.add_argument("--max-gpu-temperature-c", type=float, default=80.0)

    for action in ("resume", "status", "logs", "stop", "force-stop", "guardian", "run", "verify", "backup"):
        p = sub.add_parser(action)
        _add_common(p)
        if action == "resume":
            p.add_argument("--force-recover", action="store_true")
        elif action == "stop":
            p.add_argument("--wait-seconds", type=int, default=900)
        elif action == "logs":
            p.add_argument("--tail", type=int, default=60)
        elif action == "backup":
            p.add_argument("--output", default=None)
        elif action == "restore":
            p.add_argument("archive")
            p.add_argument("--overwrite", action="store_true")
    check = sub.add_parser("doctor", help="Check runtime files, free space, memory and CUDA preflight")
    check.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    check.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    check.add_argument("--min-free-gb", type=float, default=5.0)
    check.add_argument("--min-free-vram-mb", type=float, default=1536.0)

    restore = sub.add_parser("restore", help="Restore a verified campaign backup into a project root")
    restore.add_argument("archive")
    restore.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    restore.add_argument("--overwrite", action="store_true")

    vb = sub.add_parser("verify-backup")
    vb.add_argument("archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.root).resolve() if hasattr(args, "root") else None
        cid = _validate_id(args.campaign_id) if hasattr(args, "campaign_id") else None
        if args.action == "start":
            return start_campaign(args)
        if args.action == "resume":
            return resume_campaign(args)
        if args.action == "stop":
            return stop_campaign(args)
        if args.action == "force-stop":
            return force_stop_campaign(args)
        if args.action == "status":
            return status_campaign(args)
        if args.action == "logs":
            return logs_campaign(args)
        if args.action == "guardian":
            return Guardian(root, cid).run()
        if args.action == "run":
            paths = _paths(root, cid)
            return AMDE2Manager(root, cid, _read_plan(paths)).run()
        if args.action == "verify":
            return verify_campaign(args)
        if args.action == "doctor":
            return doctor(args)
        if args.action == "backup":
            return backup_campaign(args)
        if args.action == "restore":
            return restore_campaign(args)
        if args.action == "verify-backup":
            return verify_backup(args)
        parser.error("unknown action")
    except KeyboardInterrupt:
        print("Interrupted. Use the campaign-scoped stop command; preserve state and logs.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"AMDE 2.0 {getattr(args, 'action', 'command')} error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
