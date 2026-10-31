#!/usr/bin/env python3
"""Crash-resumable quantum/classical simulation campaign.

Runs small variational quantum classifiers on a PyTorch statevector simulator,
compares them with matched CPU MLP baselines, and records an idealized Grover
maze query-model experiment. It does not execute on physical quantum hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.benchmarks.generators import CHALLENGE_GENERATORS, GENERATORS  # noqa: E402
from amde.errors import (  # noqa: E402
    AMDEBaseError,
    CheckpointError,
    ConfigError,
    DataIntegrityError,
    FAILURE_CATEGORY_CONFIG,
    FAILURE_CATEGORY_DATA_INTEGRITY,
    FAILURE_CATEGORY_RESOURCE,
    FAILURE_CATEGORY_RUNTIME,
    LockError,
    ResourcePressure,
    EXIT_CONFIG_ERROR,
    EXIT_DATA_INTEGRITY,
    EXIT_RESOURCE_EXHAUSTION,
    EXIT_RUNTIME_ERROR,
)
from amde.quantum.maze import run_maze_grover_demo  # noqa: E402
from amde.quantum.kernel import train_quantum_kernel, train_rbf_kernel_baseline  # noqa: E402
from amde.quantum.statevector import QuantumBackendError, load_torch, resolve_device  # noqa: E402
from amde.quantum.vqc import VQCConfig, train_mlp_baseline, train_vqc  # noqa: E402
from amde.runtime.lock import SingleInstanceLock  # noqa: E402
from amde.util import atomic_write_json, atomic_write_text, hash_json, read_json, sanitize_id, seed_from, utc_now  # noqa: E402

QUANTUM_CAMPAIGN_PROTOCOL = "amde-quantum-classical-campaign-v1"
TASKS: tuple[dict[str, Any], ...] = (
    {"task_id": "q_xor", "family": "xor", "generator": "xor_classification", "n_features": 2, "shifted_test": False},
    {"task_id": "q_moons", "family": "moons", "generator": "moons_classification", "n_features": 2, "shifted_test": True},
    {"task_id": "q_shifted_xor", "family": "shifted_xor", "generator": "shifted_xor_classification", "n_features": 3, "shifted_test": True},
    {"task_id": "q_parity", "family": "parity", "generator": "parity_classification", "n_features": 4, "shifted_test": False},
)
BASE_VQC_CONFIGS: tuple[dict[str, Any], ...] = (
    {"n_qubits": 4, "n_layers": 1, "gradient_method": "autograd", "optimizer": "adam", "schedule": "constant", "learning_rate": 0.02, "feature_map": "angle_ry", "entanglement": "linear"},
    {"n_qubits": 6, "n_layers": 2, "gradient_method": "autograd", "optimizer": "adam", "schedule": "cosine", "learning_rate": 0.015, "feature_map": "angle_ry_rz", "entanglement": "ring"},
    {"n_qubits": 8, "n_layers": 2, "gradient_method": "autograd", "optimizer": "adamw", "schedule": "cosine", "learning_rate": 0.01, "feature_map": "angle_ry", "entanglement": "ring"},
)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)




def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch CUDA is unavailable"}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    temperature = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raw = completed.stdout.strip().splitlines()[0].strip()
            temperature = float(raw) if raw.replace(".", "", 1).isdigit() else None
    except Exception:
        pass
    return {
        "available": True,
        "free_vram_mb": free_bytes / (1024**2),
        "total_vram_mb": total_bytes / (1024**2),
        "temperature_c": temperature,
    }


def _wait_for_gpu_budget(root: Path, torch: Any, *, stop_path: Path | None = None, isolated_stop: bool = False,
                         heartbeat_callback: Any = None, min_free_mb: float = 1536.0,
                         max_temperature_c: float = 80.0, wait_limit_seconds: float = 1800.0) -> bool:
    """Pause the quantum worker under VRAM/thermal pressure instead of fallback.

    CPU research continues independently while the GPU branch is paused.
    Returns False only when a cooperative stop is requested.
    """
    started = time.monotonic()
    while True:
        if heartbeat_callback is not None:
            try:
                heartbeat_callback()
            except Exception:
                pass
        if stop_path is not None and stop_path.exists():
            return False
        if not isolated_stop and ((root / "runs" / "QUANTUM_STOP").exists() or (root / "runs" / "HYBRID_STOP").exists()):
            return False
        snapshot = _gpu_snapshot(torch)
        if not snapshot.get("available"):
            raise QuantumBackendError("CUDA disappeared while the quantum campaign was running")
        free_ok = float(snapshot.get("free_vram_mb", 0.0)) >= min_free_mb
        temp = snapshot.get("temperature_c")
        thermal_ok = not isinstance(temp, (int, float)) or float(temp) < max_temperature_c
        if free_ok and thermal_ok:
            return True
        elapsed = time.monotonic() - started
        print(
            f"GPU worker paused: free_vram={snapshot.get('free_vram_mb')} MB "
            f"temperature={temp} C; waiting for reserve (elapsed={elapsed:.0f}s)",
            flush=True,
        )
        if elapsed >= wait_limit_seconds:
            raise QuantumBackendError("GPU resource reserve did not recover within 30 minutes")
        time.sleep(30.0)


def _write_heartbeat(state_path: Path, state: dict[str, Any], *, round_index: int,
                     task_id: str, method: str, device: str) -> None:
    state["heartbeat_at"] = utc_now()
    state["current_round"] = round_index
    state["current_task"] = task_id
    state["current_method"] = method
    state["active_device"] = device
    state["updated_at"] = utc_now()
    try:
        atomic_write_json(state_path, state)
    except Exception as exc:
        print(f"[warning] failed to flush quantum heartbeat to {state_path.name}: {exc}", file=sys.stderr, flush=True)


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for current, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in files:
            item = Path(current) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    return total


def _disk_free_bytes(root: Path) -> int:
    import shutil
    return shutil.disk_usage(root).free


def _connect(db_path: Path) -> sqlite3.Connection:
    _mkdir(db_path.parent)
    con = sqlite3.connect(db_path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_trials (
            trial_key TEXT PRIMARY KEY,
            round_index INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            family TEXT NOT NULL,
            method TEXT NOT NULL,
            config_json TEXT NOT NULL,
            seed INTEGER NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            started_at TEXT,
            ended_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS maze_trials (
            round_index INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            ended_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            round_index INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT,
            started_at TEXT,
            ended_at TEXT
        )
    """)
    con.commit()
    return con


def _normalise_splits(
    task: dict[str, Any], *, base_seed: int, campaign_id: str, round_index: int,
    train_size: int, val_size: int, test_size: int,
) -> dict[str, Any]:
    generator = GENERATORS[str(task["generator"])]
    spec = dict(task)
    spec["seed"] = seed_from(base_seed, campaign_id, round_index, task["task_id"], "task_seed") % (2**31 - 1)
    raw: dict[str, list[tuple[list[float], float]]] = {}
    for split_name, count in (("train", train_size), ("val", val_size), ("test", test_size)):
        rng = random.Random(seed_from(base_seed, campaign_id, round_index, task["task_id"], split_name))
        factory = generator
        if split_name == "test" and bool(task.get("shifted_test")):
            factory = CHALLENGE_GENERATORS.get(str(task["generator"]), generator)
        examples = [factory(rng, spec, count) for _ in range(count)]
        raw[split_name] = [(list(ex.x), float(ex.y)) for ex in examples]
    n_features = int(task["n_features"])
    train_x = [row[0] for row in raw["train"]]
    means = [sum(row[j] for row in train_x) / len(train_x) for j in range(n_features)]
    scales = []
    for j in range(n_features):
        variance = sum((row[j] - means[j]) ** 2 for row in train_x) / max(1, len(train_x) - 1)
        scales.append(max(math.sqrt(variance), 1e-6))

    def transform(items: list[tuple[list[float], float]]) -> tuple[list[list[float]], list[float]]:
        xs: list[list[float]] = []
        ys: list[float] = []
        for values, label in items:
            xs.append([math.tanh((values[j] - means[j]) / (2.0 * scales[j])) for j in range(n_features)])
            ys.append(label)
        return xs, ys

    return {
        "train": transform(raw["train"]),
        "val": transform(raw["val"]),
        "test": transform(raw["test"]),
        "normalization": {"mean": means, "scale": scales},
        "test_distribution": "shifted_generator" if task.get("shifted_test") else "fresh_iid_seed",
    }


def _pad_features(xs: list[list[float]], n_qubits: int) -> list[list[float]]:
    if not xs:
        raise ValueError("empty feature matrix")
    if len(xs[0]) > n_qubits:
        raise ValueError("n_qubits must be at least the number of input features")
    return [row + [0.0] * (n_qubits - len(row)) for row in xs]


def _config_dict(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = {
        **raw,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_statevector_mb": args.max_statevector_mb,
    }
    return merged


def _mutated_configs(archive: list[dict[str, Any]], *, seed: int, max_configs: int) -> list[dict[str, Any]]:
    """Carry structures only, then mutate deterministically; prior scores are discarded."""
    configs = [dict(item) for item in BASE_VQC_CONFIGS]
    rng = random.Random(seed_from(seed, "quantum-config-mutation"))
    choices = {
        "n_qubits": [4, 6, 8],
        "n_layers": [1, 2, 3],
        "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
        "schedule": ["constant", "cosine"],
        "learning_rate": [0.005, 0.01, 0.02, 0.04],
        "feature_map": ["angle_ry", "angle_ry_rz"],
        "entanglement": ["linear", "ring"],
        "gradient_method": ["autograd", "parameter_shift"],
    }
    for item in archive[: max(0, max_configs - len(configs))]:
        candidate = {key: item[key] for key in choices if key in item}
        if not candidate:
            continue
        mutation_count = 1 if rng.random() < 0.7 else 2
        for key in rng.sample(list(choices), mutation_count):
            candidate[key] = rng.choice(choices[key])
        configs.append(candidate)
        if len(configs) >= max_configs:
            break
    # Deduplicate and bound. A deterministic fallback ensures the requested
    # configuration count even when archive entries collide with presets.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in configs:
        candidate = {
            "n_qubits": int(item.get("n_qubits", 4)),
            "n_layers": int(item.get("n_layers", 1)),
            "gradient_method": str(item.get("gradient_method", "autograd")),
            "optimizer": str(item.get("optimizer", "adam")),
            "schedule": str(item.get("schedule", "constant")),
            "learning_rate": float(item.get("learning_rate", 0.02)),
            "feature_map": str(item.get("feature_map", "angle_ry")),
            "entanglement": str(item.get("entanglement", "ring")),
        }
        key = hash_json(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) >= max_configs:
            break
    return unique


def _trial_key(plan_hash: str, round_index: int, task_id: str, method: str,
               config_data: dict[str, Any], seed: int) -> str:
    return hash_json({
        "plan_hash": plan_hash,
        "round": round_index,
        "task_id": task_id,
        "method": method,
        "config": config_data,
        "seed": seed,
    })


def _already_complete(con: sqlite3.Connection, key: str) -> bool:
    row = con.execute("SELECT status FROM model_trials WHERE trial_key=?", (key,)).fetchone()
    return bool(row and row["status"] == "ok")


def _record_model_trial(
    con: sqlite3.Connection,
    *, key: str, round_index: int, task: dict[str, Any], method: str,
    config_data: dict[str, Any], seed: int, runner: Any, metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    con.execute(
        """INSERT INTO model_trials(trial_key,round_index,task_id,family,method,config_json,seed,status,started_at)
           VALUES(?,?,?,?,?,?,?,'running',?)
           ON CONFLICT(trial_key) DO UPDATE SET status='running', error=NULL, started_at=excluded.started_at, ended_at=NULL""",
        (key, round_index, task["task_id"], task["family"], method, json.dumps(config_data, sort_keys=True), seed, now),
    )
    con.commit()
    try:
        result = runner()
        result.update({"task_id": task["task_id"], "family": task["family"], "trial_key": key})
        if metadata:
            result.update(metadata)
        con.execute(
            "UPDATE model_trials SET status='ok', result_json=?, ended_at=?, error=NULL WHERE trial_key=?",
            (json.dumps(result, sort_keys=True, allow_nan=False), utc_now(), key),
        )
        con.commit()
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        con.execute(
            "UPDATE model_trials SET status='failed', error=?, ended_at=? WHERE trial_key=?",
            (error, utc_now(), key),
        )
        con.commit()
        raise


def _round_results(con: sqlite3.Connection, round_index: int) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT result_json FROM model_trials WHERE round_index=? AND status='ok' ORDER BY task_id,method,config_json",
        (round_index,),
    ).fetchall()
    return [json.loads(row["result_json"]) for row in rows if row["result_json"]]


def _update_archive(results: list[dict[str, Any]], max_items: int = 6) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("method") != "vqc_statevector":
            continue
        config = result.get("config") or {}
        # Only validated architecture/optimizer fields are carried forward.
        keys = ("n_qubits", "n_layers", "gradient_method", "optimizer", "schedule", "learning_rate", "feature_map", "entanglement")
        structure = {key: config[key] for key in keys if key in config}
        key = hash_json(structure)
        grouped.setdefault(key, []).append(float(result.get("val_balanced_accuracy", 0.0)))
        configs[key] = structure
    ranked = sorted(grouped, key=lambda key: (sum(grouped[key]) / len(grouped[key]), key), reverse=True)
    # Old scores choose structures for exploration only; they are not carried
    # into a future round's evidence or promotion decisions.
    return [{**configs[key], "source_round_val_mean_provenance_only": sum(grouped[key]) / len(grouped[key])}
            for key in ranked[:max_items]]


def _write_reports(campaign_root: Path, con: sqlite3.Connection, state: dict[str, Any]) -> None:
    rows = con.execute(
        "SELECT round_index, task_id, family, method, config_json, seed, status, result_json, error FROM model_trials ORDER BY round_index, task_id, method"
    ).fetchall()
    csv_lines = ["round,task_id,family,test_distribution,method,device,val_balanced_accuracy,test_balanced_accuracy,elapsed_seconds,status,error"]
    for row in rows:
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        test = result.get("test", {})
        values = [row["round_index"], row["task_id"], row["family"], result.get("test_distribution", ""), row["method"], result.get("device", ""),
                  result.get("val_balanced_accuracy", ""), test.get("balanced_accuracy", ""),
                  result.get("elapsed_seconds", ""), row["status"], row["error"] or ""]
        csv_lines.append(",".join('"' + str(value).replace('"', '""') + '"' for value in values))
    atomic_write_text(campaign_root / "leaderboard.csv", "\n".join(csv_lines) + "\n")
    round_rows = con.execute("SELECT round_index,seed,status,summary_json,started_at,ended_at FROM rounds ORDER BY round_index").fetchall()
    lines = [
        f"# AMDE Quantum-Classical Research Campaign — {state['campaign_id']}",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Rounds completed: `{state.get('next_round', 0)}`",
        f"- Requested quantum simulator device: `{state.get('device')}`",
        f"- Base seed: `{state.get('base_seed')}`",
        f"- VQC config archive: `{len(state.get('config_archive', []))}` structures",
        "",
        "## Evidence boundaries",
        "",
        "- VQC values come from an ideal exact statevector simulation, not a physical QPU.",
        "- CPU/GPU simulation timings are not quantum runtime and do not establish quantum advantage.",
        "- The maze Grover result uses a table-backed phase oracle whose construction cost is measured separately.",
        "- BFS exploits graph structure; it is the relevant practical baseline for the included mazes.",
        "- Test distributions are generated locally and do not replace independent real-world datasets.",
        "- Every round uses new deterministic seeds; model configurations may transfer, old scores do not.",
        "",
        "## Model trial leaderboard",
        "",
        "| Round | Task | Method | Device | Val BAcc | Test BAcc | Seconds | Status |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        test = result.get("test", {})
        lines.append(
            f"| {row['round_index']} | `{row['task_id']}` | `{row['method']}` | `{result.get('device','')}` | "
            f"{result.get('val_balanced_accuracy','')} | {test.get('balanced_accuracy','')} | "
            f"{result.get('elapsed_seconds','')} | {row['status']} |"
        )
    lines.extend(["", "## Round summaries", "", "| Round | Seed | Status | Models OK | Maze |", "|---:|---:|---|---:|---|"])
    for row in round_rows:
        summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
        lines.append(f"| {row['round_index']} | {row['seed']} | {row['status']} | {summary.get('models_ok','')} | {summary.get('maze_status','')} |")
    atomic_write_text(campaign_root / "REPORT.md", "\n".join(lines) + "\n")


def run_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else ROOT
    campaign_id = sanitize_id(args.campaign_id, fallback="quantum_campaign")
    if len(campaign_id) > 48:
        raise ConfigError("campaign id must be 48 characters or fewer")
    if args.rounds < 0 or args.epochs < 1 or args.max_configs < 3 or args.max_qubits < 8:
        raise ConfigError("rounds must be >= 0, epochs >= 1, max-configs >= 3 and max-qubits >= 8")
    if args.max_campaign_gb <= 0 or args.min_free_gb < 1:
        raise ConfigError("storage limits must be positive and min-free-gb >= 1")
    if args.train_size < 16 or args.val_size < 8 or args.test_size < 8:
        raise ConfigError("dataset split sizes are too small")
    if args.device not in {"cpu", "cuda", "auto"}:
        raise ConfigError("device must be cpu, cuda, or auto")

    plan = {
        "protocol": QUANTUM_CAMPAIGN_PROTOCOL,
        "campaign_id": campaign_id,
        "base_seed": int(args.seed),
        "rounds": int(args.rounds),
        "device": args.device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "max_configs": int(args.max_configs),
        "max_qubits": int(args.max_qubits),
        "max_statevector_mb": int(args.max_statevector_mb),
        "max_campaign_gb": float(args.max_campaign_gb),
        "min_free_gb": float(args.min_free_gb),
        "tasks": list(TASKS),
        "base_vqc_configs": list(BASE_VQC_CONFIGS),
    }
    plan_hash = hash_json(plan)
    campaign_root = root / "reports" / "quantum_campaigns" / campaign_id
    _mkdir(campaign_root)
    state_path = campaign_root / "state.json"
    db_path = campaign_root / "quantum_ledger.sqlite3"
    lock = SingleInstanceLock(campaign_root / "campaign.lock", run_id=campaign_id, stale_seconds=300)
    lock.acquire()
    con = _connect(db_path)
    stop_path = Path(args.campaign_stop_file).resolve() if args.campaign_stop_file else campaign_root / "STOP"

    def stop_requested() -> bool:
        if stop_path.exists():
            return True
        return (not args.isolated_stop) and ((root / "runs" / "QUANTUM_STOP").exists() or (root / "runs" / "HYBRID_STOP").exists())

    try:
        if state_path.exists():
            state = read_json(state_path)
            if state.get("plan_hash") != plan_hash:
                raise ConfigError("existing quantum campaign has a different immutable plan; choose another campaign id")
        else:
            state = {**plan, "plan_hash": plan_hash, "created_at": utc_now(), "updated_at": utc_now(),
                     "status": "initialized", "next_round": 0, "config_archive": []}
            atomic_write_json(state_path, state)
        actual_device = resolve_device(args.device)
        torch = load_torch()
        torch.set_num_threads(1)
        if actual_device == "cuda":
            snapshot = _gpu_snapshot(torch)
            if float(snapshot.get("free_vram_mb", 0.0)) < 1536.0:
                raise QuantumBackendError(
                    f"only {snapshot.get('free_vram_mb', 0.0):.0f} MiB free VRAM; 1536 MiB reserve is required"
                )
        print(f"AMDE quantum campaign={campaign_id} rounds={'infinite' if args.rounds == 0 else args.rounds} device={actual_device}", flush=True)
        print(f"State: {state_path}", flush=True)

        while args.rounds == 0 or int(state["next_round"]) < args.rounds:
            if stop_requested():
                state["status"] = "stopped"
                break
            used = _tree_bytes(campaign_root)
            free = _disk_free_bytes(root)
            if used >= args.max_campaign_gb * 1024**3:
                state["status"] = "storage_cap_reached"
                break
            if free < args.min_free_gb * 1024**3:
                state["status"] = "low_disk_space"
                break

            round_index = int(state["next_round"])
            round_seed = seed_from(args.seed, campaign_id, round_index, "quantum-round") % (2**31 - 1)
            configs = _mutated_configs(state.get("config_archive", []), seed=round_seed, max_configs=args.max_configs)
            con.execute(
                "INSERT INTO rounds(round_index,seed,status,started_at) VALUES(?,?,?,?) "
                "ON CONFLICT(round_index) DO UPDATE SET status='running',started_at=excluded.started_at,ended_at=NULL",
                (round_index, round_seed, "running", utc_now()),
            )
            con.commit()
            state["status"] = "running"
            state["current_round"] = round_index
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            print(f"\n[quantum round {round_index}] seed={round_seed} configs={len(configs)}", flush=True)

            round_start = time.monotonic()
            round_model_keys: list[str] = []
            failures: list[str] = []
            for task in TASKS:
                if stop_requested():
                    state["status"] = "stop_requested"
                    break
                data = _normalise_splits(task, base_seed=args.seed, campaign_id=campaign_id, round_index=round_index,
                                         train_size=args.train_size, val_size=args.val_size, test_size=args.test_size)
                (train_x, train_y), (val_x, val_y), (test_x, test_y) = data["train"], data["val"], data["test"]
                mlp_cfg = {"hidden": 12, "epochs": args.epochs, "optimizer": "adam"}
                mlp_seed = seed_from(round_seed, task["task_id"], "mlp") % (2**31 - 1)
                mlp_key = _trial_key(plan_hash, round_index, task["task_id"], "mlp_cpu_baseline", mlp_cfg, mlp_seed)
                round_model_keys.append(mlp_key)
                if not _already_complete(con, mlp_key):
                    try:
                        _record_model_trial(con, key=mlp_key, round_index=round_index, task=task, method="mlp_cpu_baseline",
                                            config_data=mlp_cfg, seed=mlp_seed,
                                            runner=lambda tx=train_x, ty=train_y, vx=val_x, vy=val_y, dx=test_x, dy=test_y, ss=mlp_seed:
                                                train_mlp_baseline(tx, ty, vx, vy, dx, dy, epochs=args.epochs, seed=ss),
                                            metadata={"test_distribution": data["test_distribution"], "data_manifest_hash": hash_json(data)})
                    except Exception as exc:
                        failures.append(f"{task['task_id']}/mlp: {type(exc).__name__}: {exc}")
                rbf_cfg = {"gamma": 0.5, "regularization": [1e-3, 1e-2, 1e-1, 1.0]}
                rbf_key = _trial_key(plan_hash, round_index, task["task_id"], "rbf_kernel_ridge_cpu_baseline", rbf_cfg, mlp_seed)
                round_model_keys.append(rbf_key)
                if not _already_complete(con, rbf_key):
                    _write_heartbeat(state_path, state, round_index=round_index, task_id=str(task["task_id"]), method="rbf_kernel_ridge_cpu_baseline", device="cpu")
                    try:
                        _record_model_trial(
                            con, key=rbf_key, round_index=round_index, task=task, method="rbf_kernel_ridge_cpu_baseline",
                            config_data=rbf_cfg, seed=mlp_seed,
                            runner=lambda tx=train_x, ty=train_y, vx=val_x, vy=val_y, dx=test_x, dy=test_y:
                                train_rbf_kernel_baseline(tx, ty, vx, vy, dx, dy, gamma=0.5),
                            metadata={"test_distribution": data["test_distribution"], "data_manifest_hash": hash_json(data)},
                        )
                    except Exception as exc:
                        failures.append(f"{task['task_id']}/rbf: {type(exc).__name__}: {exc}")
                qkernel_cfg = {"n_qubits": min(8, args.max_qubits), "feature_map": "angle_ry_rz", "regularization": [1e-3, 1e-2, 1e-1, 1.0]}
                qkernel_key = _trial_key(plan_hash, round_index, task["task_id"], "quantum_fidelity_kernel_ridge", qkernel_cfg, mlp_seed)
                round_model_keys.append(qkernel_key)
                if not _already_complete(con, qkernel_key):
                    if actual_device == "cuda" and not _wait_for_gpu_budget(
                        root, torch, stop_path=stop_path, isolated_stop=args.isolated_stop,
                        heartbeat_callback=lambda: _write_heartbeat(state_path, state, round_index=round_index,
                                                                    task_id=str(task["task_id"]), method="quantum_gpu_wait",
                                                                    device=actual_device),
                    ):
                        state["status"] = "stop_requested"
                        break
                    _write_heartbeat(state_path, state, round_index=round_index, task_id=str(task["task_id"]), method="quantum_fidelity_kernel_ridge", device=actual_device)
                    try:
                        _record_model_trial(
                            con, key=qkernel_key, round_index=round_index, task=task, method="quantum_fidelity_kernel_ridge",
                            config_data=qkernel_cfg, seed=mlp_seed,
                            runner=lambda tx=train_x, ty=train_y, vx=val_x, vy=val_y, dx=test_x, dy=test_y, qc=qkernel_cfg, ss=mlp_seed:
                                train_quantum_kernel(tx, ty, vx, vy, dx, dy, n_qubits=int(qc["n_qubits"]),
                                                     device=actual_device, feature_map=str(qc["feature_map"]), seed=ss),
                            metadata={"test_distribution": data["test_distribution"], "data_manifest_hash": hash_json(data)},
                        )
                    except Exception as exc:
                        failures.append(f"{task['task_id']}/quantum-kernel: {type(exc).__name__}: {exc}")
                        if actual_device == "cuda" and ("CUDA" in str(exc) or "out of memory" in str(exc).lower()):
                            break
                for raw_cfg in configs:
                    if stop_requested():
                        state["status"] = "stop_requested"
                        break
                    cfg = _config_dict(raw_cfg, args)
                    cfg.setdefault("epochs", args.epochs)
                    VQCConfig(**cfg).validate()
                    qseed = seed_from(round_seed, task["task_id"], hash_json(raw_cfg), "vqc") % (2**31 - 1)
                    key = _trial_key(plan_hash, round_index, task["task_id"], "vqc_statevector", raw_cfg, qseed)
                    round_model_keys.append(key)
                    if _already_complete(con, key):
                        continue
                    if actual_device == "cuda" and not _wait_for_gpu_budget(
                        root, torch, stop_path=stop_path, isolated_stop=args.isolated_stop,
                        heartbeat_callback=lambda: _write_heartbeat(state_path, state, round_index=round_index,
                                                                    task_id=str(task["task_id"]), method="vqc_gpu_wait",
                                                                    device=actual_device),
                    ):
                        state["status"] = "stop_requested"
                        break
                    _write_heartbeat(state_path, state, round_index=round_index, task_id=str(task["task_id"]), method="vqc_statevector", device=actual_device)
                    padded_train = _pad_features(train_x, int(cfg["n_qubits"]))
                    padded_val = _pad_features(val_x, int(cfg["n_qubits"]))
                    padded_test = _pad_features(test_x, int(cfg["n_qubits"]))
                    try:
                        _record_model_trial(
                            con, key=key, round_index=round_index, task=task, method="vqc_statevector",
                            config_data=raw_cfg, seed=qseed,
                            runner=lambda tx=padded_train, ty=train_y, vx=padded_val, vy=val_y, dx=padded_test, dy=test_y, cc=VQCConfig(**cfg), ss=qseed:
                                train_vqc(tx, ty, vx, vy, dx, dy, config=cc, device=actual_device, seed=ss),
                            metadata={"test_distribution": data["test_distribution"], "data_manifest_hash": hash_json(data)},
                        )
                    except Exception as exc:
                        failures.append(f"{task['task_id']}/vqc/{hash_json(raw_cfg)[:8]}: {type(exc).__name__}: {exc}")
                        if isinstance(exc, QuantumBackendError) and actual_device == "cuda":
                            # A CUDA backend failure is recorded and causes a
                            # fail-closed round stop instead of silent CPU fallback.
                            break
                if failures and actual_device == "cuda" and any("CUDA" in item or "out of memory" in item.lower() for item in failures):
                    break
                if state.get("status") == "stop_requested":
                    break

            maze_status = "not_run"
            if state.get("status") != "stop_requested":
                maze_key = con.execute("SELECT status FROM maze_trials WHERE round_index=?", (round_index,)).fetchone()
                if not (maze_key and maze_key["status"] == "ok"):
                    try:
                        maze_result = run_maze_grover_demo(
                            device=actual_device,
                            max_qubits=min(args.max_qubits * 2, args.maze_max_qubits),
                            max_statevector_mb=args.max_statevector_mb,
                        )
                        con.execute(
                            "INSERT INTO maze_trials(round_index,seed,status,result_json,ended_at) VALUES(?,?,?,?,?) "
                            "ON CONFLICT(round_index) DO UPDATE SET status='ok',result_json=excluded.result_json,error=NULL,ended_at=excluded.ended_at",
                            (round_index, round_seed, "ok", json.dumps(maze_result, sort_keys=True, allow_nan=False), utc_now()),
                        )
                        con.commit()
                        maze_status = str(maze_result.get("status"))
                    except Exception as exc:
                        maze_status = "failed"
                        con.execute(
                            "INSERT INTO maze_trials(round_index,seed,status,error,ended_at) VALUES(?,?,?,?,?) "
                            "ON CONFLICT(round_index) DO UPDATE SET status='failed',error=excluded.error,ended_at=excluded.ended_at",
                            (round_index, round_seed, "failed", f"{type(exc).__name__}: {exc}", utc_now()),
                        )
                        con.commit()
                        failures.append(f"maze: {type(exc).__name__}: {exc}")
                else:
                    maze_status = str(maze_key["status"])

            results = _round_results(con, round_index)
            vqc_results = [item for item in results if item.get("method") == "vqc_statevector"]
            models_ok = len(results)
            if vqc_results:
                state["config_archive"] = _update_archive(vqc_results, max_items=6)
            summary = {
                "models_ok": models_ok,
                "vqc_trials_ok": len(vqc_results),
                "quantum_kernel_trials_ok": sum(1 for item in results if item.get("method") == "quantum_fidelity_kernel_ridge"),
                "rbf_baseline_trials_ok": sum(1 for item in results if item.get("method") == "rbf_kernel_ridge_cpu_baseline"),
                "unique_tasks": sorted({item.get("task_id") for item in results}),
                "mean_vqc_val_bacc": (sum(float(item["val_balanced_accuracy"]) for item in vqc_results) / len(vqc_results)) if vqc_results else None,
                "mean_vqc_test_bacc": (sum(float(item["test"]["balanced_accuracy"]) for item in vqc_results) / len(vqc_results)) if vqc_results else None,
                "mean_mlp_test_bacc": (sum(float(item["test"]["balanced_accuracy"]) for item in results if item.get("method") == "mlp_cpu_baseline") / max(1, sum(1 for item in results if item.get("method") == "mlp_cpu_baseline"))),
                "mean_rbf_test_bacc": (sum(float(item["test"]["balanced_accuracy"]) for item in results if item.get("method") == "rbf_kernel_ridge_cpu_baseline") / max(1, sum(1 for item in results if item.get("method") == "rbf_kernel_ridge_cpu_baseline"))),
                "mean_quantum_kernel_test_bacc": (sum(float(item["test"]["balanced_accuracy"]) for item in results if item.get("method") == "quantum_fidelity_kernel_ridge") / max(1, sum(1 for item in results if item.get("method") == "quantum_fidelity_kernel_ridge"))),
                "maze_status": maze_status,
                "failures": failures,
                "elapsed_seconds": time.monotonic() - round_start,
            }
            round_status = "completed" if not failures and len(results) >= len(TASKS) * (3 + len(configs)) else ("partial" if results else "failed")
            if state.get("status") == "stop_requested":
                round_status = "interrupted"
            con.execute(
                "UPDATE rounds SET status=?,summary_json=?,ended_at=? WHERE round_index=?",
                (round_status, json.dumps(summary, sort_keys=True, allow_nan=False), utc_now(), round_index),
            )
            con.commit()
            # The ledger is the quantum worker's durable trial-level checkpoint.
            # Force a passive WAL checkpoint at every round boundary; do not
            # discard uncommitted work or remove the database on shutdown.
            try:
                con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            except sqlite3.DatabaseError:
                pass
            _write_reports(campaign_root, con, state)
            state["last_round_summary"] = summary
            state["last_round_status"] = round_status
            state["updated_at"] = utc_now()
            if round_status == "completed":
                state["next_round"] = round_index + 1
                state["status"] = "running"
            elif round_status == "interrupted":
                state["status"] = "stopped" if stop_requested() else "stop_requested"
            else:
                state["status"] = round_status
            atomic_write_json(state_path, state)
            print(
                f"[quantum round {round_index}] {round_status}; models_ok={models_ok}; "
                f"VQC mean test BAcc={summary['mean_vqc_test_bacc']}; maze={maze_status}",
                flush=True,
            )
            if round_status != "completed":
                break
        if args.rounds > 0 and int(state.get("next_round", 0)) >= args.rounds and state.get("status") == "running":
            state["status"] = "round_limit_reached"
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)
        # Reports must describe the terminal state written above, not the
        # transient `running` state from immediately before the round limit.
        _write_reports(campaign_root, con, state)
        try:
            con.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        except sqlite3.DatabaseError:
            pass
        print(f"Quantum campaign state={state['status']} next_round={state.get('next_round')}", flush=True)
        return 0 if state.get("status") in {"running", "round_limit_reached", "stop_requested", "stopped", "storage_cap_reached", "low_disk_space"} else 1
    finally:
        con.close()
        lock.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--rounds", type=int, default=2, help="Number of independently seeded rounds; 0 continues until stopped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--val-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=96)
    parser.add_argument("--max-configs", type=int, default=5)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--max-statevector-mb", type=int, default=256)
    parser.add_argument("--maze-max-qubits", type=int, default=20)
    parser.add_argument("--max-campaign-gb", type=float, default=10.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--campaign-stop-file", default=None, help="Campaign-scoped cooperative stop marker")
    parser.add_argument("--isolated-stop", action="store_true", help="Ignore project-wide legacy stop markers")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_campaign(args)
    except KeyboardInterrupt:
        print("quantum campaign interrupted; completed trials remain committed", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"quantum campaign configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (DataIntegrityError, CheckpointError) as exc:
        print(f"quantum campaign data integrity failure: {exc}", file=sys.stderr)
        return EXIT_DATA_INTEGRITY
    except ResourcePressure as exc:
        print(f"quantum campaign resource exhaustion: {exc}", file=sys.stderr)
        return EXIT_RESOURCE_EXHAUSTION
    except (AMDEBaseError, OSError, ValueError, QuantumBackendError) as exc:
        print(f"quantum campaign runtime error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
