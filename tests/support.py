"""Shared helpers for the AMDE standard-library test suite."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from amde.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "resource_policy.json"

SMALL_TASKS = [
    {
        "task_id": "tab_linear",
        "suite": "classical",
        "family": "linear",
        "kind": "regression",
        "generator": "linear_regression",
        "n_features": 3,
        "seed": 101,
    },
    {
        "task_id": "tab_xor",
        "suite": "classical",
        "family": "xor",
        "kind": "classification",
        "generator": "xor_classification",
        "n_features": 2,
        "seed": 103,
    },
]


def _read_default() -> dict[str, Any]:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def small_config_data(
    *,
    challenge: bool = False,
    proposals: int = 2,
    confirm: int = 1,
    seeds: list[int] | None = None,
    quick_timeout: float = 15.0,
    confirm_timeout: float = 20.0,
) -> dict[str, Any]:
    data = copy.deepcopy(_read_default())
    # Test fixtures must be independent of optional GPU/PyTorch installations.
    # CUDA capability is tested separately by resource-policy unit tests and doctor.
    data["runtime"]["device"] = "cpu"
    data["benchmarks"]["task_specs"] = copy.deepcopy(SMALL_TASKS)
    data["benchmarks"]["quick_task_ids"] = [task["task_id"] for task in SMALL_TASKS]
    data["benchmarks"]["split"] = {
        "train": 8,
        "val": 8,
        "dev": 8,
        "challenge": 8,
        "duplicate_rounding": 8,
    }
    data["benchmarks"]["arc"] = {"enabled": False, "mode": "baseline_only", "path": "data/arc"}
    data["evaluation"]["seeds"] = list(seeds or [1, 2, 3])
    data["evaluation"]["confirm"]["seed_count"] = min(2, len(data["evaluation"]["seeds"]))
    data["evaluation"]["quick"]["seed_count"] = 1
    data["evaluation"]["quick"]["timeout_seconds"] = float(quick_timeout)
    data["evaluation"]["confirm"]["timeout_seconds"] = float(confirm_timeout)
    data["evaluation"]["quick"]["max_workers"] = 2
    data["evaluation"]["confirm"]["max_workers"] = 2
    data["evaluation"]["quick"]["epochs_scale"] = 0.15
    data["evaluation"]["max_steps"] = 300
    data["search"]["levels"] = ["A", "B", "C", "D"]
    data["search"]["open_level_after_generation"] = {"A": 0, "B": 0, "C": 0, "D": 0}
    data["search"]["proposals_per_generation"] = int(proposals)
    data["search"]["confirm_per_generation"] = int(confirm)
    data["search"]["islands"] = 1
    data["search"]["population_size"] = 4
    data["search"]["elite_size"] = 2
    data["search"]["meta_policy_enabled"] = False
    data["challenge"] = {
        "enabled": bool(challenge),
        "max_openings_per_run": 1,
        "min_generations_between_openings": 1,
        "seeds": [11, 12],
    }
    data["budgets"] = {
        "default_generations": 1,
        "smoke_max_generations": 1,
        "smoke_max_seconds": 60,
        "max_trials": 0,
        "max_disk_gb": 1.0,
        "max_artifact_bytes": 1048576,
    }
    data["runtime"]["heartbeat_seconds"] = 0.1
    data["runtime"]["lock_stale_seconds"] = 30
    data["runtime"]["max_transient_retries"] = 0
    data["runtime"]["retry_backoff_seconds"] = 0.0
    data["logging"]["level"] = "WARNING"
    return data


def write_project(
    root: Path,
    *,
    challenge: bool = False,
    config_data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    root = Path(root)
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_POLICY_PATH, root / "config" / "resource_policy.json")
    data = copy.deepcopy(config_data) if config_data is not None else small_config_data(challenge=challenge, **kwargs)
    (root / "config" / "default.json").write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return root


def make_project(
    case: unittest.TestCase,
    *,
    challenge: bool = False,
    config_data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Path, Config]:
    holder = tempfile.TemporaryDirectory(prefix="amde_test_")
    case.addCleanup(holder.cleanup)
    root = Path(holder.name)
    write_project(root, challenge=challenge, config_data=config_data, **kwargs)
    return root, load_config(project_root=root)


def make_config(case: unittest.TestCase, **kwargs: Any) -> Config:
    return make_project(case, **kwargs)[1]


def fake_hardware(
    *,
    logical_cores: int = 8,
    physical_cores: int = 4,
    ram_available_mb: float = 16384.0,
    ram_total_mb: float = 32768.0,
    disk_free_gb: float = 100.0,
    gpu_available: bool = False,
    gpu_memory_free_mb: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        ram_available_mb=ram_available_mb,
        ram_total_mb=ram_total_mb,
        disk_free_gb=disk_free_gb,
        disk_total_gb=1000.0,
        gpu_available=gpu_available,
        gpu_memory_free_mb=gpu_memory_free_mb,
        gpu_memory_total_mb=(gpu_memory_free_mb or 0.0) + 1024,
        gpu_temperature_c=55.0,
        gpu_name="fake-gpu" if gpu_available else "unavailable",
    )


def task_specs(config: Config) -> list[dict[str, Any]]:
    return config.task_specs()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk_files(root: Path) -> list[Path]:
    return sorted(path for path in Path(root).rglob("*") if path.is_file())
