"""Configuration loading and validation."""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .util import CANONICAL_JSON_VERSION, deep_merge, hash_json, read_json, sanitize_id

# Bump these when semantics change. They are part of the baseline protocol
# hash, so changing a metric or generator invalidates old baseline comparisons.
PROTOCOL_GENERATOR_VERSION = "amde-data-protocol-v5-open-ended-search-generalization"
SEARCH_RNG_PROTOCOL_VERSION = "amde-search-rng-v4-open-ended-canonical-dsl"
METRIC_VERSION = "metric-v2-family-balanced"
CONFIG_SCHEMA_VERSION = "1.0.0"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _builtin_config_path() -> Path:
    return _default_project_root() / "config" / "default.json"


class Config:
    """Validated configuration with stable hashes and resolved paths."""

    def __init__(self, data: dict[str, Any], project_root: Path, source_path: Path | None = None) -> None:
        self.data = data
        self.project_root = project_root.resolve()
        self.source_path = source_path
        self.config_hash = hash_json(data)
        self.protocol_hash = hash_json(self._protocol_material())
        self._validate()

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self.data
        parts = dotted.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def refresh_hashes(self) -> None:
        self.config_hash = hash_json(self.data)
        self.protocol_hash = hash_json(self._protocol_material())

    def path(self, key: str) -> Path:
        rel = self.get(f"paths.{key}")
        if not isinstance(rel, str) or not rel:
            raise ConfigError(f"paths.{key} must be a non-empty string")
        result = (self.project_root / rel).resolve()
        try:
            result.relative_to(self.project_root)
        except ValueError as exc:
            raise ConfigError(f"paths.{key} escapes project root") from exc
        return result

    def run_dir(self, run_id: str) -> Path:
        return self.path("runs_dir") / sanitize_id(run_id)

    def artifact_dir(self) -> Path:
        return self.path("artifacts_dir")

    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    def data_dir(self) -> Path:
        return self.path("data_dir")

    def arc_path(self) -> Path:
        value = self.get("benchmarks.arc.path", "data/arc")
        result = (self.project_root / str(value)).resolve()
        try:
            result.relative_to(self.project_root)
        except ValueError as exc:
            raise ConfigError("benchmarks.arc.path escapes project root") from exc
        return result

    def task_specs(self) -> list[dict[str, Any]]:
        specs = self.get("benchmarks.task_specs")
        if not isinstance(specs, list):
            raise ConfigError("benchmarks.task_specs must be a list")
        return copy.deepcopy(specs)

    def dev_task_specs(self) -> list[dict[str, Any]]:
        """Return task specifications eligible for open development/search.

        Tasks with ``holdout_only: true`` are reserved exclusively for the
        sealed challenge evaluation boundary so that development search never
        evaluates candidates on held-out task families.
        """
        return [
            spec
            for spec in self.task_specs()
            if not bool(spec.get("holdout_only", False))
        ]

    def challenge_task_specs(self) -> list[dict[str, Any]]:
        """Return held-out task specifications for the sealed challenge.

        Legacy/custom configs without an explicit holdout remain supported by
        falling back to their full task set, but the shipped default config
        defines a dedicated held-out family set.
        """
        specs = self.task_specs()
        held_out = [spec for spec in specs if bool(spec.get("holdout_only", False))]
        return held_out if held_out else specs

    def quick_task_ids(self) -> list[str]:
        ids = self.get("benchmarks.quick_task_ids", [])
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ConfigError("benchmarks.quick_task_ids must be a list of strings")
        return list(ids)

    def eval_seeds(self, count: int | None = None) -> list[int]:
        seeds = self.get("evaluation.seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("evaluation.seeds must be a non-empty list")
        seeds = [int(s) for s in seeds]
        if count is not None:
            if count < 1 or count > len(seeds):
                raise ConfigError("requested seed count is outside evaluation.seeds")
            seeds = seeds[:count]
        return seeds

    def effective_copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def _protocol_material(self) -> dict[str, Any]:
        return {
            "generator_protocol": PROTOCOL_GENERATOR_VERSION,
            "canonical_json_version": CANONICAL_JSON_VERSION,
            "search_rng_protocol": SEARCH_RNG_PROTOCOL_VERSION,
            "metric_version": self.get("evaluation.metric_version", METRIC_VERSION),
            "dataset_version": self.get("benchmarks.dataset_version"),
            "task_specs": self.get("benchmarks.task_specs"),
            "split": self.get("benchmarks.split"),
            "eval_seeds": self.get("evaluation.seeds"),
            "confirm_seed_count": self.get("evaluation.confirm.seed_count"),
            "arc": {
                "enabled": self.get("benchmarks.arc.enabled"),
                "mode": self.get("benchmarks.arc.mode"),
            },
            # Effective device affects numerical behavior and cache validity.
            "training_device": self.get("runtime.effective_device", self.get("runtime.device", "auto")),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        required = [
            "schema_version",
            "engine_version",
            "seed",
            "paths",
            "benchmarks",
            "evaluation",
            "search",
            "budgets",
            "runtime",
            "safety",
        ]
        missing = [key for key in required if key not in self.data]
        if missing:
            raise ConfigError(f"missing config sections: {missing}")
        if self.data["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported config schema {self.data['schema_version']!r}; "
                f"expected {CONFIG_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.data["seed"], int) or isinstance(self.data["seed"], bool):
            raise ConfigError("seed must be an integer")

        specs = self.data["benchmarks"].get("task_specs")
        if not isinstance(specs, list) or not specs:
            raise ConfigError("benchmarks.task_specs must be a non-empty list")
        ids: set[str] = set()
        dev_ids: set[str] = set()
        allowed_generators = {
            "linear_regression",
            "quadratic_regression",
            "xor_classification",
            "blobs_classification",
            "noisy_linear_classification",
            "circles_classification",
            "imbalanced_classification",
            "regime_switch_regression",
            "shifted_xor_classification",
            "sine_regression",
            "friedman_regression",
            "sparse_linear_regression",
            "heteroscedastic_regression",
            "moons_classification",
            "parity_classification",
            "spiral_classification",
            "checkerboard_classification",
            "sparse_parity_classification",
        }
        for spec in specs:
            if not isinstance(spec, dict):
                raise ConfigError("every task spec must be an object")
            for key in ("task_id", "suite", "family", "kind", "generator", "n_features", "seed"):
                if key not in spec:
                    raise ConfigError(f"task spec missing {key}: {spec!r}")
            task_id = str(spec["task_id"])
            if task_id in ids:
                raise ConfigError(f"duplicate task_id: {task_id}")
            ids.add(task_id)
            if not bool(spec.get("holdout_only", False)):
                dev_ids.add(task_id)
            if spec["kind"] not in ("classification", "regression"):
                raise ConfigError(f"unsupported task kind for {task_id}")
            if spec["generator"] not in allowed_generators:
                raise ConfigError(f"unsupported generator for {task_id}: {spec['generator']}")
            if not isinstance(spec["n_features"], int) or not 1 <= spec["n_features"] <= 64:
                raise ConfigError(f"n_features out of range for {task_id}")
            minimum_features = {
                "friedman_regression": 5,
                "parity_classification": 4,
                "checkerboard_classification": 2,
                "sparse_parity_classification": 8,
            }.get(str(spec["generator"]), 1)
            if int(spec["n_features"]) < minimum_features:
                raise ConfigError(
                    f"{spec['generator']} for {task_id} requires n_features >= {minimum_features}"
                )
        if not dev_ids:
            raise ConfigError("at least one task in benchmarks.task_specs must be open for development")
        quick = self.data["benchmarks"].get("quick_task_ids")
        if not isinstance(quick, list) or len(quick) < 2:
            raise ConfigError("benchmarks.quick_task_ids must contain at least two tasks")
        unknown = sorted(set(quick) - ids)
        if unknown:
            raise ConfigError(f"quick_task_ids not in task_specs: {unknown}")
        holdout_in_quick = sorted(set(quick) - dev_ids)
        if holdout_in_quick:
            raise ConfigError(f"quick_task_ids cannot reference holdout_only tasks: {holdout_in_quick}")

        split = self.data["benchmarks"].get("split", {})
        for key in ("train", "val", "dev", "challenge"):
            if not isinstance(split.get(key), int) or split[key] < 8:
                raise ConfigError(f"benchmarks.split.{key} must be an integer >= 8")

        seeds = self.data["evaluation"].get("seeds")
        if not isinstance(seeds, list) or len(seeds) < 3:
            raise ConfigError("evaluation.seeds needs at least three fixed seeds")
        if len(set(seeds)) != len(seeds):
            raise ConfigError("evaluation.seeds must be unique")
        confirm_count = self.data["evaluation"].get("confirm", {}).get("seed_count", 3)
        if not isinstance(confirm_count, int) or not 2 <= confirm_count <= len(seeds):
            raise ConfigError("evaluation.confirm.seed_count must be between 2 and len(seeds)")

        levels = self.data["search"].get("levels")
        if not isinstance(levels, list) or not levels or not set(levels) <= {"A", "B", "C", "D"}:
            raise ConfigError("search.levels must be a non-empty subset of A-D")
        if int(self.data["search"].get("proposals_per_generation", 0)) < 1:
            raise ConfigError("search.proposals_per_generation must be positive")
        if int(self.data["search"].get("confirm_per_generation", 0)) < 1:
            raise ConfigError("search.confirm_per_generation must be positive")
        if int(self.data["search"].get("islands", 0)) < 1:
            raise ConfigError("search.islands must be positive")
        if int(self.data["search"].get("population_size", 0)) < 2:
            raise ConfigError("search.population_size must be >= 2")
        if int(self.data["search"].get("stagnation_patience_generations", 10)) < 1:
            raise ConfigError("search.stagnation_patience_generations must be >= 1")
        allowed_strategies = {"random", "evolutionary", "crossover", "novelty", "local", "open_ended"}
        strategies = self.data["search"].get("strategies", [])
        if not isinstance(strategies, list) or not strategies or any(item not in allowed_strategies for item in strategies):
            raise ConfigError("search.strategies must be a non-empty list of supported proposal strategies")
        if len(set(strategies)) != len(strategies):
            raise ConfigError("search.strategies must not contain duplicates")

        max_disk = self.data["budgets"].get("max_disk_gb")
        if not isinstance(max_disk, (int, float)) or max_disk <= 0:
            raise ConfigError("budgets.max_disk_gb must be positive")
        metric_version = self.data["evaluation"].get("metric_version")
        if metric_version != METRIC_VERSION:
            raise ConfigError(
                f"this engine implements {METRIC_VERSION!r}, config requested {metric_version!r}"
            )

        safety = self.data["safety"]
        if not (8 <= int(safety.get("max_dsl_nodes", 0)) <= 512):
            raise ConfigError("safety.max_dsl_nodes must be between 8 and 512")
        if not (3 <= int(safety.get("max_dsl_depth", 0)) <= 64):
            raise ConfigError("safety.max_dsl_depth must be between 3 and 64")
        if not (1 <= int(safety.get("max_hidden_layers", 0)) <= 4):
            raise ConfigError("safety.max_hidden_layers must be between 1 and 4")
        if not (1 <= int(safety.get("max_hidden_width", 0)) <= 128):
            raise ConfigError("safety.max_hidden_width must be between 1 and 128")


def load_config(
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    root = Path(project_root) if project_root is not None else _default_project_root()
    root = root.resolve()
    path = Path(config_path) if config_path is not None else (root / "config" / "default.json")
    if not path.is_absolute():
        path = (root / path).resolve()
    try:
        base = read_json(path)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON config {path}: {exc}") from exc
    if not isinstance(base, dict):
        raise ConfigError("config root must be an object")
    data = deep_merge(base, overrides or {})
    cfg = Config(data, root, path)
    for key in ("runs_dir", "artifacts_dir", "reports_dir", "data_dir"):
        cfg.path(key)
    return cfg


@dataclass(frozen=True)
class ResourcePlan:
    cpu_workers: int
    device_requested: str
    device_effective: str
    reason: str
    reserve_ram_mb: int
    per_worker_memory_mb: int
    hard_cap_mb: int
    max_disk_gb: float
    gpu_slots: int = 0
    gpu_effective: bool = False


@dataclass
class ResourcePolicy:
    data: dict[str, Any]
    source_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, project_root: Path, filename: str = "config/resource_policy.json") -> "ResourcePolicy":
        path = (project_root / filename).resolve()
        try:
            data = read_json(path)
        except FileNotFoundError as exc:
            raise ConfigError(f"resource policy not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid resource policy JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("resource policy root must be an object")
        policy = cls(data=data, source_path=path)
        policy._validate()
        return policy

    def _validate(self) -> None:
        try:
            cpu = self.data["cpu"]
            mem = self.data["memory"]
            gpu = self.data["gpu"]
            disk = self.data["disk"]
            assert int(cpu["min_workers"]) >= 1
            assert int(cpu["max_workers"]) >= int(cpu["min_workers"])
            assert int(cpu["reserve_logical_cores"]) >= 0
            assert int(cpu["threads_per_trial"]) >= 1
            assert float(mem["reserve_mb"]) >= 0
            assert float(mem["per_worker_estimate_mb"]) > 0
            assert float(mem["hard_cap_mb_per_worker"]) > 0
            assert float(gpu["reserve_vram_mb"]) >= 0
            assert float(gpu["required_free_vram_mb"]) >= 0
            assert int(gpu["max_concurrent_trials"]) >= 0
            assert float(disk["min_free_gb"]) >= 0
            assert float(disk["max_run_gb"]) > 0
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            raise ConfigError(f"invalid resource policy: {exc}") from exc

    def effective_ram_reserve_mb(self, total_ram_mb: float | None) -> int:
        """Return a conservative memory reserve scaled to the machine size.

        The configured reserve is a ceiling, not a reason to deadlock a run on
        a smaller machine/container. On known hardware it is capped by
        ``max_reserve_fraction`` of physical RAM, but never below the explicit
        minimum reserve. Unknown hardware keeps the configured reserve.
        """
        memory = self.data["memory"]
        minimum = max(128, int(memory.get("minimum_reserve_mb", 512)))
        configured = max(minimum, int(memory["reserve_mb"]))
        try:
            total = float(total_ram_mb or 0.0)
        except (TypeError, ValueError):
            total = 0.0
        if total <= 0:
            return configured
        fraction = float(memory.get("max_reserve_fraction", 0.25))
        fraction = min(0.80, max(0.05, fraction))
        dynamic_cap = max(minimum, int(total * fraction))
        return max(minimum, min(configured, dynamic_cap))

    def compute_plan(self, hardware: Any, requested_device: str = "auto") -> ResourcePlan:
        """Compute a conservative CPU/GPU plan from probed hardware.

        CUDA is only selected when the optional tensor backend passed an actual
        tensor smoke test and the configured VRAM/thermal reserve is satisfied.
        Merely seeing a GPU in ``nvidia-smi`` is not evidence that training can
        run on it.
        """

        cpu = self.data["cpu"]
        mem = self.data["memory"]
        disk = self.data["disk"]
        gpu = self.data["gpu"]
        logical = max(1, int(getattr(hardware, "logical_cores", 1) or 1))
        available_mb = float(getattr(hardware, "ram_available_mb", 0.0) or 0.0)
        free_disk_gb = float(getattr(hardware, "disk_free_gb", 0.0) or 0.0)

        usable_cpu = max(1, logical - int(cpu["reserve_logical_cores"]))
        workers_by_cpu = max(1, usable_cpu // max(1, int(cpu["threads_per_trial"])))
        workers = min(int(cpu["max_workers"]), workers_by_cpu)

        total_mb = float(getattr(hardware, "ram_total_mb", 0.0) or 0.0)
        reserve = self.effective_ram_reserve_mb(total_mb)
        per_worker = int(mem["per_worker_estimate_mb"])
        if available_mb > 0:
            budget_for_workers = max(0.0, available_mb - reserve)
            workers_by_ram = int(budget_for_workers // max(1.0, float(per_worker)))
            if workers_by_ram < 1:
                workers = 1
                self.warnings.append(
                    "available RAM is below the configured reserve plus one worker; "
                    "running one trial with a strict memory cap"
                )
            else:
                workers = min(workers, workers_by_ram)

        if free_disk_gb > 0 and free_disk_gb < float(disk["min_free_gb"]):
            workers = 1
            self.warnings.append(
                f"free disk {free_disk_gb:.1f} GB is below reserve; engine will pause before long runs"
            )

        gpu_available = bool(getattr(hardware, "gpu_available", False))
        gpu_free_mb = float(getattr(hardware, "gpu_memory_free_mb", 0.0) or 0.0)
        gpu_temp = getattr(hardware, "gpu_temperature_c", None)
        gpu_needed = float(gpu["reserve_vram_mb"]) + float(gpu["required_free_vram_mb"])
        torch_info = getattr(hardware, "torch", {}) or {}
        smoke = torch_info.get("tensor_smoke", {}) if isinstance(torch_info, dict) else {}
        tensor_backend_ready = bool(
            torch_info.get("import_ok")
            and torch_info.get("cuda_available")
            and isinstance(smoke, dict)
            and smoke.get("ok")
        )
        thermal_ok = not isinstance(gpu_temp, (int, float)) or gpu_temp < float(gpu.get("thermal_warn_c", 80))
        gpu_reserve_ok = (
            bool(gpu["enabled"])
            and gpu_available
            and tensor_backend_ready
            and gpu_free_mb >= gpu_needed
            and thermal_ok
            and int(gpu["max_concurrent_trials"]) >= 1
        )
        gpu_slots = 1 if gpu_reserve_ok else 0
        gpu_effective = bool(gpu_reserve_ok and requested_device != "cpu")
        if requested_device == "cpu":
            device_effective = "cpu"
            reason = "CPU explicitly requested"
            gpu_effective = False
            gpu_slots = 0
        elif gpu_effective:
            device_effective = "cuda"
            reason = "CUDA tensor smoke test passed; GPU VRAM and thermal reserves are satisfied"
            # Each isolated worker creates its own CUDA context. One concurrent
            # GPU trial is the safe default on 4 GB GPUs; CPU search remains on
            # the main process and dispatches work sequentially to the GPU.
            workers = 1
        else:
            device_effective = "cpu"
            if requested_device == "cuda":
                if not gpu_available:
                    reason = "CUDA requested but no NVIDIA GPU was detected; CPU fallback"
                elif not tensor_backend_ready:
                    reason = "CUDA requested but the optional PyTorch CUDA tensor smoke test failed; CPU fallback"
                elif gpu_free_mb < gpu_needed:
                    reason = f"CUDA requested but free VRAM {gpu_free_mb:.0f} MB is below reserve {gpu_needed:.0f} MB; CPU fallback"
                elif not thermal_ok:
                    reason = f"CUDA requested but GPU temperature {gpu_temp}C is above warning threshold; CPU fallback"
                else:
                    reason = "CUDA requested but GPU policy disallows a concurrent trial; CPU fallback"
            else:
                reason = "auto selected CPU because no compatible CUDA backend passed the smoke test or resource reserve"
            if gpu_available:
                self.warnings.append(reason)
        workers = max(int(cpu["min_workers"]), workers)
        return ResourcePlan(
            cpu_workers=workers,
            device_requested=requested_device,
            device_effective=device_effective,
            reason=reason,
            reserve_ram_mb=reserve,
            per_worker_memory_mb=per_worker,
            hard_cap_mb=int(mem["hard_cap_mb_per_worker"]),
            max_disk_gb=float(disk["max_run_gb"]),
            gpu_slots=gpu_slots,
            gpu_effective=gpu_effective,
        )

    def min_free_disk_bytes(self) -> int:
        return int(float(self.data["disk"]["min_free_gb"]) * (1024**3))

    def hard_memory_cap_bytes(self) -> int:
        return int(float(self.data["memory"]["hard_cap_mb_per_worker"]) * (1024**2))

    def available_disk_gb(self, path: Path | None = None) -> float:
        target = Path(path or ".").resolve()
        while not target.exists() and target != target.parent:
            target = target.parent
        usage = shutil.disk_usage(target)
        return usage.free / (1024**3)


def apply_cli_overrides(config: Config, *, seed: int | None, device: str | None, stop_file: str | None = None) -> Config:
    if seed is not None:
        config.set("seed", int(seed))
    if device is not None:
        if device not in ("auto", "cpu", "cuda"):
            raise ConfigError("device must be auto, cpu, or cuda")
        config.set("runtime.device", device)
    if stop_file:
        config.set("paths.stop_file_name", stop_file)
    config.refresh_hashes()
    return config


def config_from_env() -> Config:
    root = os.environ.get("AMDE_ROOT")
    cfg_path = os.environ.get("AMDE_CONFIG")
    return load_config(project_root=Path(root) if root else None, config_path=cfg_path)
