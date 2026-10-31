"""Protocol-bound, process-isolated evaluation of open splits.

The evaluator never receives challenge data. It builds a request containing
only train/val/dev task specifications, executes it in a spawned worker, and
validates the returned aggregate against the expected task/seed coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, METRIC_VERSION
from ..discovery.factory import safety_kwargs
from ..discovery.genome import Genome
from ..errors import ConfigError, ValidationError
from ..util import hash_json
from .isolation import WorkerJob, run_jobs
from .metrics import METRIC_HASH, aggregate_scores


def evaluation_cache_key(
    *,
    protocol_hash: str,
    manifest_hash: str | None,
    stage: str,
    identity: str,
    task_ids: list[str],
    seeds: list[int],
    epochs_scale: float,
    metric_hash: str,
    dataset_version: str,
    engine_version: str,
    device: str = "cpu",
) -> str:
    return hash_json(
        {
            "protocol_hash": protocol_hash,
            "manifest_hash": manifest_hash,
            "stage": stage,
            "identity": identity,
            "task_ids": sorted(task_ids),
            "seeds": sorted(int(seed) for seed in seeds),
            "epochs_scale": round(float(epochs_scale), 8),
            "metric_hash": metric_hash,
            "dataset_version": dataset_version,
            "engine_version": engine_version,
            "training_device": str(device),
        }
    )


@dataclass
class EvalRequest:
    stage: str
    task_specs: list[dict[str, Any]]
    seeds: list[int]
    epochs_scale: float
    timeout_seconds: float
    cache_key: str
    genome: Genome | None = None
    baseline: str | None = None
    manifest_hash: str | None = None
    protocol_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    @property
    def task_ids(self) -> list[str]:
        return [str(spec["task_id"]) for spec in self.task_specs]

    @property
    def identity(self) -> str:
        if self.baseline and self.genome is None:
            return f"baseline:{self.baseline}"
        return self.genome.hash if self.genome is not None else "missing"


class Evaluator:
    def __init__(
        self,
        config: Config,
        *,
        stop_callback: Any | None = None,
        memory_cap_mb: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.stop_callback = stop_callback
        self.memory_cap_mb = int(memory_cap_mb or 0)
        self.device = str(device)
        self.split_config = dict(config.get("benchmarks.split"))
        self.safety = safety_kwargs(config)
        self.max_steps = int(config.get("evaluation.max_steps", 1200))
        self.parameter_limit = int(config.get("evaluation.parameter_limit", 20000))
        self.ci_level = float(config.get("evaluation.promotion.ci_level", 0.8))
        self.engine_version = str(config.get("engine_version"))
        self.dataset_version = str(config.get("benchmarks.dataset_version"))
        self.metric_hash = METRIC_HASH

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------
    def build_request(
        self,
        *,
        genome: Genome | None,
        baseline: str | None,
        stage: str,
        task_specs: list[dict[str, Any]],
        seeds: list[int],
        epochs_scale: float,
        timeout_seconds: float,
        manifest_hash: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalRequest:
        if not task_specs:
            raise ConfigError("evaluation request needs at least one task")
        if genome is None and baseline != "majority_mean":
            raise ConfigError("only the majority baseline may omit a genome")
        if genome is not None and baseline is not None:
            raise ConfigError("a request cannot be both a candidate and a baseline")
        task_ids = [str(spec["task_id"]) for spec in task_specs]
        identity = genome.hash if genome is not None else f"baseline:{baseline}"
        cache_key = evaluation_cache_key(
            protocol_hash=self.config.protocol_hash,
            manifest_hash=manifest_hash,
            stage=stage,
            identity=identity,
            task_ids=task_ids,
            seeds=seeds,
            epochs_scale=epochs_scale,
            metric_hash=self.metric_hash,
            dataset_version=self.dataset_version,
            engine_version=self.engine_version,
            device=self.device,
        )
        return EvalRequest(
            stage=stage,
            task_specs=[dict(spec) for spec in task_specs],
            seeds=[int(seed) for seed in seeds],
            epochs_scale=float(epochs_scale),
            timeout_seconds=float(timeout_seconds),
            cache_key=cache_key,
            genome=genome,
            baseline=baseline,
            manifest_hash=manifest_hash,
            protocol_hash=self.config.protocol_hash,
            metadata=dict(metadata or {}),
            device=self.device,
        )

    def payload(self, request: EvalRequest) -> dict[str, Any]:
        return {
            "kind": "dev",
            "include_challenge": False,
            "genome": request.genome.to_dict() if request.genome is not None else None,
            "baseline": request.baseline,
            "task_specs": request.task_specs,
            "split_config": self.split_config,
            "seeds": request.seeds,
            "epochs_scale": request.epochs_scale,
            "safety": self.safety,
            "max_steps": self.max_steps,
            "parameter_limit": self.parameter_limit,
            "ci_level": self.ci_level,
            "device": request.device,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def evaluate_many(self, requests: list[EvalRequest], *, max_workers: int = 1) -> list[dict[str, Any]]:
        jobs = [
            WorkerJob(
                payload=self.payload(request),
                timeout_seconds=request.timeout_seconds,
                candidate=request.genome is not None,
                label=f"{request.stage}-{request.identity[:12]}",
            )
            for request in requests
        ]
        raw_results = run_jobs(
            jobs,
            max_workers=max_workers,
            memory_cap_mb=self.memory_cap_mb,
            stop_callback=self.stop_callback,
        )
        return [self._normalize(raw, request) for raw, request in zip(raw_results, requests)]

    def evaluate(self, request: EvalRequest) -> dict[str, Any]:
        return self.evaluate_many([request], max_workers=1)[0]

    def _normalize(self, raw: dict[str, Any], request: EvalRequest) -> dict[str, Any]:
        result = {
            "status": "error",
            "aggregate": None,
            "error_type": raw.get("error_type"),
            "error_code": raw.get("error_code"),
            "message": raw.get("message", ""),
            "elapsed_seconds": raw.get("elapsed_seconds"),
            "peak_rss_mb": raw.get("peak_rss_mb"),
            "exit_code": raw.get("exit_code"),
            "worker_pid": raw.get("worker_pid"),
            "memory_cap_enforced": bool(raw.get("memory_cap_enforced")),
            "cache_key": request.cache_key,
            "cache_hit": False,
            "stage": request.stage,
            "task_ids": request.task_ids,
            "seeds": request.seeds,
        }
        status = str(raw.get("status", "error"))
        if status == "ok":
            try:
                aggregate = raw.get("result", {}).get("aggregate")
                _validate_dev_aggregate(aggregate, request)
            except Exception as exc:
                result.update(
                    status="protocol_error",
                    error_type="infrastructure",
                    error_code="invalid_worker_aggregate",
                    message=f"worker aggregate failed validation: {exc}",
                )
                return result
            result.update(status="ok", aggregate=aggregate, error_type=None, error_code=None, message="")
            return result
        if status == "error":
            if raw.get("error_type") == "candidate":
                result["status"] = "invalid"
            else:
                result["status"] = "error"
            return result
        if status == "timeout":
            result["status"] = "timeout"
            return result
        if status == "memory":
            result["status"] = "memory"
            return result
        if status == "interrupted":
            result["status"] = "interrupted"
            return result
        result["status"] = "error"
        result["error_type"] = result.get("error_type") or "infrastructure"
        result["error_code"] = result.get("error_code") or "worker_failure"
        return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is non-finite")
    return number


def _validate_dev_aggregate(aggregate: Any, request: EvalRequest) -> None:
    if not isinstance(aggregate, dict):
        raise ValueError("aggregate is not an object")
    if aggregate.get("metric_hash") != METRIC_HASH:
        raise ValueError("metric hash mismatch")
    if aggregate.get("metric_version") != METRIC_VERSION:
        raise ValueError("metric version mismatch")
    expected_tasks = set(request.task_ids)
    per_task = aggregate.get("per_task")
    if not isinstance(per_task, dict) or set(per_task) != expected_tasks:
        raise ValueError("per-task coverage does not match the request")
    expected_seeds = {int(seed) for seed in request.seeds}
    for task_id, stats in per_task.items():
        if not isinstance(stats, dict):
            raise ValueError(f"per-task stats for {task_id} are not an object")
        scores = stats.get("scores")
        if not isinstance(scores, dict) or {int(key) for key in scores} != expected_seeds:
            raise ValueError(f"seed coverage mismatch for {task_id}")
        for key, value in scores.items():
            score = _finite_number(value, f"{task_id}/{key} score")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score outside [0,1] for {task_id}/{key}")
        mean = _finite_number(stats.get("mean"), f"{task_id} mean")
        if not 0.0 <= mean <= 1.0:
            raise ValueError(f"task mean outside [0,1] for {task_id}")
    seed_aggregates = aggregate.get("seed_aggregates")
    if not isinstance(seed_aggregates, dict) or {int(key) for key in seed_aggregates} != expected_seeds:
        raise ValueError("seed aggregate coverage mismatch")
    for key, value in seed_aggregates.items():
        _finite_number(value, f"seed {key} aggregate")
    for key in ("mean", "median", "std_tasks", "std_families", "lower_quartile_family_score", "std_seeds", "ci_low", "ci_high", "worst_family_score"):
        _finite_number(aggregate.get(key), key)
    if int(aggregate.get("n_tasks", -1)) != len(expected_tasks):
        raise ValueError("n_tasks mismatch")
    if int(aggregate.get("n_families", -1)) != len(aggregate.get("family_scores", {})):
        raise ValueError("n_families mismatch")
    if int(aggregate.get("n_seeds", -1)) != len(expected_seeds):
        raise ValueError("n_seeds mismatch")
    records = aggregate.get("records")
    if not isinstance(records, list) or len(records) != len(expected_tasks) * len(expected_seeds):
        raise ValueError("record count mismatch")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record is not an object")
        if set(record) & {"x", "y", "input", "output", "challenge", "examples"}:
            raise ValueError("record contains example data")
        if str(record.get("task_id")) not in expected_tasks:
            raise ValueError("record has an unexpected task")
        if int(record.get("seed")) not in expected_seeds:
            raise ValueError("record has an unexpected seed")
    if "challenge" in aggregate:
        raise ValueError("dev aggregate contains a challenge field")


def aggregate_summary(aggregate: dict[str, Any] | None) -> dict[str, Any] | None:
    """Small aggregate copy for checkpoints and status output."""

    if aggregate is None:
        return None
    allowed = (
        "mean",
        "median",
        "std_tasks",
        "std_families",
        "lower_quartile_family_score",
        "std_seeds",
        "ci_low",
        "ci_high",
        "ci_level",
        "worst_family",
        "worst_family_score",
        "family_scores",
        "n_tasks",
        "n_families",
        "n_seeds",
        "metric_version",
        "metric_hash",
    )
    return {key: aggregate[key] for key in allowed if key in aggregate}
