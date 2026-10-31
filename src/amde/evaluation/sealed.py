"""Sealed challenge evaluation boundary.

The challenge split is generated inside a spawned worker. Only aggregate
scores for the candidate and the chosen baseline cross the process boundary.
Per-task scores, per-seed scores, and challenge examples are deliberately
discarded before returning.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import Config, METRIC_VERSION
from ..discovery.factory import safety_kwargs
from ..discovery.genome import Genome
from ..errors import ConfigError
from .isolation import WorkerJob, run_jobs
from .metrics import METRIC_HASH


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is non-finite")
    return number


_ALLOWED_SUMMARY = {
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
}


def _validate_summary(summary: Any, *, expected_tasks: int, expected_seeds: int) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("sealed summary is not an object")
    extra = set(summary) - _ALLOWED_SUMMARY
    if extra:
        raise ValueError(f"sealed summary has non-aggregate fields: {sorted(extra)}")
    missing = _ALLOWED_SUMMARY - set(summary)
    if missing:
        raise ValueError(f"sealed summary is missing fields: {sorted(missing)}")
    if summary.get("metric_hash") != METRIC_HASH:
        raise ValueError("sealed metric hash mismatch")
    if summary.get("metric_version") != METRIC_VERSION:
        raise ValueError("sealed metric version mismatch")
    if int(summary.get("n_tasks", -1)) != expected_tasks:
        raise ValueError("sealed n_tasks mismatch")
    if int(summary.get("n_families", -1)) != len(summary.get("family_scores", {})):
        raise ValueError("sealed n_families mismatch")
    if int(summary.get("n_seeds", -1)) != expected_seeds:
        raise ValueError("sealed n_seeds mismatch")
    for key in ("mean", "median", "std_tasks", "std_families", "lower_quartile_family_score", "std_seeds", "ci_low", "ci_high", "worst_family_score"):
        value = _finite(summary.get(key), key)
        if key in ("mean", "median", "worst_family_score", "lower_quartile_family_score") and not 0.0 <= value <= 1.0:
            raise ValueError(f"sealed {key} outside [0,1]")
        if key in ("std_tasks", "std_families", "std_seeds") and value < 0.0:
            raise ValueError(f"sealed {key} cannot be negative")
    if float(summary["ci_low"]) > float(summary["ci_high"]):
        raise ValueError("sealed confidence interval is inverted")
    families = summary.get("family_scores")
    if not isinstance(families, dict) or not families:
        raise ValueError("sealed family scores are missing")
    for name, value in families.items():
        _finite(value, f"family {name}")
    if summary.get("worst_family") not in families:
        raise ValueError("sealed worst_family is not in family_scores")
    return summary


class SealedEvaluator:
    def __init__(self, config: Config, *, memory_cap_mb: int | None = None, device: str = "cpu") -> None:
        self.config = config
        self.memory_cap_mb = int(memory_cap_mb or 0)
        self.device = str(device)
        self.split_config = dict(config.get("benchmarks.split"))
        self.safety = safety_kwargs(config)
        self.max_steps = int(config.get("evaluation.max_steps", 1200))
        self.parameter_limit = int(config.get("evaluation.parameter_limit", 20000))
        self.ci_level = float(config.get("evaluation.promotion.ci_level", 0.8))

    def payload(
        self,
        *,
        candidate: Genome,
        baseline_name: str,
        baseline_genome: Genome | None,
        task_specs: list[dict[str, Any]],
        seeds: list[int],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if baseline_genome is None and baseline_name != "majority_mean":
            raise ConfigError("a trained challenge baseline requires a genome")
        return {
            "kind": "sealed",
            "include_challenge": True,
            "genome": candidate.to_dict(),
            "baseline": baseline_name,
            "baseline_genome": baseline_genome.to_dict() if baseline_genome is not None else None,
            "task_specs": [dict(spec) for spec in task_specs],
            "split_config": self.split_config,
            "seeds": [int(seed) for seed in seeds],
            "epochs_scale": 1.0,
            "safety": self.safety,
            "max_steps": self.max_steps,
            "parameter_limit": self.parameter_limit,
            "ci_level": self.ci_level,
            "device": self.device,
            "timeout_seconds": float(timeout_seconds),
        }

    def open_opening(
        self,
        *,
        candidate: Genome,
        baseline_name: str,
        baseline_genome: Genome | None,
        task_specs: list[dict[str, Any]],
        seeds: list[int],
        opening_index: int,
        cohort: str,
        stop_callback: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        timeout_seconds = max(60.0, 2.0 * float(self.config.get("evaluation.confirm.timeout_seconds", 40)))
        payload = self.payload(
            candidate=candidate,
            baseline_name=baseline_name,
            baseline_genome=baseline_genome,
            task_specs=task_specs,
            seeds=seeds,
            timeout_seconds=timeout_seconds,
        )
        raw = run_jobs(
            [
                WorkerJob(
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                    candidate=True,
                    label=f"sealed-{opening_index}",
                )
            ],
            max_workers=1,
            memory_cap_mb=self.memory_cap_mb,
            stop_callback=stop_callback,
        )[0]
        result = {
            "status": "error",
            "candidate": None,
            "baseline": None,
            "error_type": raw.get("error_type"),
            "error_code": raw.get("error_code"),
            "message": raw.get("message", ""),
            "elapsed_seconds": raw.get("elapsed_seconds"),
            "peak_rss_mb": raw.get("peak_rss_mb"),
            "exit_code": raw.get("exit_code"),
            "memory_cap_enforced": bool(raw.get("memory_cap_enforced")),
            "opening_index": int(opening_index),
            "cohort": cohort,
            "seeds": [int(seed) for seed in seeds],
            "baseline_name": baseline_name,
        }
        if raw.get("status") != "ok":
            status = str(raw.get("status", "error"))
            if status == "timeout":
                result["status"] = "timeout"
            elif status == "memory":
                result["status"] = "memory"
            elif status == "interrupted":
                result["status"] = "interrupted"
            elif status == "error" and raw.get("error_type") == "candidate":
                result["status"] = "invalid"
            else:
                result["status"] = "error"
            return result
        try:
            body = raw.get("result") or {}
            candidate_summary = _validate_summary(
                body.get("candidate"),
                expected_tasks=len(task_specs),
                expected_seeds=len(seeds),
            )
            baseline_summary = _validate_summary(
                body.get("baseline"),
                expected_tasks=len(task_specs),
                expected_seeds=len(seeds),
            )
            if set(body) - {"candidate", "baseline", "n_tasks", "n_seeds", "challenge_used"}:
                raise ValueError("sealed worker returned unexpected fields")
            if not body.get("challenge_used"):
                raise ValueError("sealed worker did not report challenge use")
        except Exception as exc:
            result.update(
                status="protocol_error",
                error_type="infrastructure",
                error_code="invalid_sealed_result",
                message=f"sealed result failed validation: {exc}",
            )
            return result
        result.update(
            status="ok",
            candidate=candidate_summary,
            baseline=baseline_summary,
            error_type=None,
            error_code=None,
            message="",
            delta=float(candidate_summary["mean"]) - float(baseline_summary["mean"]),
        )
        return result
