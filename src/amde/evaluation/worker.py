"""Worker-side evaluation. Imported only in spawned children.

No network, filesystem writes, or challenge data are exposed to the search
process. The dev worker generates only train/val/dev splits. The sealed worker
generates the challenge split, evaluates it, and returns only aggregate
numbers.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..benchmarks.base import generate_task
from ..benchmarks.generators import CHALLENGE_GENERATORS, GENERATORS
from ..discovery.genome import Genome
from ..errors import (
    CandidateMemoryError,
    CandidateNumericError,
    CandidateTimeoutError,
    InfrastructureError,
    ValidationError,
)
from ..evaluation.metrics import aggregate_scores, score_task
from ..models.trainer import baseline_majority_mean, train_and_predict


def _safety_kwargs(safety: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_dsl_nodes": int(safety.get("max_dsl_nodes", 96)),
        "max_dsl_depth": int(safety.get("max_dsl_depth", 12)),
        "max_genome_bytes": int(safety.get("max_genome_bytes", 65536)),
        "max_hidden_layers": int(safety.get("max_hidden_layers", 2)),
        "max_hidden_width": int(safety.get("max_hidden_width", 24)),
    }


def _make_tasks(payload: dict[str, Any], *, include_challenge: bool):
    tasks = []
    for spec in payload["task_specs"]:
        tasks.append(
            generate_task(
                spec,
                split_config=payload["split_config"],
                generators=GENERATORS,
                challenge_generators=CHALLENGE_GENERATORS,
                include_challenge=include_challenge,
            )
        )
    if not tasks:
        raise ValidationError("evaluation request contained no tasks", code="empty_tasks")
    return tasks


def _run_one(
    *,
    genome: Genome | None,
    baseline: str | None,
    task: Any,
    split_name: str,
    seed: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    examples = task.split(split_name)
    x_train = [list(example.x) for example in task.train]
    y_train = [float(example.y) for example in task.train]
    x_val = [list(example.x) for example in task.val]
    y_val = [float(example.y) for example in task.val]
    x_eval = [list(example.x) for example in examples]
    y_eval = [float(example.y) for example in examples]
    started = time.perf_counter()
    if baseline == "majority_mean":
        if genome is not None:
            raise ValidationError("majority baseline must not carry a genome", code="baseline_payload")
        predictions = baseline_majority_mean(
            train_x=x_train,
            train_y=y_train,
            dev_x=x_eval,
            task_kind=task.kind,
        )
    else:
        if genome is None:
            raise ValidationError("trained baseline is missing its genome", code="baseline_payload")
        output = train_and_predict(
            genome,
            train_x=x_train,
            train_y=y_train,
            val_x=x_val,
            val_y=y_val,
            dev_x=x_eval,
            task_kind=task.kind,
            eval_seed=int(seed),
            task_id=task.task_id,
            epochs_scale=float(payload["epochs_scale"]),
            max_steps=int(payload["max_steps"]),
            parameter_limit=int(payload["parameter_limit"]),
            device=str(payload.get("device", "cpu")),
        )
        predictions = output["predictions"]
    metric = score_task(task.kind, y_eval, predictions)
    score = float(metric["score"])
    if not math.isfinite(score):
        raise CandidateNumericError("non-finite metric score")
    return {
        "task_id": task.task_id,
        "family": task.family,
        "suite": task.suite,
        "kind": task.kind,
        "seed": int(seed),
        "score": score,
        "elapsed_seconds": time.perf_counter() - started,
    }


def dev_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a candidate or baseline on open (train/val/dev) splits."""

    if payload.get("include_challenge"):
        raise InfrastructureError("dev worker was asked for a challenge split")
    genome = None
    if payload.get("genome") is not None:
        genome = Genome(payload["genome"], **_safety_kwargs(payload["safety"]))
    baseline = payload.get("baseline")
    if genome is None and baseline != "majority_mean":
        raise ValidationError("a candidate or trained baseline is required", code="missing_candidate")
    tasks = _make_tasks(payload, include_challenge=False)
    records = []
    for task in tasks:
        for seed in payload["seeds"]:
            records.append(
                _run_one(
                    genome=genome,
                    baseline=baseline,
                    task=task,
                    split_name="dev",
                    seed=int(seed),
                    payload=payload,
                )
            )
    aggregate = aggregate_scores(records, ci_level=float(payload["ci_level"]))
    return {
        "aggregate": aggregate,
        "n_tasks": len(tasks),
        "n_seeds": len(payload["seeds"]),
        "challenge_used": False,
    }


def _summary(aggregate: dict[str, Any]) -> dict[str, Any]:
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
    summary = {key: aggregate[key] for key in allowed if key in aggregate}
    for key in ("mean", "median", "std_tasks", "std_families", "lower_quartile_family_score", "std_seeds", "ci_low", "ci_high", "worst_family_score"):
        if key in summary and not math.isfinite(float(summary[key])):
            raise CandidateNumericError(f"non-finite sealed aggregate field {key}")
    return summary


def sealed_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one opening on sealed challenge splits.

    Only aggregate summaries are returned. Challenge examples, per-task
    scores, and per-seed scores remain inside this worker.
    """

    if not payload.get("include_challenge"):
        raise InfrastructureError("sealed worker requires the challenge cohort")
    candidate = None
    if payload.get("genome") is not None:
        candidate = Genome(payload["genome"], **_safety_kwargs(payload["safety"]))
    baseline_genome = None
    if payload.get("baseline_genome") is not None:
        baseline_genome = Genome(payload["baseline_genome"], **_safety_kwargs(payload["safety"]))
    baseline_name = payload.get("baseline")
    if candidate is None:
        raise ValidationError("sealed opening requires a candidate", code="missing_candidate")
    if baseline_genome is None and baseline_name != "majority_mean":
        raise ValidationError("sealed opening requires a valid baseline", code="missing_baseline")

    tasks = _make_tasks(payload, include_challenge=True)
    candidate_records = []
    baseline_records = []
    for task in tasks:
        if task.challenge is None:
            raise InfrastructureError(f"challenge split missing for {task.task_id}")
        for seed in payload["seeds"]:
            candidate_records.append(
                _run_one(
                    genome=candidate,
                    baseline=None,
                    task=task,
                    split_name="challenge",
                    seed=int(seed),
                    payload=payload,
                )
            )
            baseline_records.append(
                _run_one(
                    genome=baseline_genome,
                    baseline=baseline_name,
                    task=task,
                    split_name="challenge",
                    seed=int(seed),
                    payload=payload,
                )
            )
    candidate_aggregate = aggregate_scores(
        candidate_records, ci_level=float(payload["ci_level"])
    )
    baseline_aggregate = aggregate_scores(
        baseline_records, ci_level=float(payload["ci_level"])
    )
    return {
        "candidate": _summary(candidate_aggregate),
        "baseline": _summary(baseline_aggregate),
        "n_tasks": len(tasks),
        "n_seeds": len(payload["seeds"]),
        "challenge_used": True,
    }


def child_entry(payload: dict[str, Any], connection: Any) -> None:
    """Spawn target. Classifies failures before sending one JSON-safe result."""

    try:
        kind = payload.get("kind")
        if kind == "dev":
            result = dev_worker(payload)
        elif kind == "sealed":
            result = sealed_worker(payload)
        else:
            raise InfrastructureError(f"unknown worker kind: {kind!r}")
        connection.send({"status": "ok", "result": result})
    except ValidationError as exc:
        connection.send(
            {
                "status": "error",
                "error_type": "candidate",
                "error_code": getattr(exc, "code", "validation_error"),
                "message": str(exc)[:4000],
            }
        )
    except CandidateNumericError as exc:
        connection.send(
            {
                "status": "error",
                "error_type": "candidate",
                "error_code": "candidate_numeric",
                "message": str(exc)[:4000],
            }
        )
    except CandidateMemoryError as exc:
        connection.send(
            {
                "status": "memory",
                "error_type": "candidate",
                "error_code": "candidate_memory",
                "message": str(exc)[:4000],
            }
        )
    except CandidateTimeoutError as exc:
        connection.send(
            {
                "status": "timeout",
                "error_type": "candidate",
                "error_code": "candidate_timeout",
                "message": str(exc)[:4000],
            }
        )
    except Exception as exc:  # infrastructure must not look like a low score
        connection.send(
            {
                "status": "error",
                "error_type": "infrastructure",
                "error_code": type(exc).__name__,
                "message": str(exc)[:4000],
            }
        )
    finally:
        try:
            connection.close()
        except Exception:
            pass
