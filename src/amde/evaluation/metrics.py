"""Versioned metrics and aggregation.

The candidate never chooses or implements the metric. Metric code lives in
this module and its version/hash is written to every trial, baseline, and
decision row.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from ..config import METRIC_VERSION
from ..errors import CandidateNumericError
from ..util import hash_json, mean, stdev

METRIC_SPEC = {
    "version": METRIC_VERSION,
    "classification": "balanced_accuracy over classes {0,1}",
    "regression": "clipped R^2 relative to the dev mean predictor",
    "aggregate": "equal-weight mean across families; tasks are averaged within family; seed CI over family-balanced seed aggregates",
    "robustness": "reports worst-family and lower-quartile-family score; no missing results are zero-filled",
    "orientation": "higher_is_better",
    "missing_result": "candidate invalid / infrastructure error; never zero-filled",
}
METRIC_HASH = hash_json(METRIC_SPEC)

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerically stable continued fraction for the incomplete beta function."""
    max_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")

def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b

def student_t_critical(level: float, degrees_of_freedom: int) -> float:
    """Two-sided Student-t critical value, computed without SciPy."""
    if not 0.0 < float(level) < 1.0:
        raise ValueError("confidence level must be strictly between 0 and 1")
    if degrees_of_freedom < 1:
        raise ValueError("degrees of freedom must be positive")
    target = 1.0 - float(level)
    a = degrees_of_freedom / 2.0
    b = 0.5
    low, high = 0.0, 1.0
    for _ in range(90):
        midpoint = (low + high) / 2.0
        if _regularized_beta(midpoint, a, b) < target:
            low = midpoint
        else:
            high = midpoint
    x = (low + high) / 2.0
    return math.sqrt(degrees_of_freedom * (1.0 - x) / x)


def _finite(values: Iterable[float], name: str) -> list[float]:
    result = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise CandidateNumericError(f"non-finite {name} in predictions")
        result.append(number)
    return result


def classification_score(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    trues = _finite(y_true, "labels")
    preds = _finite(y_pred, "predictions")
    if len(trues) != len(preds) or not trues:
        raise ValueError("classification metric received empty or mismatched arrays")
    labels = [1 if y >= 0.5 else 0 for y in trues]
    guesses = [1 if p >= 0.5 else 0 for p in preds]
    correct = sum(1 for a, b in zip(labels, guesses) if a == b)
    recalls = []
    for klass in (0, 1):
        indices = [idx for idx, label in enumerate(labels) if label == klass]
        if indices:
            recalls.append(sum(1 for idx in indices if guesses[idx] == klass) / len(indices))
    balanced = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "score": float(balanced),
        "accuracy": correct / len(labels),
        "balanced_accuracy": float(balanced),
        "n": len(labels),
    }


def regression_score(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    trues = _finite(y_true, "targets")
    preds = _finite(y_pred, "predictions")
    if len(trues) != len(preds) or not trues:
        raise ValueError("regression metric received empty or mismatched arrays")
    mu = mean(trues)
    sse = sum((a - b) ** 2 for a, b in zip(trues, preds))
    sst = sum((a - mu) ** 2 for a in trues)
    if sst <= 1e-15:
        r2 = 1.0 if sse <= 1e-15 else 0.0
    else:
        r2 = 1.0 - sse / sst
    clipped = max(0.0, min(1.0, r2))
    return {
        "score": float(clipped),
        "r2": float(r2),
        "mse": float(sse / len(trues)),
        "n": len(trues),
    }


def score_task(kind: str, y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    if kind == "classification":
        return classification_score(y_true, y_pred)
    if kind == "regression":
        return regression_score(y_true, y_pred)
    raise ValueError(f"unknown task kind: {kind}")


def ci95(values: list[float], level: float = 0.95) -> tuple[float, float]:
    """Two-sided Student-t interval over independent seed-level aggregates.

    This interval quantifies seed variability on the configured development
    cohort. It is not a confidence interval over unseen task families or a
    substitute for an external benchmark.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    critical = student_t_critical(level, len(values) - 1)
    mu = mean(values)
    se = stdev(values) / math.sqrt(len(values))
    return (mu - critical * se, mu + critical * se)


def aggregate_scores(
    records: list[dict[str, Any]],
    *,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate scores hierarchically so each task family has equal weight.

    Multiple benchmark variants within a family are averaged first, then
    families are averaged with equal weight. This prevents a family with more
    configured tasks from dominating the objective. Seed intervals are based
    on the same family-balanced aggregation. Failed/missing results must never
    be passed as zero-valued records.
    """
    if not records:
        raise ValueError("no score records to aggregate")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task_id = str(record["task_id"])
        score = float(record["score"])
        if not math.isfinite(score):
            raise CandidateNumericError("non-finite aggregate score")
        by_task.setdefault(task_id, []).append(record)

    per_task: dict[str, dict[str, Any]] = {}
    for task_id, task_records in sorted(by_task.items()):
        scores = [float(item["score"]) for item in task_records]
        seeds = [int(item["seed"]) for item in task_records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed records for task {task_id}")
        per_task[task_id] = {
            "family": str(task_records[0]["family"]),
            "suite": str(task_records[0].get("suite", "")),
            "kind": str(task_records[0].get("kind", "")),
            "mean": mean(scores),
            "std": stdev(scores),
            "n_seeds": len(scores),
            "scores": {int(item["seed"]): float(item["score"]) for item in task_records},
        }

    # Every task must have the same seed set; otherwise a family/seed aggregate
    # could be biased by missing observations.
    expected_seeds = None
    for task_id, stats in per_task.items():
        task_seeds = set(stats["scores"])
        if expected_seeds is None:
            expected_seeds = task_seeds
        elif task_seeds != expected_seeds:
            raise ValueError(f"inconsistent seed coverage for task {task_id}")

    family_tasks: dict[str, list[str]] = {}
    for task_id, stats in per_task.items():
        family_tasks.setdefault(stats["family"], []).append(task_id)
    family_scores = {
        family: mean(per_task[task_id]["mean"] for task_id in task_ids)
        for family, task_ids in sorted(family_tasks.items())
    }
    family_values = list(family_scores.values())
    family_order = sorted(family_values)
    lower_quartile_count = max(1, int(math.ceil(len(family_order) * 0.25)))
    lower_quartile = mean(family_order[:lower_quartile_count])
    worst_family = min(family_scores, key=lambda key: family_scores[key])

    seed_aggregates: dict[str, float] = {}
    for seed in sorted(expected_seeds or set()):
        seed_family_scores = []
        for family, task_ids in sorted(family_tasks.items()):
            seed_family_scores.append(
                mean(per_task[task_id]["scores"][seed] for task_id in task_ids)
            )
        seed_aggregates[str(seed)] = mean(seed_family_scores)
    seed_values = list(seed_aggregates.values())
    low, high = ci95(seed_values, level=ci_level)
    task_means = [stats["mean"] for stats in per_task.values()]

    return {
        "mean": mean(family_values),
        "median": sorted(family_values)[len(family_values) // 2],
        "std_tasks": stdev(task_means),
        "std_families": stdev(family_values),
        "lower_quartile_family_score": lower_quartile,
        "std_seeds": stdev(seed_values),
        "ci_low": low,
        "ci_high": high,
        "ci_level": ci_level,
        "worst_family": worst_family,
        "worst_family_score": family_scores[worst_family],
        "family_scores": family_scores,
        "per_task": per_task,
        "seed_aggregates": seed_aggregates,
        "n_tasks": len(per_task),
        "n_families": len(family_scores),
        "n_seeds": len(seed_aggregates),
        "n_records": len(records),
        "metric_version": METRIC_VERSION,
        "metric_hash": METRIC_HASH,
        "records": records,
    }


def compare_aggregates(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Compare candidate and baseline using identical task/seed coverage."""

    candidate_tasks = set(candidate.get("per_task", {}))
    baseline_tasks = set(baseline.get("per_task", {}))
    if candidate_tasks != baseline_tasks:
        raise ValueError(
            f"task coverage mismatch: candidate={sorted(candidate_tasks)}, baseline={sorted(baseline_tasks)}"
        )
    if candidate.get("metric_hash") != baseline.get("metric_hash"):
        raise ValueError("metric versions differ")
    candidate_seeds = set(int(x) for x in candidate.get("seed_aggregates", {}))
    baseline_seeds = set(int(x) for x in baseline.get("seed_aggregates", {}))
    if candidate_seeds != baseline_seeds:
        raise ValueError("seed coverage differs between candidate and baseline")
    differences = {
        seed: float(candidate["seed_aggregates"][str(seed)]) - float(baseline["seed_aggregates"][str(seed)])
        for seed in sorted(candidate_seeds)
    }
    family_deltas = {
        family: float(candidate["family_scores"].get(family, 0.0)) - float(baseline["family_scores"].get(family, 0.0))
        for family in sorted(candidate.get("family_scores", {}))
    }
    families_won = sum(1 for delta in family_deltas.values() if delta > 0)
    seed_wins = sum(1 for delta in differences.values() if delta > 0)
    low, high = ci95(list(differences.values()), level=float(candidate.get("ci_level", 0.95)))
    return {
        "effect": float(candidate["mean"]) - float(baseline["mean"]),
        "family_deltas": family_deltas,
        "families_won": families_won,
        "seed_deltas": {str(key): value for key, value in differences.items()},
        "seed_wins": seed_wins,
        "ci_low": low,
        "ci_high": high,
        "worst_family_delta": float(candidate["worst_family_score"]) - float(baseline["worst_family_score"]),
        "candidate_worst_family": candidate["worst_family"],
        "baseline_worst_family": baseline["worst_family"],
    }
