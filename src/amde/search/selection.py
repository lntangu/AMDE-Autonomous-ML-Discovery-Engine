"""Promotion guard separating a real improvement from noise."""

from __future__ import annotations

from typing import Any

from ..evaluation.metrics import compare_aggregates


def choose_baseline(baselines: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [
        item
        for item in baselines
        if item.get("status") == "ok" and isinstance(item.get("aggregate"), dict)
    ]
    if not usable:
        return None
    return max(usable, key=lambda item: float(item["aggregate"].get("mean", -1.0)))


def evaluate_promotion(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    config: dict[str, Any],
    candidate_hash: str,
    baseline_name: str,
) -> dict[str, Any]:
    """Return a conservative promote/reject decision with full evidence."""

    min_effect = float(config.get("min_effect", 0.01))
    min_seeds = int(config.get("min_seeds", 3))
    min_families = int(config.get("min_families_won", 2))
    max_worst_drop = float(config.get("max_worst_family_drop", 0.03))
    require_non_regression = bool(config.get("require_non_regression", True))

    try:
        comparison = compare_aggregates(candidate, baseline)
    except Exception as exc:
        return {
            "decision": "reject",
            "reason": f"evidence cannot be compared: {exc}",
            "evidence": {
                "candidate_hash": candidate_hash,
                "baseline_name": baseline_name,
                "error": str(exc),
            },
        }

    reasons = []
    if int(candidate.get("n_seeds", 0)) < min_seeds:
        reasons.append(f"only {candidate.get('n_seeds')} seeds; need {min_seeds}")
    if comparison["effect"] < min_effect:
        reasons.append(f"effect {comparison['effect']:.4f} below min_effect {min_effect:.4f}")
    if comparison["families_won"] < min_families:
        reasons.append(f"won {comparison['families_won']} families; need {min_families}")
    if comparison["worst_family_delta"] < -max_worst_drop:
        reasons.append(
            f"worst-family delta {comparison['worst_family_delta']:.4f} below -{max_worst_drop:.4f}"
        )
    if require_non_regression and comparison["effect"] < 0:
        reasons.append("mean score regressed against the baseline")
    if comparison["ci_low"] < -max_worst_drop:
        reasons.append(
            f"lower confidence bound {comparison['ci_low']:.4f} is below -{max_worst_drop:.4f}"
        )
    decision = "promote" if not reasons else "reject"
    return {
        "decision": decision,
        "reason": "all promotion guard conditions passed" if not reasons else "; ".join(reasons),
        "evidence": {
            "candidate_hash": candidate_hash,
            "baseline_name": baseline_name,
            "candidate_mean": float(candidate.get("mean", 0.0)),
            "baseline_mean": float(baseline.get("mean", 0.0)),
            "candidate_n_seeds": int(candidate.get("n_seeds", 0)),
            "baseline_n_seeds": int(baseline.get("n_seeds", 0)),
            "thresholds": {
                "min_effect": min_effect,
                "min_seeds": min_seeds,
                "min_families_won": min_families,
                "max_worst_family_drop": max_worst_drop,
                "require_non_regression": require_non_regression,
            },
            "comparison": comparison,
        },
    }
