"""Group-level statistical audit for scope-limited generalization claims.

The unit of resampling is the independent benchmark family, not individual
examples or repeated seeds. This avoids pretending correlated examples are
independent samples. The module deliberately emits scoped evidence tiers and
never labels any finite benchmark result as proof of universal generalization.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Mapping


def _finite_map(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty family->score mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{label}[{key!r}] is not finite")
        if not str(key).strip():
            raise ValueError(f"{label} contains an empty family identifier")
        result[str(key)] = score
    return result


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate quantile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _bootstrap_mean_ci(
    differences: list[float], *, repetitions: int, rng: random.Random
) -> tuple[float, float]:
    n = len(differences)
    estimates = []
    for _ in range(max(1000, int(repetitions))):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _sign_flip_test(
    differences: list[float], *, repetitions: int, rng: random.Random
) -> dict[str, Any]:
    n = len(differences)
    observed = abs(sum(differences) / n)
    tolerance = 1e-15
    if n <= 16:
        extreme = 0
        total = 1 << n
        for mask in range(total):
            signed = [value if (mask >> index) & 1 else -value for index, value in enumerate(differences)]
            statistic = abs(sum(signed) / n)
            if statistic + tolerance >= observed:
                extreme += 1
        return {
            "method": "exact_paired_sign_flip",
            "p_value": extreme / total,
            "permutations": total,
        }
    extreme = 0
    total = max(10000, int(repetitions))
    for _ in range(total):
        statistic = abs(sum(value if rng.random() < 0.5 else -value for value in differences) / n)
        if statistic + tolerance >= observed:
            extreme += 1
    # Plus-one correction prevents zero Monte Carlo p-values.
    return {
        "method": "monte_carlo_paired_sign_flip",
        "p_value": (extreme + 1) / (total + 1),
        "permutations": total,
    }


def audit_generalization(
    candidate_family_scores: Mapping[str, Any],
    baseline_family_scores: Mapping[str, Any],
    *,
    family_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    preregistered: bool = False,
    evaluation_used_for_selection: bool = True,
    minimum_families: int = 10,
    minimum_external_families: int = 10,
    minimum_domains: int = 3,
    minimum_effect: float = 0.0,
    minimum_win_rate: float = 0.65,
    maximum_family_regression: float = 0.10,
    bootstrap_repetitions: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare a candidate and fixed baseline on paired unseen families.

    Scores must already be aggregated per independent family on an evaluation
    set that was not used for search/selection. The caller supplies provenance
    metadata for every family. Missing or mismatched families are rejected.
    """
    if int(minimum_families) < 2 or int(minimum_external_families) < 1 or int(minimum_domains) < 1:
        raise ValueError("minimum family/domain thresholds must be positive (minimum_families >= 2)")
    if int(bootstrap_repetitions) < 1 or int(bootstrap_repetitions) > 1_000_000:
        raise ValueError("bootstrap_repetitions must be between 1 and 1,000,000")
    numeric_thresholds = {
        "minimum_effect": float(minimum_effect),
        "minimum_win_rate": float(minimum_win_rate),
        "maximum_family_regression": float(maximum_family_regression),
    }
    if not all(math.isfinite(value) for value in numeric_thresholds.values()):
        raise ValueError("generalization thresholds must be finite")
    if not 0.0 <= numeric_thresholds["minimum_win_rate"] <= 1.0:
        raise ValueError("minimum_win_rate must be between 0 and 1")
    if numeric_thresholds["maximum_family_regression"] < 0.0:
        raise ValueError("maximum_family_regression must be non-negative")
    candidate = _finite_map(candidate_family_scores, "candidate_family_scores")
    baseline = _finite_map(baseline_family_scores, "baseline_family_scores")
    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline must cover exactly the same families")
    families = sorted(candidate)
    metadata = family_metadata or {}
    missing_metadata = [family for family in families if family not in metadata]
    if missing_metadata:
        raise ValueError(f"family_metadata missing entries: {missing_metadata[:5]}")

    differences = [candidate[family] - baseline[family] for family in families]
    n = len(differences)
    mean_delta = sum(differences) / n
    sorted_deltas = sorted(differences)
    wins = sum(value > 1e-12 for value in differences)
    losses = sum(value < -1e-12 for value in differences)
    ties = n - wins - losses
    win_rate = wins / n
    worst_regression = max(0.0, -min(differences))
    lower_quartile_delta = _quantile(differences, 0.25)

    rng = random.Random(int(seed))
    ci_low, ci_high = _bootstrap_mean_ci(
        differences, repetitions=bootstrap_repetitions, rng=rng
    )
    permutation = _sign_flip_test(
        differences, repetitions=bootstrap_repetitions, rng=rng
    )

    source_by_family = {
        family: str(metadata[family].get("source", "unspecified")).lower()
        for family in families
    }
    domains = {
        str(metadata[family].get("domain", "unspecified")).strip().lower()
        for family in families
    }
    unseen = {
        family for family in families
        if not bool(metadata[family].get("seen_during_search", True))
    }
    external = {
        family for family in unseen if source_by_family[family] == "external"
    }
    all_external = len(external) == n
    no_selection_leakage = not bool(evaluation_used_for_selection)
    preregistration_ok = bool(preregistered)

    effect_gate = ci_low > numeric_thresholds["minimum_effect"]
    significance_gate = float(permutation["p_value"]) <= 0.05
    robustness_gate = win_rate >= numeric_thresholds["minimum_win_rate"] and worst_regression <= numeric_thresholds["maximum_family_regression"]
    internal_gate = (
        n >= int(minimum_families)
        and len(unseen) == n
        and effect_gate
        and significance_gate
        and robustness_gate
        and no_selection_leakage
        and preregistration_ok
    )
    external_gate = (
        internal_gate
        and all_external
        and len(external) >= int(minimum_external_families)
        and len(domains) >= int(minimum_domains)
    )

    if external_gate:
        evidence_tier = "multi_domain_external_holdout_support"
    elif internal_gate:
        evidence_tier = "unseen_family_synthetic_or_mixed_support"
    elif n >= 3 and len(unseen) == n and no_selection_leakage:
        evidence_tier = "preliminary_unseen_family_evidence"
    else:
        evidence_tier = "insufficient_or_selection_contaminated"

    return {
        "schema_version": "amde-generalization-audit-v1",
        "n_families": n,
        "n_unseen_families": len(unseen),
        "n_external_families": len(external),
        "n_domains": len(domains),
        "mean_candidate_score": sum(candidate.values()) / n,
        "mean_baseline_score": sum(baseline.values()) / n,
        "mean_paired_delta": mean_delta,
        "median_paired_delta": _quantile(differences, 0.5),
        "bootstrap_ci_95": [ci_low, ci_high],
        "permutation_test": permutation,
        "family_wins": wins,
        "family_losses": losses,
        "family_ties": ties,
        "family_win_rate": win_rate,
        "lower_quartile_family_delta": lower_quartile_delta,
        "worst_family": families[differences.index(min(differences))],
        "worst_family_delta": min(differences),
        "maximum_family_regression": worst_regression,
        "family_deltas": {family: candidate[family] - baseline[family] for family in families},
        "provenance": {
            "sources": source_by_family,
            "domains": sorted(domains),
            "all_families_unseen": len(unseen) == n,
            "all_families_external": all_external,
            "preregistered": preregistration_ok,
            "evaluation_used_for_selection": bool(evaluation_used_for_selection),
        },
        "gates": {
            "minimum_family_count": n >= int(minimum_families),
            "all_families_unseen": len(unseen) == n,
            "positive_bootstrap_lower_bound": effect_gate,
            "paired_permutation_p_le_0_05": significance_gate,
            "family_robustness": robustness_gate,
            "not_used_for_selection": no_selection_leakage,
            "preregistered": preregistration_ok,
            "minimum_external_family_count": len(external) >= int(minimum_external_families),
            "minimum_domain_count": len(domains) >= int(minimum_domains),
        },
        "internal_generalization_gate": internal_gate,
        "external_generalization_gate": external_gate,
        "evidence_tier": evidence_tier,
        "interpretation": (
            "Empirical evidence is limited to the specified benchmark families, "
            "domains, datasets, metric and evaluation protocol. It is not a proof "
            "of universal or distribution-free generalization."
        ),
        "protocol": {
            "bootstrap_unit": "independent benchmark family",
            "bootstrap_repetitions": max(1000, int(bootstrap_repetitions)),
            "seed": int(seed),
            "minimum_families": int(minimum_families),
            "minimum_external_families": int(minimum_external_families),
            "minimum_domains": int(minimum_domains),
            "minimum_effect": numeric_thresholds["minimum_effect"],
            "minimum_win_rate": numeric_thresholds["minimum_win_rate"],
            "maximum_family_regression": numeric_thresholds["maximum_family_regression"],
        },
    }
