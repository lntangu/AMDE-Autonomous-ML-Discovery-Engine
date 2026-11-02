"""Dependency-free statistical primitives for preregistered research audits.

These routines deliberately return assumptions and method labels. They are
not substitutes for domain-specific design review or an independent statistician.
"""
from __future__ import annotations

import itertools
import math
import random
from statistics import NormalDist
from typing import Iterable


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    p = min(1.0, max(0.0, float(probability)))
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(values: list[float], *, repetitions: int = 20000,
                      seed: int = 42, confidence: float = 0.95) -> tuple[float, float]:
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("bootstrap requires non-empty finite values")
    if repetitions < 1000 or repetitions > 1_000_000:
        raise ValueError("repetitions must be between 1000 and 1,000,000")
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must be in (0.5, 1)")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(float(values[rng.randrange(n)]) for _ in range(n)) / n for _ in range(repetitions)]
    tail = (1.0 - confidence) / 2.0
    return quantile(means, tail), quantile(means, 1.0 - tail)


def paired_sign_flip(differences: list[float], *, alternative: str = "two-sided",
                     repetitions: int = 100000, seed: int = 42) -> dict[str, float | int | str]:
    """Paired sign-flip test over independent clusters (e.g. benchmark families)."""
    if not differences or any(not math.isfinite(float(v)) for v in differences):
        raise ValueError("sign-flip test requires non-empty finite paired differences")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    values = [float(value) for value in differences]
    n = len(values)
    observed = sum(values) / n

    def extreme(candidate: float) -> bool:
        if alternative == "greater":
            return candidate >= observed - 1e-15
        if alternative == "less":
            return candidate <= observed + 1e-15
        return abs(candidate) >= abs(observed) - 1e-15

    if n <= 18:
        hits = 0
        total = 1 << n
        for mask in range(total):
            statistic = sum(value if (mask >> index) & 1 else -value
                            for index, value in enumerate(values)) / n
            hits += int(extreme(statistic))
        p_value = hits / total
        method = "exact_paired_sign_flip"
    else:
        rng = random.Random(seed)
        total = max(10000, int(repetitions))
        hits = 0
        for _ in range(total):
            statistic = sum(value if rng.random() < 0.5 else -value for value in values) / n
            hits += int(extreme(statistic))
        p_value = (hits + 1) / (total + 1)
        method = "monte_carlo_paired_sign_flip"
    return {"method": method, "alternative": alternative, "p_value": p_value,
            "permutations": total, "mean_difference": observed,
            "n_independent_clusters": n}


def power_analysis(*, minimum_effect: float, expected_sd: float, alpha: float = 0.05,
                   target_power: float = 0.80, direction: str = "two-sided",
                   minimum_clusters: int = 6, maximum_clusters: int = 100000) -> dict[str, float | int | str]:
    """Normal-approximation planning for paired family-level mean differences.

    This is a transparent first-pass approximation, not an exact t-test or a
    guarantee of power. The cluster floor also accounts for the minimum
    attainable p-value of the exact sign-flip test at the selected alpha.
    Actual design should revisit variance and dependence assumptions.
    """
    effect = float(minimum_effect)
    sd = float(expected_sd)
    alpha = float(alpha)
    target_power = float(target_power)
    if not math.isfinite(effect) or effect <= 0:
        raise ValueError("minimum_effect must be finite and > 0")
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("expected_sd must be finite and > 0")
    if not 0.001 <= alpha <= 0.2:
        raise ValueError("alpha must be between 0.001 and 0.2")
    if not 0.5 < target_power < 0.999:
        raise ValueError("target_power must be between 0.5 and 0.999")
    if direction not in {"two-sided", "greater", "less"}:
        raise ValueError("direction must be two-sided, greater, or less")
    if minimum_clusters < 2 or maximum_clusters < minimum_clusters:
        raise ValueError("invalid cluster limits")
    normal = NormalDist()
    critical_probability = 1.0 - alpha / 2.0 if direction == "two-sided" else 1.0 - alpha
    z_critical = normal.inv_cdf(critical_probability)
    z_power = normal.inv_cdf(target_power)
    raw = ((z_critical + z_power) * sd / effect) ** 2
    # Exact sign-flip resolution is 2/2**n (two-sided) or 1/2**n
    # (one-sided); ensure at least one attainable p-value can reach alpha.
    resolution_numerator = 2.0 if direction == "two-sided" else 1.0
    resolution_floor = int(math.ceil(math.log2(resolution_numerator / alpha)))
    effective_floor = max(int(minimum_clusters), resolution_floor)
    estimated = max(effective_floor, int(math.ceil(raw)))
    capped = estimated > maximum_clusters
    return {
        "method": "paired_mean_normal_approximation",
        "direction": direction,
        "minimum_effect": effect,
        "expected_sd": sd,
        "standardized_effect": effect / sd,
        "alpha": alpha,
        "target_power": target_power,
        "z_critical": z_critical,
        "z_power": z_power,
        "raw_cluster_estimate": raw,
        "recommended_independent_families": min(estimated, maximum_clusters),
        "minimum_cluster_floor": minimum_clusters,
        "exact_sign_flip_resolution_floor": resolution_floor,
        "effective_cluster_floor": effective_floor,
        "maximum_clusters": maximum_clusters,
        "capped": capped,
        "assumption": "independent family-level paired differences with approximately normal mean estimator; ignores finite-sample t correction and family dependence",
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    """Return Holm adjusted p-values and reject flags, preserving original keys."""
    for key, value in p_values.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid p-value for {key}: {value}")
    ordered = sorted(((float(value), str(key)) for key, value in p_values.items()))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (p_value, key) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[key] = running
    return {key: {"p_value": float(value), "p_adjusted": adjusted[key]}
            for key, value in p_values.items()}


def add_rejections(values: dict[str, dict[str, float]], *, alpha: float) -> dict[str, dict[str, float | bool]]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    return {key: {**result, "reject": bool(result["p_adjusted"] <= alpha)}
            for key, result in values.items()}
