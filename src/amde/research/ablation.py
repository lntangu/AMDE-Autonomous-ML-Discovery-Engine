"""Deterministic ablation design and family-clustered paired analysis."""
from __future__ import annotations

import itertools
import math
from typing import Any, Mapping

from ..util import hash_json
from .statistics import add_rejections, bootstrap_mean_ci, holm_bonferroni, paired_sign_flip


def build_ablation_plan(base_config: Mapping[str, Any], factors: Mapping[str, list[Any]], *,
                        mode: str = "one_factor", seeds: list[int] | None = None,
                        max_variants: int = 128) -> dict[str, Any]:
    if mode not in {"one_factor", "full_factorial"}:
        raise ValueError("mode must be one_factor or full_factorial")
    if not isinstance(base_config, Mapping) or not isinstance(factors, Mapping) or not factors:
        raise ValueError("base_config and non-empty factors are required")
    cleaned: dict[str, list[Any]] = {}
    for name, values in factors.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(values, list) or len(values) < 2:
            raise ValueError("each factor needs a name and at least two levels")
        if len({hash_json(value) for value in values}) != len(values):
            raise ValueError(f"factor {name!r} has duplicate levels")
        cleaned[name] = values
    variants: list[dict[str, Any]] = [{"variant_id": "control", "factor_values": {}, "config": dict(base_config)}]
    expected_variants = (1 + sum(len(values) for values in cleaned.values())
                         if mode == "one_factor" else 1 + math.prod(len(values) for values in cleaned.values()))
    if expected_variants > max_variants:
        raise ValueError(f"ablation plan needs {expected_variants} variants, above max_variants={max_variants}")
    if mode == "one_factor":
        combos = [(name, value) for name, values in cleaned.items() for value in values]
        for name, value in combos:
            if value == base_config.get(name):
                continue
            variants.append({"variant_id": "", "factor_values": {name: value},
                             "config": {**dict(base_config), name: value}})
    else:
        combinations = itertools.product(*(cleaned[name] for name in cleaned))
        for values in combinations:
            update = dict(zip(cleaned, values))
            if all(base_config.get(key) == value for key, value in update.items()):
                continue
            variants.append({"variant_id": "", "factor_values": update,
                             "config": {**dict(base_config), **update}})
    if len(variants) > max_variants:
        raise ValueError(f"ablation plan has {len(variants)} variants, above max_variants={max_variants}")
    for item in variants[1:]:
        item["variant_id"] = "v_" + hash_json(item["factor_values"])[:12]
    seed_list = list(seeds or [11, 23, 37, 53, 71])
    if len(seed_list) < 3 or len(set(seed_list)) != len(seed_list) or any(isinstance(x, bool) or not isinstance(x, int) for x in seed_list):
        raise ValueError("seeds must contain at least three unique integers")
    plan = {
        "schema_version": "amde-ablation-plan-v1",
        "mode": mode,
        "base_config": dict(base_config),
        "factors": cleaned,
        "seeds": sorted(seed_list),
        "control_variant_id": "control",
        "variants": variants,
        "max_variants": max_variants,
        "decision_rule": "paired_family_seed_differences; Holm correction across non-control variants",
    }
    plan["protocol_sha256"] = hash_json(plan)
    return plan


def audit_ablation_results(payload: Mapping[str, Any], *, alpha: float = 0.05,
                           bootstrap_repetitions: int = 20000, seed: int = 42) -> dict[str, Any]:
    if payload.get("schema_version") != "amde-ablation-results-v1":
        raise ValueError("schema_version must be amde-ablation-results-v1")
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("results must be a non-empty list")
    control_id = str(payload.get("control_variant_id", "control"))
    index: dict[tuple[str, str, int], float] = {}
    variants: set[str] = set()
    for row in rows:
        variant = str(row.get("variant_id", ""))
        family = str(row.get("family_id", ""))
        run_seed = row.get("seed")
        score = float(row.get("score"))
        if not variant or not family or isinstance(run_seed, bool) or not isinstance(run_seed, int) or not math.isfinite(score):
            raise ValueError("each result needs variant_id, family_id, integer seed and finite score")
        key = (variant, family, run_seed)
        if key in index:
            raise ValueError(f"duplicate result key: {key}")
        index[key] = score
        variants.add(variant)
    if control_id not in variants:
        raise ValueError("control variant is missing")
    control = {(family, run_seed): score for (variant, family, run_seed), score in index.items() if variant == control_id}
    reports: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for variant in sorted(variants - {control_id}):
        treatment = {(family, run_seed): score for (v, family, run_seed), score in index.items() if v == variant}
        if set(control) != set(treatment):
            missing = sorted(set(control) - set(treatment))[:5]
            extra = sorted(set(treatment) - set(control))[:5]
            raise ValueError(
                f"variant {variant} has incomplete/mismatched paired coverage; "
                f"missing={missing}, extra={extra}"
            )
        shared = sorted(set(control))
        family_count = len({family for family, _ in shared})
        if family_count < 6:
            raise ValueError(
                f"variant {variant} has {family_count} independent families; "
                "at least six are required for this confirmatory sign-flip audit"
            )
        by_family: dict[str, list[float]] = {}
        for family, run_seed in shared:
            by_family.setdefault(family, []).append(treatment[(family, run_seed)] - control[(family, run_seed)])
        family_deltas = {family: sum(values) / len(values) for family, values in by_family.items()}
        deltas = [family_deltas[key] for key in sorted(family_deltas)]
        ci = bootstrap_mean_ci(deltas, repetitions=bootstrap_repetitions, seed=seed, confidence=0.95)
        test = paired_sign_flip(deltas, alternative="two-sided", seed=seed)
        raw_p[variant] = float(test["p_value"])
        reports[variant] = {
            "n_paired_family_seed": len(shared),
            "n_families": len(family_deltas),
            "mean_family_effect": sum(deltas) / len(deltas),
            "family_cluster_bootstrap_ci_95": list(ci),
            "paired_sign_flip": test,
            "family_deltas": family_deltas,
        }
    adjusted = add_rejections(holm_bonferroni(raw_p), alpha=alpha)
    for variant, adjustment in adjusted.items():
        reports[variant]["multiple_testing"] = adjustment
    return {
        "schema_version": "amde-ablation-audit-v1",
        "control_variant_id": control_id,
        "alpha": alpha,
        "correction": "Holm-Bonferroni",
        "variants": reports,
        "interpretation": "Effects are descriptive/confirmatory only to the extent the design and held-out families were preregistered and not used for selection.",
    }
