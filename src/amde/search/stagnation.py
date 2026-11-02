"""Passive search-stagnation diagnostics.

This module never mutates search policy automatically. It turns search outcomes
into auditable observations and recommendations, avoiding an untested feedback
loop that could silently change the experimental protocol.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def _phenotype(individual: Any) -> tuple[Any, ...]:
    genome = getattr(individual, "genome", None)
    data = getattr(genome, "data", {}) if genome is not None else {}
    model = data.get("model", {}) if isinstance(data, dict) else {}
    optimizer = data.get("optimizer", {}) if isinstance(data, dict) else {}
    recipe = data.get("recipe", {}) if isinstance(data, dict) else {}
    dsl_data = data.get("dsl", {}) if isinstance(data, dict) else {}
    lr = max(1e-12, float(recipe.get("learning_rate", 0.0) or 0.0))
    return (
        getattr(individual, "level", "?"),
        model.get("kind", "?"),
        tuple(model.get("hidden", ())),
        model.get("activation", "?"),
        optimizer.get("kind", "?"),
        round(math.log10(lr), 1),
        recipe.get("batch_size"),
        recipe.get("loss"),
        str(dsl_data.get("w_update"))[:160],
        str(dsl_data.get("b_update"))[:160],
    )


def analyze_stagnation(
    *,
    plateau_generations: int,
    patience: int,
    population: Iterable[Any],
    proposals: int,
    valid: int,
    invalid: int,
    best_score: float | None,
) -> dict[str, Any]:
    """Return JSON-safe plateau/phenotype-diversity diagnostics and suggestions."""
    items = list(population)
    phenotypes = {_phenotype(item) for item in items}
    diversity = len(phenotypes) / len(items) if items else 0.0
    denominator = max(1, int(valid) + int(invalid))
    invalid_rate = float(invalid) / denominator
    is_stagnated = int(plateau_generations) >= max(1, int(patience))
    recommendations: list[str] = []
    if is_stagnated:
        if diversity < 0.35:
            recommendations.append("phenotype diversity is low: consider a bounded exploration/diversity ablation")
        if invalid_rate > 0.50:
            recommendations.append("invalid-candidate rate is high: inspect mutation operators and genome validation failures")
        if diversity >= 0.35 and invalid_rate <= 0.50:
            recommendations.append("search remains diverse but is not promoting: inspect objective, task split, and promotion guard")
        recommendations.append("do not change search automatically; record a controlled intervention and compare against baseline")
    return {
        "is_stagnated": is_stagnated,
        "plateau_generations": int(plateau_generations),
        "patience_generations": max(1, int(patience)),
        "phenotype_diversity": round(diversity, 6),
        "unique_phenotypes": len(phenotypes),
        "population_size": len(items),
        "invalid_rate": round(invalid_rate, 6),
        "valid": int(valid),
        "invalid": int(invalid),
        "proposals": int(proposals),
        "best_score": float(best_score) if best_score is not None else None,
        "recommendations": recommendations,
        "automatic_intervention": False,
    }
