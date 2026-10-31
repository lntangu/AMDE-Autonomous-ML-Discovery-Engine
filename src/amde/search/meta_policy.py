"""Experimental task-conditioned search meta-policy.

Level E in the original design is a *search policy*, not a gene in a genome.
It is deliberately disabled by default. When enabled it only reweights which
safe proposal strategy is tried next; it cannot modify candidates, metrics, or
sealed data. It uses only aggregate task metadata from the open data manifest.
"""

from __future__ import annotations

import random
from typing import Any

from .portfolio import Portfolio
from .strategies import STRATEGIES


class MetaPolicy:
    def __init__(self, config: Any) -> None:
        self.enabled = bool(config.get("search.meta_policy_enabled", False))
        self.status = "enabled_experimental" if self.enabled else "disabled"
        self.version = "meta-policy-experimental-v1"

    def features(self, data_manifest: dict[str, Any]) -> dict[str, Any]:
        # Held-out task metadata must not influence the development search policy.
        tasks = [task for task in data_manifest.get("tasks", []) if not bool(task.get("holdout_only", False))]
        n_tasks = max(1, len(tasks))
        n_features = [float(task.get("n_features", 0)) for task in tasks]
        kinds = [str(task.get("kind", "")) for task in tasks]
        families = {str(task.get("family", "")) for task in tasks}
        return {
            "n_tasks": n_tasks,
            "mean_n_features": sum(n_features) / n_tasks if n_features else 0.0,
            "max_n_features": max(n_features) if n_features else 0.0,
            "classification_ratio": sum(1 for kind in kinds if kind == "classification") / n_tasks,
            "n_families": len(families),
        }

    def prior(self, features: dict[str, Any]) -> dict[str, float]:
        weights = {name: 1.0 for name in STRATEGIES}
        if float(features.get("classification_ratio", 0.0)) >= 0.5:
            weights["crossover"] += 0.25
            weights["evolutionary"] += 0.15
        if float(features.get("mean_n_features", 0.0)) >= 5.0:
            weights["local"] += 0.25
        if int(features.get("n_tasks", 0)) >= 12:
            weights["random"] += 0.2
        total = sum(weights.values())
        return {name: value / total for name, value in weights.items()}

    def choose(
        self,
        portfolio: Portfolio,
        rng: random.Random,
        features: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return portfolio.select(rng), {"mode": "disabled", "prior": None}
        prior = self.prior(features)
        draw = rng.random()
        cumulative = 0.0
        chosen = STRATEGIES[-1]
        for name in STRATEGIES:
            cumulative += prior[name]
            if draw <= cumulative:
                chosen = name
                break
        return chosen, {"mode": "experimental_prior", "prior": prior, "features": features}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "enabled": self.enabled, "status": self.status}
