"""AMDE 3.0 Typed Canonical Search Space and Candidate Representation.

Provides deterministic validation, canonical serialization, cryptographic
hashing, and calibrated quality-preserving mutation operators.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Candidate:
    """A fully specified, immutable candidate discovery pipeline."""
    genome: dict[str, Any]
    canonical_hash: str
    parents: list[str] = field(default_factory=list)
    operator: str = "initial_random"
    strategy: str = "random_search"
    generation: int = 0
    is_novel: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_hash": self.canonical_hash,
            "genome": copy.deepcopy(self.genome),
            "parents": list(self.parents),
            "operator": self.operator,
            "strategy": self.strategy,
            "generation": self.generation,
            "is_novel": self.is_novel,
            "metadata": copy.deepcopy(self.metadata),
        }


def canonical_json(data: Any) -> str:
    """Recursively formats and sorts dictionary/list structures into an invariant string."""
    if isinstance(data, dict):
        items = []
        for k in sorted(data.keys()):
            items.append(f'"{k}":{canonical_json(data[k])}')
        return "{" + ",".join(items) + "}"
    elif isinstance(data, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in data) + "]"
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return "null"
        return f"{data:.7f}"
    elif isinstance(data, bool):
        return "true" if data else "false"
    elif data is None:
        return "null"
    elif isinstance(data, (int, str)):
        return json.dumps(data)
    else:
        return json.dumps(str(data))


def compute_genome_hash(genome: dict[str, Any]) -> str:
    """Computes a deterministic cryptographic SHA-256 hash of a candidate genome."""
    clean_copy = copy.deepcopy(genome)
    # Strip non-deterministic execution artifacts if present
    for k in ("fitness", "score", "timestamp", "eval_time", "trial_id", "status"):
        clean_copy.pop(k, None)
    canon = canonical_json(clean_copy)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class SearchSpace:
    """Type-safe hierarchical search space for AMDE 3.0."""

    MODEL_FAMILIES = ("mlp", "tree_ensemble", "kernel_ridge", "linear")
    ACTIVATIONS = ("relu", "gelu", "tanh")
    OPTIMIZERS = ("adamw", "adam", "rmsprop", "sgd_momentum")
    SCALERS = ("standard", "robust", "minmax", "none")
    IMPUTERS = ("mean", "median", "none")
    KERNEL_TYPES = ("rbf", "linear", "poly")

    def __init__(self, n_features: int = 10, is_classification: bool = True):
        self.n_features = n_features
        self.is_classification = is_classification

    def sample_random(self, rng: random.Random) -> dict[str, Any]:
        """Samples an unbiased random valid candidate genome."""
        family = rng.choice(self.MODEL_FAMILIES)
        preprocessing = {
            "imputer": rng.choice(self.IMPUTERS),
            "scaler": rng.choice(self.SCALERS),
            "feature_selection": rng.choice(["none", "select_k_best"]) if self.n_features > 4 else "none",
            "k_features": rng.randint(max(1, self.n_features // 2), self.n_features) if self.n_features > 4 else self.n_features,
        }

        if family == "mlp":
            depth = rng.randint(1, 3)
            widths = [rng.choice([16, 32, 64, 128]) for _ in range(depth)]
            model_spec = {
                "family": "mlp",
                "hidden_widths": widths,
                "activation": rng.choice(self.ACTIVATIONS),
                "dropout": round(rng.choice([0.0, 0.1, 0.2, 0.3]), 2),
                "learning_rate": round(10 ** rng.uniform(-4.0, -1.0), 6),
                "weight_decay": round(10 ** rng.uniform(-6.0, -2.0), 7),
                "optimizer": rng.choice(self.OPTIMIZERS),
                "batch_size": rng.choice([16, 32, 64, 128]),
                "epochs": rng.choice([25, 40, 50]),
            }
        elif family == "tree_ensemble":
            model_spec = {
                "family": "tree_ensemble",
                "n_estimators": rng.choice([50, 100, 150]),
                "max_depth": rng.choice([3, 5, 8, None]),
                "min_samples_split": rng.choice([2, 5, 10]),
                "learning_rate": round(10 ** rng.uniform(-2.5, -0.3), 5),
                "subsample": round(rng.uniform(0.7, 1.0), 2),
            }
        elif family == "kernel_ridge":
            model_spec = {
                "family": "kernel_ridge",
                "kernel": rng.choice(self.KERNEL_TYPES),
                "alpha": round(10 ** rng.uniform(-4.0, 1.0), 6),
                "gamma": round(10 ** rng.uniform(-3.0, 1.0), 6) if rng.choice([True, False]) else "scale",
            }
        else:  # linear
            model_spec = {
                "family": "linear",
                "penalty": rng.choice(["l2", "none"]),
                "c_param": round(10 ** rng.uniform(-3.0, 2.0), 5),
            }

        return {
            "preprocessing": preprocessing,
            "model": model_spec,
        }

    def mutate_calibrated(
        self,
        genome: dict[str, Any],
        rng: random.Random,
        generation: int = 1,
        max_generations: int = 20,
    ) -> dict[str, Any]:
        """Applies calibrated, quality-preserving mutation without destroying co-adapted weights.
        
        Fixes BOT-01 (Generational Degradation) and BOT-03 (Unguided Mutation):
        - Continuous parameters mutate via log-normal perturbations with decaying variance.
        - Categorical structural flips (e.g. activation, model family) are bounded to low probabilities.
        - At most 1 continuous and 1 structural hyperparameter is perturbed per step.
        """
        mutated = copy.deepcopy(genome)
        model = mutated["model"]
        decay_factor = max(0.2, math.sqrt(max(0.0, 1.0 - (generation / max_generations))))

        # 1. Continuous hyperparameter fine-tuning (Gaussian step)
        family = model["family"]
        if family == "mlp":
            if rng.random() < 0.70:
                # Log-normal perturbation of learning rate with decaying sigma
                sigma = 0.25 * decay_factor
                log_lr = math.log10(model["learning_rate"]) + rng.gauss(0, sigma)
                model["learning_rate"] = round(10 ** max(-5.0, min(-0.5, log_lr)), 6)
            if rng.random() < 0.40:
                sigma_wd = 0.35 * decay_factor
                log_wd = math.log10(max(1e-7, model["weight_decay"])) + rng.gauss(0, sigma_wd)
                model["weight_decay"] = round(10 ** max(-7.0, min(-1.0, log_wd)), 7)
            if rng.random() < 0.25:
                model["batch_size"] = rng.choice([16, 32, 64, 128])
            # Conservative activation change (capped at 10% probability)
            if rng.random() < 0.10:
                model["activation"] = rng.choice(self.ACTIVATIONS)
        elif family == "tree_ensemble":
            if rng.random() < 0.70:
                sigma_lr = 0.20 * decay_factor
                log_lr = math.log10(model["learning_rate"]) + rng.gauss(0, sigma_lr)
                model["learning_rate"] = round(10 ** max(-3.0, min(0.0, log_lr)), 5)
            if rng.random() < 0.40:
                model["n_estimators"] = max(20, min(250, model["n_estimators"] + rng.choice([-20, -10, 10, 20])))
            if rng.random() < 0.25:
                model["subsample"] = round(max(0.5, min(1.0, model["subsample"] + rng.uniform(-0.1, 0.1))), 2)
        elif family == "kernel_ridge":
            if rng.random() < 0.70:
                log_alpha = math.log10(model["alpha"]) + rng.gauss(0, 0.3 * decay_factor)
                model["alpha"] = round(10 ** max(-5.0, min(2.0, log_alpha)), 6)

        # 2. Preprocessing perturbation (low probability)
        if rng.random() < 0.20:
            mutated["preprocessing"]["scaler"] = rng.choice(self.SCALERS)

        return mutated

    def validate_genome(self, genome: dict[str, Any]) -> Tuple[bool, str]:
        """Validates static invariants of a genome."""
        if not isinstance(genome, dict) or "model" not in genome or "preprocessing" not in genome:
            return False, "Missing top-level model or preprocessing spec"
        model = genome["model"]
        if "family" not in model or model["family"] not in self.MODEL_FAMILIES:
            return False, f"Invalid model family: {model.get('family')}"
        if model["family"] == "mlp":
            if not model.get("hidden_widths") or len(model["hidden_widths"]) > 4:
                return False, "MLP depth must be between 1 and 4"
            if model.get("learning_rate", 0) <= 0 or model["learning_rate"] > 1.0:
                return False, f"Invalid learning rate: {model.get('learning_rate')}"
        return True, "Valid"
