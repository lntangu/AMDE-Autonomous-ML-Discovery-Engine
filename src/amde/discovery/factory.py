"""Reproducible candidate factory with a search curriculum."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Iterable

from ..errors import ValidationError
from . import dsl
from .genome import (
    DEFAULT_DSL,
    DEFAULT_MODEL,
    DEFAULT_OPTIMIZER,
    DEFAULT_RECIPE,
    Genome,
    validate_genome,
)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _jitter(rng: random.Random, value: float, low: float, high: float, factor: float = 1.8) -> float:
    proposal = value * math.exp(rng.uniform(-math.log(factor), math.log(factor)))
    return max(low, min(high, proposal))


def default_genome(level: str = "A") -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "level": level,
        "model": copy.deepcopy(DEFAULT_MODEL),
        "optimizer": copy.deepcopy(DEFAULT_OPTIMIZER),
        "recipe": copy.deepcopy(DEFAULT_RECIPE),
        "dsl": copy.deepcopy(DEFAULT_DSL),
    }


def safety_kwargs(config: Any) -> dict[str, Any]:
    safety = config.get("safety", {})
    return {
        "max_dsl_nodes": int(safety.get("max_dsl_nodes", 96)),
        "max_dsl_depth": int(safety.get("max_dsl_depth", 12)),
        "max_genome_bytes": int(safety.get("max_genome_bytes", 65536)),
        "max_hidden_layers": int(safety.get("max_hidden_layers", 2)),
        "max_hidden_width": int(safety.get("max_hidden_width", 24)),
    }


def random_genome(
    level: str,
    rng: random.Random,
    *,
    safety: dict[str, Any] | None = None,
) -> Genome:
    """Create a valid random genome at exactly the requested level."""

    if level not in ("A", "B", "C", "D"):
        raise ValueError(f"unknown level: {level}")
    safety = safety or {}
    data = default_genome(level)
    recipe = data["recipe"]
    recipe["learning_rate"] = round(_log_uniform(rng, 0.002, 0.2), 6)
    recipe["batch_size"] = rng.choice([8, 12, 16, 24, 32, 48, 64, 96, 128])
    recipe["epochs"] = rng.choice([12, 18, 24, 30, 40, 55, 70, 80])
    recipe["l2"] = round(_log_uniform(rng, 1e-5, 0.02), 7)
    recipe["init_scale"] = round(_log_uniform(rng, 0.1, 1.6), 5)
    recipe["seed_policy"] = rng.choice(["hash", "constant"])

    if level in ("B", "C", "D"):
        optimizer = data["optimizer"]
        optimizer["kind"] = rng.choice(["sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion"])
        optimizer["beta1"] = round(rng.uniform(0.75, 0.98), 5)
        optimizer["beta2"] = round(rng.uniform(0.95, 0.9995), 7)
        optimizer["epsilon"] = round(_log_uniform(rng, 1e-9, 1e-5), 10)
        recipe["init"] = rng.choice(["normal", "uniform", "zeros"])
        recipe["loss"] = rng.choice(["auto", "auto", "mse"])
        recipe["standardize"] = rng.choice([True, True, False])
        recipe["grad_clip"] = rng.choice([0.0, 1.0, 2.5, 5.0, 10.0])
        recipe["early_stopping_patience"] = rng.choice([0, 3, 5, 8])
        recipe["lr_schedule"] = rng.choice(["constant", "cosine", "linear", "warmup_cosine"])
        if rng.random() < 0.35:
            recipe["l1"] = round(_log_uniform(rng, 1e-6, 0.003), 8)
    if level in ("C", "D"):
        if rng.random() < 0.35:
            data["model"] = {"kind": "linear", "hidden": [], "activation": "tanh"}
        else:
            max_width = int(safety.get("max_hidden_width", 48))
            max_layers = int(safety.get("max_hidden_layers", 3))
            widths = [width for width in (4, 8, 12, 16, 24, 32, 48) if width <= max_width]
            hidden = [rng.choice(widths or [max_width])]
            while len(hidden) < max_layers and rng.random() < 0.24:
                hidden.append(rng.choice(widths or [max_width]))
            data["model"] = {
                "kind": "mlp",
                "hidden": hidden,
                "activation": rng.choice(["tanh", "relu", "sigmoid"]),
            }
    if level == "D":
        data["optimizer"]["kind"] = "dsl"
        template = rng.choice(["sgd", "momentum", "adam", "rmsprop", "sign", "clipped"])
        try:
            updates = dsl.dsl_template(
                template,
                beta1=data["optimizer"]["beta1"],
                beta2=data["optimizer"]["beta2"],
            )
        except Exception:
            updates = dsl.dsl_template("sgd")
        data["dsl"].update(updates)
        data["dsl"]["version"] = dsl.DSL_VERSION
        if rng.random() < 0.35:
            # Keep the template structure but explore small local variants.
            key = rng.choice(list(updates))
            try:
                data["dsl"][key] = dsl.mutate_expr(
                    updates[key],
                    rng,
                    max_nodes=int(safety.get("max_dsl_nodes", 96)),
                )
            except Exception:
                pass
    try:
        return Genome(data, **_validate_kwargs(safety))
    except ValidationError:
        # A deterministic, conservative fallback at the same level. The
        # fallback is still validated; it is never an unchecked candidate.
        data = default_genome(level)
        if level in ("B", "C", "D"):
            data["optimizer"]["kind"] = "adam"
            data["recipe"]["learning_rate"] = 0.01
        if level in ("C", "D"):
            data["model"] = {"kind": "mlp", "hidden": [8], "activation": "tanh"}
        if level == "D":
            data["optimizer"]["kind"] = "dsl"
            updates = dsl.dsl_template("sgd")
            data["dsl"].update(updates)
        return Genome(data, **_validate_kwargs(safety))


def _validate_kwargs(safety: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_dsl_nodes": int(safety.get("max_dsl_nodes", 96)),
        "max_dsl_depth": int(safety.get("max_dsl_depth", 12)),
        "max_genome_bytes": int(safety.get("max_genome_bytes", 65536)),
        "max_hidden_layers": int(safety.get("max_hidden_layers", 2)),
        "max_hidden_width": int(safety.get("max_hidden_width", 24)),
    }


def available_levels(config: Any, generation: int) -> list[str]:
    configured = list(config.get("search.levels", ["A", "B", "C", "D"]))
    opening = config.get("search.open_level_after_generation", {})
    result = []
    for level in configured:
        if generation >= int(opening.get(level, 0)):
            result.append(level)
    return result or [configured[0]]


def upgrade_genome(
    genome: Genome,
    target_level: str,
    rng: random.Random,
    *,
    safety: dict[str, Any] | None = None,
) -> Genome:
    """Move a genome to a higher level, filling new capabilities safely."""

    if target_level == genome.level:
        return genome
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    if order[target_level] < order[genome.level]:
        return genome
    data = genome.to_dict()
    data["level"] = target_level
    if target_level in ("B", "C", "D"):
        if data["optimizer"]["kind"] == "sgd" and rng.random() < 0.5:
            data["optimizer"]["kind"] = rng.choice(["momentum", "adagrad", "adam", "nadam", "rmsprop"])
    if target_level in ("C", "D") and data["model"]["kind"] == "linear" and rng.random() < 0.55:
        data["model"] = {"kind": "mlp", "hidden": [rng.choice([4, 6, 8, 12])], "activation": rng.choice(["tanh", "relu"])}
    if target_level == "D":
        data["optimizer"]["kind"] = "dsl"
        updates = dsl.dsl_template(rng.choice(["adam", "momentum", "sgd", "sign"]))
        data["dsl"].update(updates)
    return Genome(data, **_validate_kwargs(safety or {}))
