"""Schema-safe mutation and crossover.

Operators validate every child and repair/fallback deterministically. They may
not emit an unchecked genome; invalid genetic material is isolated before it
reaches the evaluator.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from ..errors import ValidationError
from . import dsl
from .factory import _validate_kwargs, random_genome, upgrade_genome
from .genome import Genome


def _safe_validate(data: dict[str, Any], safety: dict[str, Any]) -> Genome:
    return Genome(data, **_validate_kwargs(safety))


def mutate(
    genome: Genome,
    rng: random.Random,
    *,
    target_level: str | None = None,
    safety: dict[str, Any] | None = None,
    small: bool = False,
) -> Genome:
    safety = safety or {}
    target = target_level or genome.level
    if target not in ("A", "B", "C", "D"):
        raise ValueError(f"unknown target level: {target}")
    base = upgrade_genome(genome, target, rng, safety=safety)
    for attempt in range(8):
        data = base.to_dict()
        data["level"] = target
        recipe = data["recipe"]
        optimizer = data["optimizer"]
        factor = 1.25 if small else 1.8
        # Numeric parameters (all levels).
        if rng.random() < 0.8:
            recipe["learning_rate"] = round(
                max(1e-4, min(1.0, recipe["learning_rate"] * rng.expovariate(1.0 / 0.35) if small else
                    recipe["learning_rate"] * rng.choice([0.5, 0.7, 0.9, 1.1, 1.4, 2.0]))),
                7,
            )
        if rng.random() < 0.45:
            recipe["batch_size"] = rng.choice([8, 12, 16, 24, 32, 48, 64, 96, 128])
        if rng.random() < 0.45:
            recipe["epochs"] = max(2, min(80, int(recipe["epochs"] + rng.choice([-10, -5, -2, 2, 5, 10]))))
        if rng.random() < 0.22:
            recipe["lr_schedule"] = rng.choice(["constant", "cosine", "linear", "warmup_cosine"])
        if rng.random() < 0.45:
            recipe["l2"] = round(
                max(0.0, min(0.25, recipe["l2"] * rng.choice([0.3, 0.6, 1.5, 3.0]))), 8
            )
        if rng.random() < 0.3:
            recipe["init_scale"] = round(max(0.001, min(2.0, recipe["init_scale"] * rng.choice([0.6, 1.4]))), 6)
        if rng.random() < 0.2:
            recipe["seed_policy"] = rng.choice(["hash", "constant"])

        if target in ("B", "C", "D"):
            if rng.random() < 0.25:
                optimizer["kind"] = rng.choice(["sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion"])
            if rng.random() < 0.35:
                optimizer["beta1"] = round(max(0.0, min(0.999, optimizer["beta1"] + rng.uniform(-0.08, 0.08))), 6)
                optimizer["beta2"] = round(max(0.9, min(0.999999, optimizer["beta2"] + rng.uniform(-0.02, 0.02))), 8)
            if rng.random() < 0.25:
                recipe["init"] = rng.choice(["normal", "uniform", "zeros"])
            if rng.random() < 0.22:
                recipe["loss"] = rng.choice(["auto", "mse", "bce"])
            if rng.random() < 0.2:
                recipe["standardize"] = not bool(recipe["standardize"])
            if rng.random() < 0.25:
                recipe["grad_clip"] = rng.choice([0.0, 0.5, 1.0, 2.5, 5.0, 10.0])
            if rng.random() < 0.25:
                recipe["early_stopping_patience"] = rng.choice([0, 2, 4, 6, 10])
            if rng.random() < 0.15:
                recipe["l1"] = round(max(0.0, min(0.05, recipe["l1"] * rng.choice([0.2, 2.0]))), 8)

        if target in ("C", "D") and rng.random() < 0.55:
            if rng.random() < 0.25:
                data["model"] = {"kind": "linear", "hidden": [], "activation": data["model"]["activation"]}
            else:
                max_width = int(safety.get("max_hidden_width", 48))
                max_layers = int(safety.get("max_hidden_layers", 3))
                widths = [width for width in (4, 8, 12, 16, 24, 32, 48) if width <= max_width]
                if data["model"]["kind"] == "linear" or not data["model"]["hidden"]:
                    hidden = [rng.choice(widths or [max_width])]
                else:
                    hidden = list(data["model"]["hidden"])
                    idx = rng.randrange(len(hidden))
                    hidden[idx] = max(1, min(max_width, hidden[idx] + rng.choice([-8, -4, -2, 2, 4, 8])))
                    if rng.random() < 0.12 and len(hidden) < max_layers:
                        hidden.append(rng.choice(widths or [max_width]))
                data["model"] = {
                    "kind": "mlp",
                    "hidden": hidden,
                    "activation": rng.choice(["tanh", "relu", "sigmoid"]),
                }

        if target == "D":
            optimizer["kind"] = "dsl"
            if rng.random() < 0.16:
                updates = dsl.dsl_template(
                    rng.choice(["sgd", "momentum", "adam", "rmsprop", "sign", "clipped"]),
                    beta1=optimizer["beta1"],
                    beta2=optimizer["beta2"],
                )
                for key in ("w_update", "b_update", "m_update", "v_update"):
                    data["dsl"][key] = updates.get(key)
            else:
                for key in ("m_update", "v_update", "w_update", "b_update"):
                    expression = data["dsl"].get(key)
                    if expression is not None and rng.random() < 0.55:
                        try:
                            data["dsl"][key] = dsl.mutate_expr(
                                expression,
                                rng,
                                max_nodes=int(safety.get("max_dsl_nodes", 96)),
                            )
                        except Exception:
                            pass
                if data["dsl"].get("w_update") is None or data["dsl"].get("b_update") is None:
                    updates = dsl.dsl_template("sgd")
                    data["dsl"]["w_update"] = updates["w_update"]
                    data["dsl"]["b_update"] = updates["w_update"]
        else:
            for key in ("w_update", "b_update", "m_update", "v_update"):
                data["dsl"][key] = None
            if optimizer["kind"] == "dsl":
                optimizer["kind"] = "sgd"
        try:
            return _safe_validate(data, safety)
        except ValidationError:
            # Try a different perturbation of the same validated parent.
            continue
    # The fallback is deliberately random but fully validated; operators never
    # leak an invalid genome to the evaluator.
    return random_genome(target, rng, safety=safety)


def crossover(
    left: Genome,
    right: Genome,
    rng: random.Random,
    *,
    target_level: str | None = None,
    safety: dict[str, Any] | None = None,
) -> Genome:
    safety = safety or {}
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    target = target_level or (left.level if order[left.level] >= order[right.level] else right.level)
    for attempt in range(10):
        # Start from one parent at a level that supports the target.
        start = left if order[left.level] >= order[right.level] else right
        data = start.to_dict()
        data["level"] = target
        other = right.to_dict() if start is left else left.to_dict()
        if target in ("B", "C", "D"):
            data["optimizer"] = copy.deepcopy(other["optimizer"] if rng.random() < 0.5 else data["optimizer"])
            for key in ("learning_rate", "batch_size", "epochs", "l2", "l1", "init", "init_scale",
                        "loss", "standardize", "grad_clip", "early_stopping_patience", "seed_policy", "lr_schedule"):
                if rng.random() < 0.5:
                    data["recipe"][key] = copy.deepcopy(other["recipe"].get(key, data["recipe"][key]))
        if target in ("C", "D"):
            if rng.random() < 0.5:
                data["model"] = copy.deepcopy(other["model"])
        if target == "A":
            # A permits only the narrow numerical gene set. Preserve the
            # higher-level parent's A-compatible genes, then sanitize.
            data["model"] = copy.deepcopy(base_model())
            data["optimizer"] = copy.deepcopy(_default_optimizer())
            for key in ("l1", "init", "loss", "standardize", "grad_clip", "early_stopping_patience"):
                data["recipe"][key] = copy.deepcopy(_default_recipe()[key])
            for key in ("w_update", "b_update", "m_update", "v_update"):
                data["dsl"][key] = None
        if target == "B":
            data["model"] = {"kind": "linear", "hidden": [], "activation": data["model"]["activation"]}
            for key in ("w_update", "b_update", "m_update", "v_update"):
                data["dsl"][key] = None
            if data["optimizer"]["kind"] == "dsl":
                data["optimizer"]["kind"] = "sgd"
        if target == "C":
            for key in ("w_update", "b_update", "m_update", "v_update"):
                data["dsl"][key] = None
            if data["optimizer"]["kind"] == "dsl":
                data["optimizer"]["kind"] = "adam"
        if target == "D":
            data["optimizer"]["kind"] = "dsl"
            if left.level == "D":
                source = left.to_dict()["dsl"]
            elif right.level == "D":
                source = right.to_dict()["dsl"]
            else:
                source = dsl.dsl_template("sgd")
            for key in ("w_update", "b_update", "m_update", "v_update"):
                data["dsl"][key] = copy.deepcopy(source.get(key))
            if data["dsl"].get("w_update") is None or data["dsl"].get("b_update") is None:
                fallback = dsl.dsl_template("sgd")
                data["dsl"]["w_update"] = fallback["w_update"]
                data["dsl"]["b_update"] = fallback["w_update"]
        try:
            return _safe_validate(data, safety)
        except ValidationError:
            continue
    return mutate(left, rng, target_level=target, safety=safety)


def base_model() -> dict[str, Any]:
    return {"kind": "linear", "hidden": [], "activation": "tanh"}


def _default_optimizer() -> dict[str, Any]:
    return {"kind": "sgd", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}


def _default_recipe() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "batch_size": 24,
        "epochs": 30,
        "l2": 0.001,
        "l1": 0.0,
        "init": "normal",
        "init_scale": 0.5,
        "loss": "auto",
        "standardize": True,
        "grad_clip": 5.0,
        "early_stopping_patience": 5,
        "seed_policy": "hash",
    }
