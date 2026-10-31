"""Versioned, schema-checked candidate genomes.

A genome is plain JSON. Levels A-D progressively add capability:

A: numerical/optimization parameters of a fixed linear SGD recipe.
B: learning recipe (optimizer, loss, initialization, preprocessing, scheduling).
C: schema-checked architecture (linear or small MLP).
D: a typed, allow-listed scalar update-rule DSL.

Level E meta-policy is implemented separately in `meta_policy.py`; it selects
among candidates using training-side meta-features and never receives sealed
examples or per-example sealed outcomes.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping

from ..errors import ValidationError
from ..util import canonical_json
from . import dsl

GENOME_SCHEMA_VERSION = "2.0.0"
LEVELS = ("A", "B", "C", "D")

DEFAULT_MODEL = {
    "kind": "linear",
    "hidden": [],
    "activation": "tanh",
}

DEFAULT_OPTIMIZER = {
    "kind": "sgd",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
}

DEFAULT_RECIPE = {
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
    "lr_schedule": "constant",
}

DEFAULT_DSL = {
    "version": dsl.DSL_VERSION,
    "w_update": None,
    "b_update": None,
    "m_update": None,
    "v_update": None,
}

OPTIMIZER_KINDS = ("sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion", "dsl")
INIT_KINDS = ("zeros", "uniform", "normal")
LOSS_KINDS = ("auto", "mse", "bce")
ACTIVATIONS = ("tanh", "relu", "sigmoid")
SEED_POLICIES = ("hash", "constant")
LR_SCHEDULES = ("constant", "cosine", "linear", "warmup_cosine")


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValidationError(
            f"{where} has wrong keys (missing={missing}, extra={extra})",
            code="schema_keys",
        )


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric", code="schema_type")
    number = float(value)
    if not math.isfinite(number) or not (low <= number <= high):
        raise ValidationError(f"{name} must be in [{low}, {high}]", code="schema_range")
    return number


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer", code="schema_type")
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be in [{low}, {high}]", code="schema_range")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be boolean", code="schema_type")
    return value


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f"{name} must be one of {choices}", code="schema_choice")
    return value


def validate_genome(
    raw: Mapping[str, Any],
    *,
    max_dsl_nodes: int = 96,
    max_dsl_depth: int = 12,
    max_genome_bytes: int = 65536,
    max_hidden_layers: int = 2,
    max_hidden_width: int = 24,
) -> dict[str, Any]:
    """Validate and normalize a genome. Returns a deep, canonical structure."""

    if not isinstance(raw, Mapping):
        raise ValidationError("genome must be an object", code="schema_type")
    _exact_keys(raw, {"schema_version", "level", "model", "optimizer", "recipe", "dsl"}, "genome")
    if raw["schema_version"] != GENOME_SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported genome schema {raw['schema_version']!r}",
            code="schema_version",
        )
    level = _choice(raw["level"], "level", LEVELS)

    model_raw = raw["model"]
    if not isinstance(model_raw, Mapping):
        raise ValidationError("model must be an object", code="schema_type")
    _exact_keys(model_raw, {"kind", "hidden", "activation"}, "model")
    kind = _choice(model_raw["kind"], "model.kind", ("linear", "mlp"))
    hidden_raw = model_raw["hidden"]
    if not isinstance(hidden_raw, list) or not all(
        isinstance(width, int) and not isinstance(width, bool) for width in hidden_raw
    ):
        raise ValidationError("model.hidden must be a list of integers", code="schema_type")
    if len(hidden_raw) > max_hidden_layers:
        raise ValidationError("too many hidden layers", code="schema_range")
    hidden = [
        _integer(width, "model.hidden[]", 1, max_hidden_width)
        for width in hidden_raw
    ]
    if kind == "linear" and hidden:
        raise ValidationError("linear model must have empty hidden list", code="schema_shape")
    if kind == "mlp" and not hidden:
        raise ValidationError("mlp model needs at least one hidden layer", code="schema_shape")
    activation = _choice(model_raw["activation"], "model.activation", ACTIVATIONS)

    opt_raw = raw["optimizer"]
    if not isinstance(opt_raw, Mapping):
        raise ValidationError("optimizer must be an object", code="schema_type")
    _exact_keys(opt_raw, {"kind", "beta1", "beta2", "epsilon"}, "optimizer")
    opt_kind = _choice(opt_raw["kind"], "optimizer.kind", OPTIMIZER_KINDS)
    beta1 = _number(opt_raw["beta1"], "optimizer.beta1", 0.0, 0.999)
    beta2 = _number(opt_raw["beta2"], "optimizer.beta2", 0.9, 0.999999)
    epsilon = _number(opt_raw["epsilon"], "optimizer.epsilon", 1e-12, 1e-3)

    recipe_raw = raw["recipe"]
    if not isinstance(recipe_raw, Mapping):
        raise ValidationError("recipe must be an object", code="schema_type")
    _exact_keys(
        recipe_raw,
        {
            "learning_rate",
            "batch_size",
            "epochs",
            "l2",
            "l1",
            "init",
            "init_scale",
            "loss",
            "standardize",
            "grad_clip",
            "early_stopping_patience",
            "seed_policy",
            "lr_schedule",
        },
        "recipe",
    )
    recipe = {
        "learning_rate": _number(recipe_raw["learning_rate"], "recipe.learning_rate", 1e-4, 1.0),
        "batch_size": _integer(recipe_raw["batch_size"], "recipe.batch_size", 4, 128),
        "epochs": _integer(recipe_raw["epochs"], "recipe.epochs", 2, 80),
        "l2": _number(recipe_raw["l2"], "recipe.l2", 0.0, 0.25),
        "l1": _number(recipe_raw["l1"], "recipe.l1", 0.0, 0.05),
        "init": _choice(recipe_raw["init"], "recipe.init", INIT_KINDS),
        "init_scale": _number(recipe_raw["init_scale"], "recipe.init_scale", 0.001, 2.0),
        "loss": _choice(recipe_raw["loss"], "recipe.loss", LOSS_KINDS),
        "standardize": _boolean(recipe_raw["standardize"], "recipe.standardize"),
        "grad_clip": _number(recipe_raw["grad_clip"], "recipe.grad_clip", 0.0, 10.0),
        "early_stopping_patience": _integer(
            recipe_raw["early_stopping_patience"], "recipe.early_stopping_patience", 0, 20
        ),
        "seed_policy": _choice(recipe_raw["seed_policy"], "recipe.seed_policy", SEED_POLICIES),
        "lr_schedule": _choice(recipe_raw["lr_schedule"], "recipe.lr_schedule", LR_SCHEDULES),
    }

    dsl_raw = raw["dsl"]
    if not isinstance(dsl_raw, Mapping):
        raise ValidationError("dsl must be an object", code="schema_type")
    _exact_keys(dsl_raw, {"version", "w_update", "b_update", "m_update", "v_update"}, "dsl")
    dsl_version = dsl_raw["version"]
    if dsl_version not in dsl.SUPPORTED_DSL_VERSIONS:
        raise ValidationError(f"unsupported DSL version {dsl_version!r}", code="dsl_version")
    dsl_updates: dict[str, Any] = {"version": dsl_version}
    for key in ("w_update", "b_update", "m_update", "v_update"):
        value = dsl_raw[key]
        if value is None:
            dsl_updates[key] = None
        else:
            try:
                dsl_updates[key] = dsl.validate_dsl(
                    value,
                    max_nodes=max_dsl_nodes,
                    max_depth=max_dsl_depth,
                    version=dsl_version,
                )
            except ValidationError:
                raise
            except Exception as exc:  # defensive: conversion errors are schema errors
                raise ValidationError(f"invalid DSL expression {key}: {exc}", code="dsl_invalid") from exc

    if opt_kind == "dsl":
        if dsl_updates["w_update"] is None or dsl_updates["b_update"] is None:
            raise ValidationError("DSL optimizer needs w_update and b_update", code="dsl_required")
    else:
        if any(dsl_updates[key] is not None for key in ("w_update", "b_update", "m_update", "v_update")):
            raise ValidationError("non-DSL optimizer cannot carry DSL expressions", code="dsl_unexpected")

    normalized = {
        "schema_version": GENOME_SCHEMA_VERSION,
        "level": level,
        "model": {"kind": kind, "hidden": hidden, "activation": activation},
        "optimizer": {
            "kind": opt_kind,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        },
        "recipe": recipe,
        "dsl": dsl_updates,
    }

    # Level capability checks. A level is a search-space claim, not merely a
    # label: a level-A genome may not quietly smuggle in an MLP or a DSL.
    if level == "A":
        if model_raw.get("kind") != DEFAULT_MODEL["kind"] or hidden:
            raise ValidationError("level A is limited to the linear model", code="level_scope")
        if opt_kind != "sgd":
            raise ValidationError("level A is limited to sgd", code="level_scope")
        if any(dsl_updates[key] is not None for key in ("w_update", "b_update", "m_update", "v_update")):
            raise ValidationError("level A cannot carry a DSL", code="level_scope")
        for key in ("l1", "init", "loss", "standardize", "grad_clip", "early_stopping_patience"):
            if recipe[key] != DEFAULT_RECIPE[key]:
                raise ValidationError(f"level A cannot change recipe.{key}", code="level_scope")
        for key in ("beta1", "beta2", "epsilon"):
            if abs(float(normalized["optimizer"][key]) - float(DEFAULT_OPTIMIZER[key])) > 1e-15:
                raise ValidationError(f"level A cannot change optimizer.{key}", code="level_scope")
    elif level == "B":
        if kind != "linear" or hidden:
            raise ValidationError("level B is limited to the linear model", code="level_scope")
        if any(dsl_updates[key] is not None for key in ("w_update", "b_update", "m_update", "v_update")):
            raise ValidationError("level B cannot carry a DSL", code="level_scope")
    elif level == "C":
        if any(dsl_updates[key] is not None for key in ("w_update", "b_update", "m_update", "v_update")):
            raise ValidationError("level C cannot carry a DSL", code="level_scope")
    elif level == "D":
        if opt_kind != "dsl":
            raise ValidationError("level D requires the DSL optimizer", code="level_scope")

    encoded = canonical_json(normalized).encode("utf-8")
    if len(encoded) > max_genome_bytes:
        raise ValidationError(f"genome is {len(encoded)} bytes; limit is {max_genome_bytes}", code="too_large")
    return normalized


class Genome:
    """Immutable-ish validated genome wrapper with stable hash and metrics."""

    __slots__ = ("data", "_hash")

    def __init__(self, data: Mapping[str, Any], **validate_kwargs: Any) -> None:
        self.data = validate_genome(data, **validate_kwargs)
        self._hash = hashlib.sha256(canonical_json(self.data).encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def level(self) -> str:
        return self.data["level"]

    @property
    def short_id(self) -> str:
        return self._hash[:12]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def canonical_json(self) -> str:
        return canonical_json(self.data)

    def with_level(self, level: str, **validate_kwargs: Any) -> "Genome":
        """Upgrade (or relabel) a genome for curriculum crossover.

        A downgrade is allowed only if the resulting genome satisfies the
        lower level's restricted grammar; validation enforces that.
        """

        changed = self.to_dict()
        changed["level"] = level
        return Genome(changed, **validate_kwargs)

    def structural_signature(self) -> str:
        # Exact duplicates are the only things deduplicated. Two genomes with
        # the same score but different recipes remain distinct evidence.
        return self.hash

    def complexity(self) -> float:
        """A bounded complexity proxy for Pareto selection (not a truth claim)."""

        hidden = self.data["model"]["hidden"]
        arch = 1.0 + sum(width * width for width in hidden) / 128.0
        dsl_nodes = sum(
            dsl.node_count(self.data["dsl"][key])
            for key in ("w_update", "b_update", "m_update", "v_update")
            if self.data["dsl"][key] is not None
        )
        return round(arch + dsl_nodes / 24.0 + 0.25, 6)

    def parameter_estimate(self, n_features: int) -> int:
        hidden = self.data["model"]["hidden"]
        if not hidden:
            return int(n_features + 1)
        total = 0
        prev = n_features
        for width in hidden:
            total += prev * width + width
            prev = width
        return int(total + prev + 1)

    def describe(self) -> dict[str, Any]:
        d = self.to_dict()
        return {
            "schema_version": d["schema_version"],
            "level": d["level"],
            "model": d["model"],
            "optimizer": d["optimizer"]["kind"],
            "learning_rate": d["recipe"]["learning_rate"],
            "epochs": d["recipe"]["epochs"],
            "hidden": d["model"]["hidden"],
            "dsl_nodes": sum(
                dsl.node_count(d["dsl"][key])
                for key in ("w_update", "b_update", "m_update", "v_update")
                if d["dsl"][key] is not None
            ),
        }
