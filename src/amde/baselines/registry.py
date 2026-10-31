"""Fixed baseline recipes.

These are actual methods trained/evaluated by the same evaluator as candidate
genomes. They are not hard-coded score constants. `majority_mean` is a
parameter-free reference; `linear_sgd` and `mlp_small` are simple but serious
baselines for the generated suites.
"""

from __future__ import annotations

from typing import Any

from ..discovery.genome import Genome

BASELINE_NAMES = ("majority_mean", "linear_sgd", "linear_adam", "linear_adagrad", "linear_nadam", "mlp_small", "mlp_medium")


def _default_genome(level: str, **overrides: Any) -> dict[str, Any]:
    data = {
        "schema_version": "2.0.0",
        "level": level,
        "model": {"kind": "linear", "hidden": [], "activation": "tanh"},
        "optimizer": {"kind": "sgd", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        "recipe": {
            "learning_rate": 0.05,
            "batch_size": 16,
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
        },
        "dsl": {
            "version": "dsl-v1",
            "w_update": None,
            "b_update": None,
            "m_update": None,
            "v_update": None,
        },
    }
    for section, changes in overrides.items():
        if section in ("model", "optimizer", "recipe"):
            data[section].update(changes)
        else:
            data[section] = changes
    return data


def make_baseline_genome(name: str) -> Genome | None:
    if name == "majority_mean":
        return None
    if name == "linear_sgd":
        return Genome(_default_genome("A"))
    if name == "linear_adam":
        return Genome(_default_genome(
            "B",
            optimizer={"kind": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            recipe={"learning_rate": 0.01, "epochs": 50, "batch_size": 32,
                     "l2": 0.0005, "lr_schedule": "cosine"},
        ))
    if name == "linear_adagrad":
        return Genome(_default_genome(
            "B",
            optimizer={"kind": "adagrad", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            recipe={"learning_rate": 0.08, "epochs": 50, "batch_size": 32,
                    "l2": 0.0005, "lr_schedule": "constant"},
        ))
    if name == "linear_nadam":
        return Genome(_default_genome(
            "B",
            optimizer={"kind": "nadam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            recipe={"learning_rate": 0.01, "epochs": 50, "batch_size": 32,
                    "l2": 0.0005, "lr_schedule": "cosine"},
        ))
    if name == "mlp_small":
        return Genome(
            _default_genome(
                "C",
                model={"kind": "mlp", "hidden": [8], "activation": "tanh"},
                optimizer={"kind": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
                recipe={
                    "learning_rate": 0.01,
                    "epochs": 35,
                    "batch_size": 16,
                    "l2": 0.0001,
                    "early_stopping_patience": 5,
                    "init_scale": 0.5,
                },
            )
        )
    if name == "mlp_medium":
        return Genome(
            _default_genome(
                "C",
                model={"kind": "mlp", "hidden": [32, 16], "activation": "relu"},
                optimizer={"kind": "adamw", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
                recipe={
                    "learning_rate": 0.003,
                    "epochs": 60,
                    "batch_size": 32,
                    "l2": 0.001,
                    "early_stopping_patience": 8,
                    "init_scale": 0.35,
                    "lr_schedule": "warmup_cosine",
                },
            ),
            max_hidden_layers=3,
            max_hidden_width=48,
        )
    raise ValueError(f"unknown baseline: {name}")
