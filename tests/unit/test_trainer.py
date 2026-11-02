"""Deterministic trainer behavior on real generated data."""

from __future__ import annotations

import random
import math
import unittest
import importlib.util

from amde.benchmarks.base import generate_task
from amde.benchmarks.generators import CHALLENGE_GENERATORS, GENERATORS
from amde.discovery import dsl
from amde.discovery.factory import default_genome, random_genome
from amde.discovery.genome import Genome
from amde.errors import ValidationError
from amde.evaluation.metrics import classification_score, regression_score
from amde.models.trainer import (
    Layer, OptimizerState, _apply_updates, baseline_majority_mean, train_and_predict,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

SPLIT = {"train": 32, "val": 16, "dev": 16, "challenge": 8, "duplicate_rounding": 8}


def generated(generator: str, kind: str, n_features: int, seed: int):
    return generate_task(
        {
            "task_id": f"{generator}_{seed}",
            "suite": "unit",
            "family": generator,
            "kind": kind,
            "generator": generator,
            "n_features": n_features,
            "seed": seed,
        },
        split_config=SPLIT,
        generators=GENERATORS,
        challenge_generators=CHALLENGE_GENERATORS,
    )


def train(genome: Genome, task, *, seed: int = 1, epochs_scale: float = 1.0):
    return train_and_predict(
        genome,
        train_x=[list(example.x) for example in task.train],
        train_y=[example.y for example in task.train],
        val_x=[list(example.x) for example in task.val],
        val_y=[example.y for example in task.val],
        dev_x=[list(example.x) for example in task.dev],
        task_kind=task.kind,
        eval_seed=seed,
        task_id=task.task_id,
        epochs_scale=epochs_scale,
        max_steps=600,
        parameter_limit=20000,
    )


class TrainerTests(unittest.TestCase):
    def test_linear_regression_learns_and_is_deterministic(self) -> None:
        task = generated("linear_regression", "regression", 3, 101)
        genome = Genome(default_genome("A"))
        first = train(genome, task)
        second = train(genome, task)
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["steps"], second["steps"])
        score = regression_score([example.y for example in task.dev], first["predictions"])
        self.assertGreater(score["score"], 0.3)
        self.assertTrue(all(isinstance(value, float) for value in first["predictions"]))

    def test_mlp_classification_beats_chance(self) -> None:
        task = generated("blobs_classification", "classification", 3, 104)
        rng = random.Random(3)
        genome = random_genome("C", rng)
        data = genome.to_dict()
        data["model"] = {"kind": "mlp", "hidden": [8], "activation": "tanh"}
        data["optimizer"]["kind"] = "adam"
        data["recipe"]["learning_rate"] = 0.03
        data["recipe"]["epochs"] = 40
        genome = Genome(data)
        output = train(genome, task)
        score = classification_score([example.y for example in task.dev], output["predictions"])
        self.assertGreater(score["balanced_accuracy"], 0.6)
        self.assertGreater(output["parameter_count"], 3)

    def test_dsl_update_rule_trains_without_numerical_failure(self) -> None:
        task = generated("linear_regression", "regression", 2, 202)
        data = default_genome("D")
        data["optimizer"]["kind"] = "dsl"
        data["dsl"].update(dsl.dsl_template("sgd"))
        data["recipe"]["epochs"] = 20
        output = train(Genome(data), task)
        self.assertEqual(len(output["predictions"]), len(task.dev))
        self.assertTrue(all(value == value for value in output["predictions"]))
        score = regression_score([example.y for example in task.dev], output["predictions"])
        self.assertGreater(score["score"], 0.0)

    def test_parameter_limit_and_empty_splits_fail_closed(self) -> None:
        task = generated("linear_regression", "regression", 3, 303)
        with self.assertRaises(ValidationError):
            train_and_predict(
                Genome(default_genome("A")),
                train_x=[list(example.x) for example in task.train],
                train_y=[example.y for example in task.train],
                val_x=[list(example.x) for example in task.val],
                val_y=[example.y for example in task.val],
                dev_x=[list(example.x) for example in task.dev],
                task_kind=task.kind,
                eval_seed=1,
                task_id=task.task_id,
                parameter_limit=1,
            )
        with self.assertRaises(ValidationError):
            train_and_predict(
                Genome(default_genome("A")),
                train_x=[],
                train_y=[],
                val_x=[list(example.x) for example in task.val],
                val_y=[example.y for example in task.val],
                dev_x=[list(example.x) for example in task.dev],
                task_kind=task.kind,
                eval_seed=1,
                task_id=task.task_id,
            )


    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional")
    def test_pytorch_cpu_backend_matches_reference_for_linear_mlp_and_dsl(self) -> None:
        from amde.models.torch_trainer import train_and_predict_torch

        cases = [
            ("A", "linear_regression", "regression", 3, 501),
            ("C", "blobs_classification", "classification", 3, 502),
            ("D", "linear_regression", "regression", 2, 503),
        ]
        for level, generator, kind, n_features, seed in cases:
            with self.subTest(level=level, generator=generator):
                task = generated(generator, kind, n_features, seed)
                data = default_genome(level)
                if level == "D":
                    data["optimizer"]["kind"] = "dsl"
                    data["dsl"].update(dsl.dsl_template("sgd"))
                    data["recipe"]["epochs"] = 20
                genome = Genome(data)
                kwargs = dict(
                    train_x=[list(example.x) for example in task.train],
                    train_y=[example.y for example in task.train],
                    val_x=[list(example.x) for example in task.val],
                    val_y=[example.y for example in task.val],
                    dev_x=[list(example.x) for example in task.dev],
                    task_kind=task.kind,
                    eval_seed=1,
                    task_id=task.task_id,
                    epochs_scale=0.2,
                    max_steps=600,
                    parameter_limit=20000,
                )
                reference = train_and_predict(genome, **kwargs)
                tensor = train_and_predict_torch(genome, **kwargs, device="cpu")
                self.assertEqual(len(reference["predictions"]), len(tensor["predictions"]))
                max_delta = max(
                    abs(a - b) for a, b in zip(reference["predictions"], tensor["predictions"])
                )
                self.assertLess(max_delta, 1e-8)
                self.assertEqual(reference["steps"], tensor["steps"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional")
    def test_pytorch_cpu_backend_matches_reference_across_optimizers(self) -> None:
        from amde.models.torch_trainer import train_and_predict_torch

        task = generated("linear_regression", "regression", 3, 704)
        kwargs = dict(
            train_x=[list(example.x) for example in task.train],
            train_y=[example.y for example in task.train],
            val_x=[list(example.x) for example in task.val],
            val_y=[example.y for example in task.val],
            dev_x=[list(example.x) for example in task.dev],
            task_kind=task.kind,
            eval_seed=17,
            task_id=task.task_id,
            epochs_scale=0.1,
            max_steps=100,
            parameter_limit=20000,
        )
        for optimizer in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion", "dsl"):
            with self.subTest(optimizer=optimizer):
                level = "D" if optimizer == "dsl" else "B"
                data = default_genome(level)
                data["optimizer"]["kind"] = optimizer
                data["recipe"]["lr_schedule"] = "cosine"
                if optimizer == "dsl":
                    data["dsl"].update(dsl.dsl_template("adam"))
                genome = Genome(data)
                reference = train_and_predict(genome, **kwargs)
                tensor = train_and_predict_torch(genome, **kwargs, device="cpu")
                self.assertEqual(reference["steps"], tensor["steps"])
                self.assertLess(
                    max(abs(a - b) for a, b in zip(reference["predictions"], tensor["predictions"])),
                    1e-7,
                )

    def test_adagrad_and_nadam_are_valid_searchable_optimizer_methods(self) -> None:
        for optimizer in ("adagrad", "nadam"):
            with self.subTest(optimizer=optimizer):
                data = default_genome("B")
                data["optimizer"]["kind"] = optimizer
                genome = Genome(data)
                task = generated("linear_regression", "regression", 3, 1204)
                result = train_and_predict(
                    genome,
                    train_x=[list(example.x) for example in task.train],
                    train_y=[example.y for example in task.train],
                    val_x=[list(example.x) for example in task.val],
                    val_y=[example.y for example in task.val],
                    dev_x=[list(example.x) for example in task.dev],
                    task_kind=task.kind, eval_seed=19, task_id=task.task_id,
                    epochs_scale=0.1, max_steps=100, parameter_limit=20000,
                )
                self.assertTrue(all(value == value for value in result["predictions"]))

    def test_learning_rate_schedules_are_bounded_and_monotone(self) -> None:
        from amde.models.trainer import _schedule_scale
        for schedule in ("constant", "cosine", "linear", "warmup_cosine"):
            values = [_schedule_scale(schedule, step, 100) for step in range(1, 101)]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
            self.assertEqual(values[0], 1.0 if schedule != "warmup_cosine" else 0.05)
            self.assertLessEqual(values[-1], values[0])
        self.assertEqual(_schedule_scale("constant", 100, 100), 1.0)

    def test_majority_baseline_is_parameter_free_and_real(self) -> None:
        task = generated("blobs_classification", "classification", 2, 404)
        predictions = baseline_majority_mean(
            train_x=[list(example.x) for example in task.train],
            train_y=[example.y for example in task.train],
            dev_x=[list(example.x) for example in task.dev],
            task_kind=task.kind,
        )
        self.assertEqual(len(predictions), len(task.dev))
        self.assertEqual(len(set(predictions)), 1)
        self.assertIn(predictions[0], (0.0, 1.0))

    def test_nadam_uses_nesterov_momentum_formula(self) -> None:
        data = default_genome("B")
        data["optimizer"].update({"kind": "nadam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8})
        data["recipe"].update({"learning_rate": 0.1, "l1": 0.0, "l2": 0.0})
        layer = Layer(W=[[1.0]], b=[0.0], activation=None)
        state = OptimizerState([layer], "nadam")
        gradient = 2.0
        _apply_updates([layer], [[[gradient]]], [[0.0]], data, state, 2.0, 1)

        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.1
        moment = (1.0 - beta1) * gradient
        variance = (1.0 - beta2) * gradient * gradient
        nesterov_moment = (
            beta1 * moment / (1.0 - beta1**2)
            + (1.0 - beta1) * gradient / (1.0 - beta1)
        )
        corrected_variance = variance / (1.0 - beta2)
        expected = 1.0 - lr * nesterov_moment / (math.sqrt(corrected_variance) + eps)
        self.assertAlmostEqual(layer.W[0][0], expected, places=12)


if __name__ == "__main__":
    unittest.main()
