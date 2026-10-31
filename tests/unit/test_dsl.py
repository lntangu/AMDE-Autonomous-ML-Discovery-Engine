"""Safety properties of the scalar update-rule DSL."""

from __future__ import annotations

import random
import unittest

from amde.discovery import dsl
from amde.errors import DSLValidationError


class DSLTests(unittest.TestCase):
    def test_all_templates_validate_and_include_weight_and_bias(self) -> None:
        for kind in ("sgd", "momentum", "adam", "rmsprop", "sign", "clipped"):
            with self.subTest(kind=kind):
                updates = dsl.dsl_template(kind)
                self.assertIn("w_update", updates)
                self.assertIn("b_update", updates)
                for key, expr in updates.items():
                    normalized = dsl.validate_dsl(expr)
                    self.assertEqual(normalized, dsl.validate_dsl(normalized))

    def test_unknown_operation_and_variable_are_rejected(self) -> None:
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "eval", "args": []})
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "var", "name": "__import__"})
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "add", "args": [{"op": "const", "value": 1}]})

    def test_extra_fields_and_bad_constants_are_rejected(self) -> None:
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "const", "value": 1, "call": "os.system"})
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "const", "value": True})
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "const", "value": float("inf")})
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl({"op": "const", "value": 1e12})

    def test_size_and_depth_limits_are_enforced(self) -> None:
        expr = {"op": "const", "value": 1.0}
        for _ in range(8):
            expr = {"op": "neg", "args": [expr]}
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl(expr, max_depth=4)
        with self.assertRaises(DSLValidationError):
            dsl.validate_dsl(
                {"op": "add", "args": [{"op": "const", "value": 1.0}, {"op": "const", "value": 2.0}]},
                max_nodes=2,
            )

    def test_evaluation_is_numeric_and_bounded(self) -> None:
        variables = {"g": 0.5, "p": 2.0, "w": 2.0, "b": 2.0, "m": 0.0, "v": 0.0, "t": 1.0, "lr": 0.1, "gn": 0.5}
        expr = dsl.dsl_template("sgd")["w_update"]
        self.assertAlmostEqual(dsl.evaluate(expr, variables), 1.95)
        self.assertEqual(
            dsl.evaluate({"op": "div", "args": [{"op": "const", "value": 1.0}, {"op": "const", "value": 0.0}]}, {}),
            0.0,
        )
        self.assertEqual(
            dsl.evaluate(
                {
                    "op": "clip",
                    "args": [
                        {"op": "const", "value": 5.0},
                        {"op": "const", "value": -2.0},
                        {"op": "const", "value": 2.0},
                    ],
                },
                {},
            ),
            2.0,
        )

    def test_random_and_mutated_expressions_are_valid(self) -> None:
        rng = random.Random(7)
        for _ in range(50):
            raw = dsl.random_expr(rng, depth=3, max_nodes=40)
            normalized = dsl.validate_dsl(raw, max_nodes=40, max_depth=8)
            mutated = dsl.mutate_expr(normalized, rng, max_nodes=40)
            dsl.validate_dsl(mutated, max_nodes=40, max_depth=8)

    def test_summary_counts_nodes(self) -> None:
        summary = dsl.dsl_summary(dsl.dsl_template("adam"))
        self.assertEqual(summary["version"], dsl.DSL_VERSION)
        self.assertGreater(summary["total_nodes"], 0)
        self.assertGreater(summary["max_depth"], 1)


if __name__ == "__main__":
    unittest.main()
