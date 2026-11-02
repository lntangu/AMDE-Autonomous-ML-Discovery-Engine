"""Genome schema, level scopes, and safe genetic operators."""

from __future__ import annotations

import random
import unittest

from amde.discovery import dsl
from amde.discovery.factory import default_genome, random_genome, upgrade_genome
from amde.discovery.genome import Genome, validate_genome
from amde.discovery.operators import crossover, mutate
from amde.errors import ValidationError

SAFETY = {
    "max_dsl_nodes": 96,
    "max_dsl_depth": 12,
    "max_genome_bytes": 65536,
    "max_hidden_layers": 2,
    "max_hidden_width": 24,
}


class GenomeTests(unittest.TestCase):
    def test_random_genomes_are_valid_at_every_level(self) -> None:
        rng = random.Random(11)
        for level in ("A", "B", "C", "D"):
            seen = set()
            for _ in range(25):
                genome = random_genome(level, rng, safety=SAFETY)
                self.assertEqual(genome.level, level)
                self.assertRegex(genome.hash, r"^[a-f0-9]{64}$")
                seen.add(genome.hash)
                validated = Genome(genome.to_dict(), **SAFETY)
                self.assertEqual(validated.hash, genome.hash)
            self.assertGreater(len(seen), 1)
            self.assertGreater(genome.complexity(), 1.0)

    def test_level_a_scope_is_enforced(self) -> None:
        data = default_genome("A")
        data["model"] = {"kind": "mlp", "hidden": [4], "activation": "tanh"}
        with self.assertRaises(ValidationError):
            validate_genome(data, **SAFETY)

        data = default_genome("A")
        data["optimizer"]["kind"] = "adam"
        with self.assertRaises(ValidationError):
            validate_genome(data, **SAFETY)

        data = default_genome("A")
        data["recipe"]["standardize"] = False
        with self.assertRaises(ValidationError):
            validate_genome(data, **SAFETY)

    def test_level_b_and_c_cannot_carry_dsl(self) -> None:
        for level in ("B", "C"):
            data = default_genome(level)
            data["dsl"].update(dsl.dsl_template("sgd"))
            with self.assertRaises(ValidationError):
                validate_genome(data, **SAFETY)

    def test_level_d_requires_dsl(self) -> None:
        data = default_genome("D")
        with self.assertRaises(ValidationError):
            validate_genome(data, **SAFETY)
        data["optimizer"]["kind"] = "dsl"
        data["dsl"].update(dsl.dsl_template("sgd"))
        validate_genome(data, **SAFETY)

    def test_non_dsl_optimizer_cannot_smuggle_dsl(self) -> None:
        data = default_genome("B")
        data["optimizer"]["kind"] = "adam"
        data["dsl"]["w_update"] = {"op": "var", "name": "w"}
        with self.assertRaises(ValidationError):
            validate_genome(data, **SAFETY)

    def test_safety_limits_are_enforced(self) -> None:
        data = default_genome("C")
        data["model"] = {"kind": "mlp", "hidden": [4, 4, 4], "activation": "tanh"}
        with self.assertRaises(ValidationError):
            validate_genome(data, max_hidden_layers=2)
        data["model"] = {"kind": "mlp", "hidden": [40], "activation": "tanh"}
        with self.assertRaises(ValidationError):
            validate_genome(data, max_hidden_width=24)

    def test_hash_changes_with_content_and_is_stable(self) -> None:
        first = Genome(default_genome("A"))
        self.assertEqual(first.hash, Genome(default_genome("A")).hash)
        changed = default_genome("A")
        changed["recipe"]["learning_rate"] = 0.123
        self.assertNotEqual(first.hash, Genome(changed).hash)

    def test_parameter_estimate(self) -> None:
        linear = Genome(default_genome("A"))
        self.assertEqual(linear.parameter_estimate(5), 6)
        mlp_data = default_genome("C")
        mlp_data["model"] = {"kind": "mlp", "hidden": [4, 3], "activation": "tanh"}
        mlp = Genome(mlp_data)
        self.assertEqual(mlp.parameter_estimate(5), (5 * 4 + 4) + (4 * 3 + 3) + 4)

    def test_operators_always_emit_valid_genomes(self) -> None:
        rng = random.Random(23)
        seeds = {level: random_genome(level, rng, safety=SAFETY) for level in ("A", "B", "C", "D")}
        for level in ("A", "B", "C", "D"):
            for _ in range(12):
                child = mutate(seeds[level], rng, target_level=level, safety=SAFETY)
                Genome(child.to_dict(), **SAFETY)
                self.assertEqual(child.level, level)
        for left_level in ("A", "B", "C", "D"):
            for right_level in ("A", "B", "C", "D"):
                child = crossover(
                    seeds[left_level],
                    seeds[right_level],
                    rng,
                    safety=SAFETY,
                )
                Genome(child.to_dict(), **SAFETY)
                self.assertIn(child.level, ("A", "B", "C", "D"))

    def test_upgrade_preserves_or_adds_valid_capabilities(self) -> None:
        rng = random.Random(5)
        base = random_genome("A", rng, safety=SAFETY)
        upgraded = upgrade_genome(base, "D", rng, safety=SAFETY)
        self.assertEqual(upgraded.level, "D")
        self.assertEqual(upgraded.data["optimizer"]["kind"], "dsl")
        self.assertIsNotNone(upgraded.data["dsl"]["w_update"])
        self.assertIsNotNone(upgraded.data["dsl"]["b_update"])


if __name__ == "__main__":
    unittest.main()
