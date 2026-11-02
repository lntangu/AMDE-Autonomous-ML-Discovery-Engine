from __future__ import annotations

import random
import unittest

from amde.discovery import dsl
from amde.search.open_ended import (
    NoveltyLedger,
    behaviour_signature,
    generate_open_ended_genome,
    growth_budget,
)
from amde.search.population import ArchiveManager, Population
from amde.search.strategies import Proposer

SAFETY = {
    "max_dsl_nodes": 256,
    "max_dsl_depth": 24,
    "max_genome_bytes": 131072,
    "max_hidden_layers": 4,
    "max_hidden_width": 64,
}


class OpenEndedSearchTests(unittest.TestCase):
    def test_growth_budget_is_monotone_and_never_exceeds_safety_caps(self):
        budgets = [growth_budget(g, SAFETY) for g in range(40)]
        self.assertEqual([b.max_nodes for b in budgets], sorted(b.max_nodes for b in budgets))
        self.assertEqual([b.max_depth for b in budgets], sorted(b.max_depth for b in budgets))
        self.assertLessEqual(max(b.max_nodes for b in budgets), SAFETY["max_dsl_nodes"])
        self.assertLessEqual(max(b.max_depth for b in budgets), SAFETY["max_dsl_depth"])
        self.assertGreater(budgets[-1].max_nodes, budgets[0].max_nodes)

    def test_open_ended_genomes_validate_and_grow(self):
        small, small_budget = generate_open_ended_genome(
            random.Random(1), safety=SAFETY, generation=0
        )
        large, large_budget = generate_open_ended_genome(
            random.Random(2), safety=SAFETY, generation=18
        )
        self.assertEqual(small.level, "D")
        self.assertEqual(large.level, "D")
        self.assertEqual(small.data["dsl"]["version"], "dsl-v2")
        self.assertEqual(large.data["dsl"]["version"], "dsl-v2")
        self.assertGreater(large_budget.max_nodes, small_budget.max_nodes)
        for genome, budget in ((small, small_budget), (large, large_budget)):
            for key in ("w_update", "b_update", "m_update", "v_update"):
                expr = genome.data["dsl"][key]
                if expr is not None:
                    self.assertLessEqual(dsl.node_count(expr), budget.max_nodes)
                    self.assertLessEqual(dsl.node_depth(expr), SAFETY["max_dsl_depth"])

    def test_random_expression_generator_never_exceeds_node_budget(self):
        for seed in range(300):
            expression = dsl.random_expr(random.Random(seed), depth=24, max_nodes=256)
            self.assertLessEqual(dsl.node_count(expression), 256, msg=f"seed={seed}")
            self.assertLessEqual(dsl.node_depth(expression), 24, msg=f"seed={seed}")

    def test_novelty_ledger_uses_exact_payload_and_diagnostic_signature(self):
        genome, _ = generate_open_ended_genome(random.Random(5), safety=SAFETY, generation=4)
        ledger = NoveltyLedger()
        signature = behaviour_signature(genome, probes=16)
        self.assertTrue(ledger.add(genome, signature))
        self.assertFalse(ledger.add(genome, signature))
        self.assertEqual(ledger.probable_behavioural_collisions(signature), [genome.hash])
        self.assertEqual(len(ledger), 1)

    def test_proposer_open_ended_arm_returns_unseen_candidate(self):
        population = Population(capacity=8, islands=2, elite_size=2, safety=SAFETY)
        proposer = Proposer(
            population=population,
            archives=ArchiveManager(),
            rng=random.Random(17),
            levels=["A", "B", "C", "D"],
            safety=SAFETY,
        )
        individual = proposer.propose(generation=8, strategy="open_ended", level_override="D")
        self.assertEqual(individual.level, "D")
        self.assertTrue(individual.operator.startswith("open_ended_grammar"))
        self.assertNotIn(individual.hash, population.seen)
        population.seen.add(individual.hash)
        second = proposer.propose(generation=8, strategy="open_ended", level_override="D")
        self.assertNotEqual(individual.hash, second.hash)


if __name__ == "__main__":
    unittest.main()
