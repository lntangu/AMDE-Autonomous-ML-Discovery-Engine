"""Proposal operators, UCB portfolio, and the experimental meta-policy."""

from __future__ import annotations

import random
import unittest

from amde.search.meta_policy import MetaPolicy
from amde.search.population import ArchiveManager, Population
from amde.search.portfolio import Portfolio
from amde.search.strategies import STRATEGIES, Proposer
from tests.support import make_config

SAFETY = {
    "max_dsl_nodes": 96,
    "max_dsl_depth": 12,
    "max_genome_bytes": 65536,
    "max_hidden_layers": 2,
    "max_hidden_width": 24,
}


class SearchTests(unittest.TestCase):
    def test_portfolio_tries_every_arm_before_exploiting(self) -> None:
        portfolio = Portfolio(list(STRATEGIES), exploration_floor=0.0)
        rng = random.Random(1)
        pulled = []
        for _ in range(len(STRATEGIES)):
            name = portfolio.select(rng)
            pulled.append(name)
            portfolio.update(name, 0.1)
        self.assertEqual(set(pulled), set(STRATEGIES))
        portfolio.update("crossover", 1.0)
        portfolio.update("crossover", 1.0)
        self.assertEqual(portfolio.best(), "crossover")
        restored = Portfolio.from_dict(portfolio.to_dict())
        self.assertEqual(restored.total_pulls, portfolio.total_pulls)

    def test_meta_policy_is_disabled_by_default(self) -> None:
        config = make_config(self)
        policy = MetaPolicy(config)
        self.assertFalse(policy.enabled)
        self.assertEqual(policy.status, "disabled")
        portfolio = Portfolio(list(STRATEGIES))
        chosen, note = policy.choose(portfolio, random.Random(1), {"n_tasks": 2})
        self.assertIn(chosen, STRATEGIES)
        self.assertEqual(note["mode"], "disabled")
        self.assertIsNone(note["prior"])

    def test_experimental_meta_policy_prior_is_normalized(self) -> None:
        config = make_config(self)
        config.set("search.meta_policy_enabled", True)
        policy = MetaPolicy(config)
        features = {
            "n_tasks": 12,
            "mean_n_features": 6.0,
            "max_n_features": 9.0,
            "classification_ratio": 0.8,
            "n_families": 7,
        }
        prior = policy.prior(features)
        self.assertAlmostEqual(sum(prior.values()), 1.0)
        chosen, note = policy.choose(Portfolio(list(STRATEGIES)), random.Random(4), features)
        self.assertIn(chosen, STRATEGIES)
        self.assertEqual(note["mode"], "experimental_prior")
        self.assertEqual(set(note["prior"]), set(STRATEGIES))

    def test_proposer_exercises_all_strategies_even_when_empty(self) -> None:
        rng = random.Random(9)
        population = Population(capacity=12, islands=2, elite_size=2, safety=SAFETY)
        archives = ArchiveManager()
        proposer = Proposer(
            population=population,
            archives=archives,
            rng=rng,
            levels=["A", "B", "C", "D"],
            safety=SAFETY,
        )
        operators = set()
        levels = set()
        for index, strategy in enumerate(STRATEGIES):
            individual = proposer.propose(
                generation=0,
                strategy=strategy,
                level_override=("D" if strategy == "open_ended" else ["A", "B", "C", "D"][index % 4]),
            )
            operators.add(individual.operator)
            levels.add(individual.level)
            population.seen.add(individual.hash)
            individual.quick_score = 0.5 + index * 0.01
            population.add(individual)
        self.assertEqual(
            len(operators),
            len(STRATEGIES),
        )
        self.assertTrue(any(operator.startswith("open_ended_grammar") for operator in operators))
        self.assertEqual(levels, {"A", "B", "C", "D"})
        for strategy in STRATEGIES:
            individual = proposer.propose(generation=1, strategy=strategy)
            self.assertEqual(individual.level in {"A", "B", "C", "D"}, True)
            self.assertGreaterEqual(len(individual.parents), 0)

    def test_unknown_strategy_is_rejected(self) -> None:
        proposer = Proposer(
            population=Population(capacity=4, islands=1, elite_size=1, safety=SAFETY),
            archives=ArchiveManager(),
            rng=random.Random(0),
            levels=["A"],
            safety=SAFETY,
        )
        with self.assertRaises(ValueError):
            proposer.propose(generation=0, strategy="magic")


if __name__ == "__main__":
    unittest.main()
