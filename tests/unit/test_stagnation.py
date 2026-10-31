from __future__ import annotations

import unittest

from amde.discovery.factory import default_genome
from amde.discovery.genome import Genome
from amde.search.population import Individual
from amde.search.stagnation import analyze_stagnation


class StagnationTests(unittest.TestCase):
    def test_plateau_report_is_passive_and_actionable(self) -> None:
        item = Individual(genome=Genome(default_genome("A")), generation=2)
        report = analyze_stagnation(
            plateau_generations=5,
            patience=5,
            population=[item, item],
            proposals=6,
            valid=2,
            invalid=4,
            best_score=0.5,
        )
        self.assertTrue(report["is_stagnated"])
        self.assertTrue(report["automatic_intervention"] is False)
        self.assertLess(report["phenotype_diversity"], 1.0)
        self.assertGreater(report["invalid_rate"], 0.5)
        self.assertTrue(report["recommendations"])

    def test_no_plateau_does_not_emit_intervention_recommendation(self) -> None:
        report = analyze_stagnation(
            plateau_generations=0,
            patience=5,
            population=[],
            proposals=0,
            valid=0,
            invalid=0,
            best_score=None,
        )
        self.assertFalse(report["is_stagnated"])
        self.assertEqual(report["recommendations"], [])
        self.assertEqual(report["phenotype_diversity"], 0.0)


if __name__ == "__main__":
    unittest.main()
