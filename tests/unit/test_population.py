"""Population, Pareto/novelty archives, and deterministic serialization."""

from __future__ import annotations

import random
import unittest

from amde.discovery.factory import random_genome
from amde.discovery.genome import Genome
from amde.search.population import (
    ArchiveManager,
    Individual,
    NoveltyArchive,
    ParetoArchive,
    Population,
    behavior_cell,
    behavior_descriptor,
)

SAFETY = {
    "max_dsl_nodes": 96,
    "max_dsl_depth": 12,
    "max_genome_bytes": 65536,
    "max_hidden_layers": 2,
    "max_hidden_width": 24,
}


def make_individual(level: str = "A", *, generation: int = 0, island: int = 0, score: float | None = None) -> Individual:
    rng = random.Random(f"{level}-{generation}-{island}-{score}")
    genome = random_genome(level, rng, safety=SAFETY)
    return Individual(
        genome=genome,
        generation=generation,
        operator="test",
        parents=[],
        island=island,
        quick_score=score,
        status="quick_ok" if score is not None else "proposed",
    )


class PopulationTests(unittest.TestCase):
    def test_add_deduplicate_capacity_and_islands(self) -> None:
        population = Population(capacity=2, islands=2, elite_size=1, safety=SAFETY)
        first = make_individual("A", generation=1, island=0, score=0.5)
        self.assertTrue(population.add(first))
        self.assertFalse(population.add(first))
        self.assertEqual(len(population.all()), 1)
        second = make_individual("B", generation=2, island=0, score=0.9)
        third = make_individual("C", generation=3, island=0, score=0.1)
        population.add(second)
        population.add(third)
        self.assertLessEqual(len(population.island_members(0)), 2)
        self.assertEqual(len(population.islands), 2)

    def test_elites_and_parent_selection_prefer_rank(self) -> None:
        population = Population(capacity=10, islands=2, elite_size=2, safety=SAFETY)
        low = make_individual("A", generation=1, island=0, score=0.1)
        high = make_individual("B", generation=1, island=1, score=0.9)
        population.add(low)
        population.add(high)
        self.assertEqual(population.best().hash, high.hash)
        self.assertEqual(population.elites(1)[0].hash, high.hash)
        rng = random.Random(2)
        picks = {population.select_parent(rng).hash for _ in range(20)}
        self.assertIn(high.hash, picks)

    def test_confirm_update_and_serialization_roundtrip(self) -> None:
        population = Population(capacity=6, islands=2, elite_size=2, safety=SAFETY)
        individual = make_individual("D", generation=3, island=1, score=0.4)
        population.add(individual)
        aggregate = {
            "mean": 0.77,
            "n_tasks": 1,
            "n_seeds": 2,
            "metric_hash": "x",
            "per_task": {"t": {"mean": 0.77}},
        }
        population.update_confirm(individual.hash, aggregate)
        restored = Population.from_dict(population.to_dict(), safety=SAFETY)
        recovered = restored.members[individual.hash]
        self.assertAlmostEqual(recovered.confirm_score, 0.77)
        self.assertEqual(recovered.genome.hash, individual.hash)
        self.assertEqual(restored.best().hash, individual.hash)

    def test_pareto_archive_removes_dominated_entries(self) -> None:
        archive = ParetoArchive(capacity=10)
        weak = make_individual("A", generation=1, score=0.5)
        strong = make_individual("B", generation=1, score=0.8)
        weak.quick_score = 0.5
        strong.quick_score = 0.8
        self.assertTrue(archive.add(weak))
        self.assertTrue(archive.add(strong))
        # Stronger and not more complex by construction of these genomes.
        self.assertIn(strong.hash, archive.entries)
        if strong.complexity <= weak.complexity:
            self.assertNotIn(weak.hash, archive.entries)

    def test_novelty_cells_and_archive_manager(self) -> None:
        first = make_individual("A")
        second = make_individual("D", generation=1)
        descriptor_a = behavior_descriptor(first.genome)
        descriptor_d = behavior_descriptor(second.genome)
        self.assertNotEqual(behavior_cell(descriptor_a), behavior_cell(descriptor_d))
        archive = NoveltyArchive(capacity=4, k=2)
        result = archive.add(first.hash, descriptor_a, 0.5)
        self.assertTrue(result["new_cell"])
        self.assertGreater(result["novelty"], 0.0)
        manager = ArchiveManager()
        info = manager.observe(second)
        self.assertIn("cell", info)
        self.assertIn("novelty", info)
        restored = ArchiveManager.from_dict(manager.to_dict())
        self.assertEqual(restored.to_dict()["pareto"]["capacity"], manager.pareto.capacity)
        self.assertIn(second.hash, restored.novelty.descriptors)

    def test_optimizer_descriptor_remains_normalized_for_expanded_vocabulary(self) -> None:
        raw = random_genome("B", random.Random(909), safety=SAFETY).to_dict()
        observed = []
        for optimizer in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion"):
            candidate = dict(raw)
            candidate["optimizer"] = {**candidate["optimizer"], "kind": optimizer}
            descriptor = behavior_descriptor(Genome(candidate, **SAFETY))
            self.assertGreaterEqual(descriptor[6], 0.0)
            self.assertLessEqual(descriptor[6], 1.0)
            observed.append(descriptor[6])
        self.assertEqual(observed, sorted(observed))


if __name__ == "__main__":
    unittest.main()
