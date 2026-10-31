from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from amde.discovery.archive import DiscoveryArchive
from amde.discovery.genome import Genome
from amde.baselines.registry import make_baseline_genome


class DiscoveryArchiveTests(unittest.TestCase):
    def test_exact_identity_is_global_and_sightings_are_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DiscoveryArchive(Path(directory) / "archive.sqlite3")
            genome = make_baseline_genome("linear_adam")
            assert genome is not None
            first = archive.observe(genome, run_id="run_a", generation=0, operator="random",
                                    score=0.4, stage="quick", protocol_hash="a" * 64)
            second = archive.observe(genome, run_id="run_b", generation=2, operator="mutation",
                                     score=0.6, stage="confirm", protocol_hash="b" * 64)
            self.assertTrue(first["inserted"])
            self.assertFalse(second["inserted"])
            self.assertEqual(second["sightings"], 2)
            record = archive.get(genome.hash)
            self.assertEqual(record["best_quick_score"], 0.4)
            self.assertEqual(record["best_confirm_score"], 0.6)
            self.assertEqual(archive.stats()["campaign_count"], 2)

    def test_warm_start_returns_genomes_without_importing_old_scores_as_fitness(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DiscoveryArchive(Path(directory) / "archive.sqlite3")
            genome = make_baseline_genome("linear_adam")
            assert genome is not None
            archive.observe(genome, run_id="old_run", generation=4, operator="mutation",
                            stage="confirm", score=0.99)
            recovered = archive.warm_start_genomes(limit=2)
            self.assertEqual(recovered, [genome.to_dict()])

    def test_archive_concurrent_observations_do_not_lose_sightings(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DiscoveryArchive(Path(directory) / "archive.sqlite3")
            genome = make_baseline_genome("linear_sgd")
            assert genome is not None
            def add(index: int):
                return archive.observe(genome, run_id=f"run_{index}", generation=index,
                                       operator="random", score=0.2)
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(add, range(12)))
            self.assertEqual(archive.get(genome.hash)["sightings"], 12)
            self.assertEqual(archive.stats()["sighting_count"], 12)

    def test_archive_rejects_invalid_observation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DiscoveryArchive(Path(directory) / "archive.sqlite3")
            genome = make_baseline_genome("linear_sgd")
            assert genome is not None
            for kwargs, message in (
                ({"stage": "development"}, "stage must be quick or confirm"),
                ({"score": float("nan")}, "score must be finite"),
                ({"protocol_hash": "not-a-hash"}, "protocol_hash must be a 64-character"),
            ):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        archive.observe(genome, run_id="run_a", generation=0,
                                        operator="mutation", **kwargs)

    def test_archive_capacity_is_bounded_and_fails_open(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DiscoveryArchive(Path(directory) / "archive.sqlite3",
                                       max_candidates=1, max_sightings=2)
            first = make_baseline_genome("linear_sgd")
            second = make_baseline_genome("linear_adam")
            assert first is not None and second is not None
            accepted = archive.observe(first, run_id="run_a", generation=0, operator="random")
            repeated = archive.observe(first, run_id="run_a", generation=0, operator="random")
            rejected = archive.observe(second, run_id="run_b", generation=0, operator="random")
            self.assertTrue(accepted["archived"])
            self.assertTrue(repeated["archived"])
            self.assertFalse(rejected["archived"])
            self.assertEqual(archive.stats()["candidate_count"], 1)
            self.assertEqual(archive.stats()["sighting_count"], 2)


if __name__ == "__main__":
    unittest.main()
