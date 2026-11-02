"""Unit tests for AMDE 3.0 Discovery Engine and Bottleneck Repairs.

Verifies:
- BOT-01: Generational fitness preservation via EliteArchive and calibrated mutation
- BOT-02: Zero candidate recycling via CandidateDeduplicator and canonical hashing
- BOT-03: Guided mutation and failure basin pruning via NegativeMemory
- BOT-04: Dynamic quantum circuit synthesis with zero frozen preset configurations
- BOT-05: Unbiased sealed holdout test evaluation
- EGADA Controller: Discounted Thompson Sampling and multi-fidelity scheduling
"""

import copy
import math
import random
import unittest
import numpy as np

from amde.discovery.config import EngineConfig
from amde.discovery.controller import EGADAController
from amde.discovery.engine import AMDE3DiscoveryEngine, AMDELiteEngine
from amde.discovery.evaluator import MultiFidelityEvaluator, RankCorrelationMonitor
from amde.discovery.memory import CandidateDeduplicator, EliteArchive, NegativeMemory, ParetoArchive
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash
from amde.quantum.dynamic_circuits import DynamicQuantumCircuitSpace


class TestAMDE3Discovery(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(42)
        self.search_space = SearchSpace(n_features=6, is_classification=True)

    def test_canonical_hashing_invariance(self):
        """Verifies that dictionary key order or metadata does not alter canonical hash."""
        genome_a = {
            "model": {"family": "mlp", "learning_rate": 0.01, "hidden_widths": [32, 16], "activation": "relu"},
            "preprocessing": {"scaler": "standard", "imputer": "mean"},
        }
        # Invert key orders and add ephemeral metadata
        genome_b = {
            "preprocessing": {"imputer": "mean", "scaler": "standard"},
            "model": {"activation": "relu", "hidden_widths": [32, 16], "learning_rate": 0.01, "family": "mlp"},
            "score": 0.95,
            "timestamp": 12345678,
        }
        hash_a = compute_genome_hash(genome_a)
        hash_b = compute_genome_hash(genome_b)
        self.assertEqual(hash_a, hash_b)

    def test_bot_01_and_bot_03_calibrated_mutation(self):
        """Verifies that mutation is calibrated and does not catastrophically scramble genomes."""
        genome = self.search_space.sample_random(self.rng)
        initial_lr = genome["model"].get("learning_rate", 0.01)

        mutated = self.search_space.mutate_calibrated(genome, self.rng, generation=1)
        valid, msg = self.search_space.validate_genome(mutated)
        self.assertTrue(valid, f"Mutated genome was invalid: {msg}")

        # Learning rate should not be perturbed by more than 10x in a single calibrated step
        if "learning_rate" in mutated["model"] and "learning_rate" in genome["model"]:
            ratio = mutated["model"]["learning_rate"] / initial_lr
            self.assertTrue(0.1 <= ratio <= 10.0, f"Mutation ratio too extreme: {ratio}")

    def test_bot_02_deduplication(self):
        """Verifies that candidate deduplicator catches repeated proposals (BOT-02 fix)."""
        dedup = CandidateDeduplicator()
        genome = self.search_space.sample_random(self.rng)
        c_hash = compute_genome_hash(genome)

        self.assertFalse(dedup.is_duplicate(c_hash))
        dedup.register(c_hash)
        self.assertTrue(dedup.is_duplicate(c_hash))
        self.assertGreater(dedup.duplicate_rate, 0.0)

    def test_bot_01_elite_archive_preservation(self):
        """Verifies that EliteArchive guarantees monotonic non-degradation of best score (BOT-01 fix)."""
        archive = EliteArchive(capacity=5)
        genome_1 = self.search_space.sample_random(self.rng)
        cand_1 = Candidate(genome=genome_1, canonical_hash=compute_genome_hash(genome_1))

        archive.add(cand_1.to_dict(), score=0.85)
        self.assertEqual(archive.get_best()["score"], 0.85)

        # Adding an inferior candidate does not degrade incumbent
        genome_2 = self.search_space.sample_random(self.rng)
        cand_2 = Candidate(genome=genome_2, canonical_hash=compute_genome_hash(genome_2))
        archive.add(cand_2.to_dict(), score=0.60)
        self.assertEqual(archive.get_best()["score"], 0.85)

        # Adding a superior candidate updates incumbent
        genome_3 = self.search_space.sample_random(self.rng)
        cand_3 = Candidate(genome=genome_3, canonical_hash=compute_genome_hash(genome_3))
        archive.add(cand_3.to_dict(), score=0.92)
        self.assertEqual(archive.get_best()["score"], 0.92)

    def test_bot_03_negative_memory(self):
        """Verifies that negative memory indexes failure points and rejects neighbors (BOT-03 fix)."""
        neg = NegativeMemory(radius=0.15)
        failed_genome = {
            "model": {"family": "mlp", "learning_rate": 0.5},
            "preprocessing": {"scaler": "none"},
        }
        neg.record_failure(failed_genome)
        self.assertTrue(neg.is_in_failure_basin(failed_genome))

        # A very different genome should not be in the failure basin
        safe_genome = {
            "model": {"family": "tree_ensemble", "learning_rate": 0.01},
            "preprocessing": {"scaler": "standard"},
        }
        self.assertFalse(neg.is_in_failure_basin(safe_genome))

    def test_bot_04_dynamic_quantum_circuits(self):
        """Verifies that dynamic quantum circuit space synthesizes diverse configurations (BOT-04 fix)."""
        q_space = DynamicQuantumCircuitSpace(max_qubits=8)
        hashes = set()
        for _ in range(30):
            circuit = q_space.sample_circuit(self.rng, n_features=4)
            hashes.add(circuit["circuit_hash"])

        # BOT-04 evaluated only 3 presets; dynamic space must evaluate >15 unique configurations
        self.assertGreaterEqual(len(hashes), 15, f"Expected >= 15 unique quantum circuits, got {len(hashes)}")

    def test_pareto_archive_non_domination(self):
        """Verifies multi-objective non-domination tracking."""
        pareto = ParetoArchive()
        c1 = {"canonical_hash": "h1"}
        c2 = {"canonical_hash": "h2"}
        c3 = {"canonical_hash": "h3"}

        # c1: loss 0.20, latency 5.0, mem 10.0
        pareto.update(c1, loss=0.20, latency=5.0, memory=10.0)
        # c2: loss 0.10, latency 8.0, mem 15.0 (trade-off: lower loss, higher latency)
        pareto.update(c2, loss=0.10, latency=8.0, memory=15.0)
        self.assertEqual(len(pareto.frontier), 2)

        # c3: loss 0.25, latency 9.0, mem 20.0 (strictly dominated by c1 and c2)
        added = pareto.update(c3, loss=0.25, latency=9.0, memory=20.0)
        self.assertFalse(added)
        self.assertEqual(len(pareto.frontier), 2)

    def test_egada_controller_bandit(self):
        """Verifies that EGADA controller shifts allocation toward winning strategies."""
        strategies = ["strat_a", "strat_b"]
        controller = EGADAController(strategies)

        # Reward strat_a with high improvements and strat_b with failures
        for _ in range(10):
            controller.update_feedback("strat_a", val_score=0.9, train_duration=0.5, is_novel=True, is_duplicate=False, is_failure=False)
            controller.update_feedback("strat_b", val_score=0.2, train_duration=2.0, is_novel=True, is_duplicate=False, is_failure=True)

        # Strategy A should have higher discounted reward than Strategy B
        self.assertGreater(controller.stats["strat_a"]["discounted_reward"], controller.stats["strat_b"]["discounted_reward"])

    def test_engine_end_to_end_search_and_sealed_test(self):
        """Verifies complete end-to-end discovery run and sealed holdout test isolation (BOT-05 fix)."""
        np_rng = np.random.RandomState(42)
        X = np_rng.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        X_train, y_train = X[:60], y[:60]
        X_val, y_val = X[60:80], y[60:80]
        X_test, y_test = X[80:], y[80:]

        engine = AMDE3DiscoveryEngine(n_features=4, is_classification=True, seed=42)
        res = engine.search(X_train, y_train, X_val, y_val, budget=10, use_multi_fidelity=True)

        self.assertEqual(res["evaluations_completed"], 10)
        self.assertGreater(res["incumbent_score"], 0.50)
        self.assertIsNotNone(res["incumbent_candidate"])

        # Evaluate best candidate on sealed test set
        test_score = engine.evaluate_sealed_test(X_train, y_train, X_test, y_test, res["incumbent_candidate"])
        self.assertIsInstance(test_score, float)
        self.assertGreaterEqual(test_score, 0.0)

    def test_engine_config_presets(self):
        """Verifies evidence-driven feature flags and default presets."""
        lite_cfg = EngineConfig.amde_30_lite_default()
        self.assertFalse(lite_cfg.enable_surrogate)
        self.assertFalse(lite_cfg.enable_negative_memory)
        self.assertFalse(lite_cfg.use_multi_fidelity)
        self.assertTrue(lite_cfg.enable_controller)
        self.assertTrue(lite_cfg.enable_deduplication)
        self.assertTrue(lite_cfg.enable_elite_archive)

        exp_cfg = EngineConfig.amde_30_full_experimental()
        self.assertTrue(exp_cfg.enable_surrogate)
        self.assertTrue(exp_cfg.enable_negative_memory)
        self.assertTrue(exp_cfg.use_multi_fidelity)

    def test_amde_lite_engine_defaults(self):
        """Verifies AMDELiteEngine instantiates with Lite presets."""
        lite = AMDELiteEngine(n_features=6, is_classification=True, seed=42)
        self.assertFalse(lite.config.enable_surrogate)
        self.assertFalse(lite.config.enable_negative_memory)
        self.assertFalse(lite.config.use_multi_fidelity)
        self.assertNotIn("surrogate_ei", lite.strategies)
        self.assertIn("random_search", lite.strategies)
        self.assertIn("evolutionary_elitist", lite.strategies)
        self.assertIn("local_refinement", lite.strategies)

    def test_controller_finite_initial_reward(self):
        """Verifies that the first evaluated candidate receives finite reward and does not skew TS."""
        ctrl = EGADAController(["strat_a", "strat_b"], discount_factor=0.95)
        r = ctrl.update_feedback("strat_a", val_score=0.85, train_duration=0.5, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertFalse(math.isinf(r), f"Initial reward was infinite: {r}")
        self.assertFalse(math.isnan(r), f"Initial reward was NaN: {r}")
        self.assertFalse(math.isinf(ctrl.stats["strat_a"]["discounted_reward"]))

    def test_controller_replay_determinism(self):
        """Verifies bit-for-bit identical decision streams across identical seeds."""
        strats = ["s1", "s2", "s3"]
        decisions_1 = []
        ctrl1 = EGADAController(strats, discount_factor=0.95)
        rng1 = random.Random(1234)
        for step in range(25):
            d = ctrl1.select_strategy(rng1, budget_fraction=step / 25.0)
            decisions_1.append(d)
            ctrl1.update_feedback(d, val_score=0.75, train_duration=0.2, is_novel=True, is_duplicate=False, is_failure=False)

        decisions_2 = []
        ctrl2 = EGADAController(strats, discount_factor=0.95)
        rng2 = random.Random(1234)
        for step in range(25):
            d = ctrl2.select_strategy(rng2, budget_fraction=step / 25.0)
            decisions_2.append(d)
            ctrl2.update_feedback(d, val_score=0.75, train_duration=0.2, is_novel=True, is_duplicate=False, is_failure=False)

        self.assertEqual(decisions_1, decisions_2)

    def test_canonical_hash_invariance_property(self):
        """Property-based verification of canonical hash invariance under permutation."""
        space = SearchSpace(n_features=8, is_classification=True)
        rng = random.Random(999)
        for _ in range(15):
            cand = space.sample_random(rng)
            h_orig = compute_genome_hash(cand)

            # Invert outer dictionary keys
            cand_permuted = {k: copy.deepcopy(cand[k]) for k in reversed(list(cand.keys()))}
            h_perm = compute_genome_hash(cand_permuted)
            self.assertEqual(h_orig, h_perm)


if __name__ == "__main__":
    unittest.main()
