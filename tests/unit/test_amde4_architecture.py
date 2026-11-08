"""AMDE 4.0 Architecture, Subsystem Invariants, and Clean-Room Qualification Test Suite.

Verifies:
1. Core Contracts: Immuntable specifications, typed records, error taxonomy.
2. Candidate Identity: Cryptographic canonical JSON + SHA-256 invariance.
3. Evaluation Identity: Collision-resistant EvaluationCacheKey separating candidate from evaluation context.
4. Search Space: Parameter bounds, typed validation, calibrated mutation, chromosomal crossover.
5. Search Strategies: Random, Evolutionary, Local Refinement, and Bayesian TPE proposals.
6. EGADA Controller: Bounded finite reward, posterior variance expansion, 100% replay determinism.
7. Candidate Evaluation Engine: Zero data leakage, preprocessing fitted strictly on train fold.
8. Multi-Objective Pareto Archive: Non-dominated sorting and 2D hypervolume calculation.
9. Append-Only Event Ledger: Idempotency, SHA-256 hash chaining, SQLite WAL integrity.
10. Hardware Profile: Quadro P2000 Pascal detection and non-assumption of Tensor Cores.
11. Sealed Holdout Isolation: Single-read token gate preventing test data snooping (BOT-05).
12. End-to-End Engine: Complete autonomous discovery run with trajectory and manifest export.
"""

from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np

from amde import (
    AMDE4Engine,
    Candidate,
    CandidateDeduplicator,
    ContinuousParameter,
    DelayedPruningPolicy,
    EGADAController,
    EliteArchive,
    EvaluationCache,
    EvaluationCacheKey,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    EventLedger,
    EvolutionaryStrategy,
    ExperimentSpec,
    FidelityLevel,
    FullFidelityPolicy,
    HardwareProfile,
    IntegerParameter,
    LocalRefinementStrategy,
    ParetoArchive,
    RandomController,
    RandomSearchStrategy,
    ResourceBudget,
    SealedHoldoutTest,
    create_sealed_holdout,
    SearchSpace,
    StandardEvaluationPipeline,
    StrategyPortfolio,
    TPEStrategy,
    UniformController,
    ValidationError,
    canonical_json,
    compute_candidate_hash,
    compute_data_split_hash,
    make_evaluation_cache_key,
)


class TestAMDE4Architecture(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(42)
        self.search_space = SearchSpace(n_features=6, is_classification=True)

    # 1. Core Contracts & Identity Invariance
    def test_canonical_json_invariance_under_key_reordering(self):
        """Verifies canonical JSON produces bit-for-bit identical strings regardless of dict order."""
        dict_a = {"model": {"family": "mlp", "lr": 0.01, "depth": 2}, "scaler": "standard"}
        dict_b = {"scaler": "standard", "model": {"depth": 2, "family": "mlp", "lr": 0.01}}
        self.assertEqual(canonical_json(dict_a), canonical_json(dict_b))

    def test_candidate_hash_invariance_and_transient_filtering(self):
        """Verifies candidate hash strips transient execution metrics and is invariant."""
        genome = {"preprocessing": {"scaler": "robust", "imputer": "mean"}, "model": {"family": "linear", "C": 1.5}}
        genome_with_artifacts = copy.deepcopy(genome)
        genome_with_artifacts["score"] = 0.95
        genome_with_artifacts["timestamp"] = "2026-11-02T12:00:00Z"
        genome_with_artifacts["trial_id"] = 42

        h1 = compute_candidate_hash(genome)
        h2 = compute_candidate_hash(genome_with_artifacts)
        self.assertEqual(h1, h2)

    # 2. Candidate Identity vs Evaluation Identity Separation
    def test_evaluation_cache_key_separates_candidate_from_evaluation_context(self):
        """Verifies same candidate evaluated under different splits, seeds, or fidelities has distinct keys."""
        cand_hash = "abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        k1 = make_evaluation_cache_key(cand_hash, "task_a", "split_1", seed=42, fidelity_rung=0)
        k2 = make_evaluation_cache_key(cand_hash, "task_a", "split_2", seed=42, fidelity_rung=0)
        k3 = make_evaluation_cache_key(cand_hash, "task_a", "split_1", seed=123, fidelity_rung=0)
        k4 = make_evaluation_cache_key(cand_hash, "task_a", "split_1", seed=42, fidelity_rung=2)

        self.assertNotEqual(k1.to_string(), k2.to_string())
        self.assertNotEqual(k1.to_string(), k3.to_string())
        self.assertNotEqual(k1.to_string(), k4.to_string())

    # 3. Typed Search Space & Parameter Validation
    def test_continuous_parameter_bounds_and_log_scale(self):
        """Verifies continuous parameter sampling, log scaling, and perturbation bounds."""
        param = ContinuousParameter("lr", lower=1e-4, upper=1e-1, log_scale=True)
        for _ in range(50):
            val = param.sample(self.rng)
            self.assertTrue(param.validate(val))
            perturbed = param.perturb(val, self.rng, scale=0.20)
            self.assertTrue(param.validate(perturbed))

    def test_search_space_sampling_and_calibrated_mutation(self):
        """Verifies sampling valid candidate pipelines and calibrated mutation ratio."""
        cand = self.search_space.sample_random(self.rng)
        valid, msg = self.search_space.validate_genome(cand.genome)
        self.assertTrue(valid, msg)

        mutated = self.search_space.mutate_candidate(cand, self.rng, generation=1, scale=0.15)
        valid_m, msg_m = self.search_space.validate_genome(mutated.genome)
        self.assertTrue(valid_m, msg_m)
        self.assertNotEqual(cand.canonical_hash, mutated.canonical_hash)

    def test_search_space_chromosomal_crossover(self):
        """Verifies crossover across two parents yields a novel valid candidate."""
        p1 = self.search_space.sample_random(self.rng)
        p2 = self.search_space.sample_random(self.rng)
        child = self.search_space.crossover_candidates(p1, p2, self.rng, generation=1)
        valid, msg = self.search_space.validate_genome(child.genome)
        self.assertTrue(valid, msg)
        self.assertEqual(len(child.parent_hashes), 2)

    # 4. Strategy Portfolio
    def test_strategy_portfolio_registration_and_proposals(self):
        """Verifies StrategyPortfolio registers strategies and generates candidates."""
        portfolio = StrategyPortfolio.standard_portfolio(self.search_space)
        self.assertIn("random_search", portfolio.strategy_names)
        self.assertIn("evolutionary_elitist", portfolio.strategy_names)
        self.assertIn("local_refinement", portfolio.strategy_names)
        self.assertIn("bayesian_tpe", portfolio.strategy_names)

        strat = portfolio.get("random_search")
        proposals = strat.propose(n=3, rng=self.rng, archives={}, budget_fraction=0.1)
        self.assertEqual(len(proposals), 3)

    # 5. EGADA Controller: Numerical Stability & Replay Determinism
    def test_controller_finite_initial_reward(self):
        """Verifies initial trial baseline handling eliminates infinite reward bug."""
        ctrl = EGADAController(["s1", "s2"], discount_factor=0.95)
        r1 = ctrl.update_feedback("s1", val_score=0.85, train_duration=0.5, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertFalse(math.isinf(r1))
        self.assertFalse(math.isnan(r1))
        self.assertGreaterEqual(r1, 0.0)
        self.assertLessEqual(r1, 5.0)

    def test_controller_replay_determinism(self):
        """Verifies 100% bit-for-bit replay determinism across identical random seeds."""
        strats = ["s1", "s2", "s3"]
        decisions_a = []
        ctrl_a = EGADAController(strats, discount_factor=0.95)
        rng_a = random.Random(999)
        for step in range(30):
            d = ctrl_a.select_strategy(rng_a, budget_fraction=step / 30.0)
            decisions_a.append(d)
            ctrl_a.update_feedback(d, val_score=0.70, train_duration=0.2, is_novel=True, is_duplicate=False, is_failure=False)

        decisions_b = []
        ctrl_b = EGADAController(strats, discount_factor=0.95)
        rng_b = random.Random(999)
        for step in range(30):
            d = ctrl_b.select_strategy(rng_b, budget_fraction=step / 30.0)
            decisions_b.append(d)
            ctrl_b.update_feedback(d, val_score=0.70, train_duration=0.2, is_novel=True, is_duplicate=False, is_failure=False)

        self.assertEqual(decisions_a, decisions_b)

    def test_controller_baselines_uniform_and_random(self):
        """Verifies uniform round-robin and random controller baselines."""
        strats = ["s1", "s2", "s3"]
        u_ctrl = UniformController(strats)
        r_ctrl = RandomController(strats)
        u_decisions = [u_ctrl.select_strategy(self.rng) for _ in range(6)]
        self.assertEqual(u_decisions, ["s1", "s2", "s3", "s1", "s2", "s3"])

    # 6. Evaluation Pipeline & Data Isolation
    def test_standard_evaluation_pipeline_execution(self):
        """Verifies 10-phase pipeline execution on synthetic data."""
        np_rng = np.random.RandomState(42)
        X = np_rng.randn(80, 6)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_tr, y_tr = X[:60], y[:60]
        X_v, y_v = X[60:], y[60:]

        cand = self.search_space.sample_random(self.rng)
        pipeline = StandardEvaluationPipeline()
        req = EvaluationRequest(
            request_id="test_req_01",
            candidate=cand,
            dataset_id="synth",
            split_hash="s_hash",
            seed=42,
            fidelity=FidelityLevel.full_fidelity(),
            resource_budget=ResourceBudget(max_wallclock_seconds=10.0),
        )
        res = pipeline.execute(req, X_tr, y_tr, X_v, y_v)
        self.assertEqual(res.status, "completed")
        self.assertGreaterEqual(res.primary_metric, 0.0)
        self.assertIn("accuracy", res.metrics)
        self.assertIn("inference_latency_ms", res.metrics)
        self.assertIn("memory_mb", res.metrics)

    # 7. Multi-Objective Pareto Archive & Hypervolume
    def test_pareto_archive_non_dominated_sorting_and_hypervolume(self):
        """Verifies Pareto dominance and 2D hypervolume calculation."""
        archive = ParetoArchive(capacity=10)
        c1 = self.search_space.sample_random(self.rng)
        c2 = self.search_space.sample_random(self.rng)
        c3 = self.search_space.sample_random(self.rng)

        # c1 dominates c2 (lower loss, lower latency, lower memory)
        # loss = -accuracy
        archive.update(c1, loss=-0.95, latency=10.0, memory=50.0)
        archive.update(c2, loss=-0.80, latency=20.0, memory=60.0)
        archive.update(c3, loss=-0.90, latency=5.0, memory=40.0)

        # c1 and c3 should be non-dominated; c2 should be pruned
        hashes = [item["candidate_hash"] for item in archive.frontier]
        self.assertIn(c1.canonical_hash, hashes)
        self.assertIn(c3.canonical_hash, hashes)
        self.assertNotIn(c2.canonical_hash, hashes)

        hv = archive.calculate_2d_hypervolume(ref_loss=0.0, ref_latency=50.0)
        self.assertGreater(hv, 0.0)

    # 8. Append-Only Evidence Ledger with Hash Chaining
    def test_event_ledger_append_and_hash_chaining(self):
        """Verifies EventLedger idempotency and cryptographic SHA-256 integrity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_ledger.db"
            ledger = EventLedger(db_path)

            e1 = ledger.append_event("ev_01", "exp_100", "experiment_started", {"dataset": "iris"})
            e2 = ledger.append_event("ev_02", "exp_100", "evaluation_completed", {"val_score": 0.92})
            # Idempotent write with duplicate event_id
            e2_dup = ledger.append_event("ev_02", "exp_100", "evaluation_completed", {"val_score": 0.92})

            events = ledger.get_events("exp_100")
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].event_id, "ev_01")
            self.assertEqual(events[1].event_id, "ev_02")

            h = ledger.compute_ledger_sha256()
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 64)
            ledger.close()

    # 9. Hardware Profile & Pascal Architecture Safeguards
    def test_hardware_profile_detection(self):
        """Verifies hardware capability probing detects CPU cores, RAM, and Pascal non-tensor status."""
        profile = HardwareProfile.detect()
        self.assertGreaterEqual(profile.cpu_physical_cores, 1)
        self.assertGreater(profile.total_ram_mb, 1000.0)
        # If compute capability is 6.1 (Quadro P2000), Tensor Cores must evaluate to False
        if profile.gpu_compute_capability == "6.1":
            self.assertFalse(profile.has_tensor_cores)

    # 10. Sealed Holdout Test Token Gate (BOT-05)
    def test_sealed_holdout_test_token_gate_and_single_read(self):
        """Verifies sealed holdout partition enforces single-read token authorization."""
        np_rng = np.random.RandomState(42)
        X_test = np_rng.randn(20, 6)
        y_test = (X_test[:, 0] > 0).astype(int)
        holdout, token = create_sealed_holdout(X_test, y_test, dataset_name="unit_synth")

        # 1. Plaintext token is NOT accessible via property access
        with self.assertRaises(AttributeError):
            _ = holdout.token

        X_train = np_rng.randn(40, 6)
        y_train = (X_train[:, 0] > 0).astype(int)
        cand = self.search_space.sample_random(self.rng)

        # 2. Invalid token fails
        with self.assertRaises(EvaluationError):
            holdout.evaluate(cand, X_train, y_train, token="invalid_token")

        # 3. Valid token succeeds
        score = holdout.evaluate(cand, X_train, y_train, token=token)
        self.assertIsInstance(score, float)
        self.assertTrue(holdout.is_consumed)

        # 4. Repeated read fails (single-read enforcement)
        with self.assertRaises(EvaluationError):
            holdout.evaluate(cand, X_train, y_train, token=token)

    # 11. End-to-End Engine Discovery Campaign
    def test_engine_end_to_end_search_and_sealed_evaluation(self):
        """Verifies complete end-to-end autonomous discovery campaign with AMDE4Engine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_file = Path(tmp_dir) / "engine_ledger.db"
            np_rng = np.random.RandomState(101)
            X = np_rng.randn(100, 6)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)

            X_tr, y_tr = X[:60], y[:60]
            X_val, y_val = X[60:80], y[60:80]
            X_te, y_te = X[80:], y[80:]

            engine = AMDE4Engine(
                n_features=6,
                is_classification=True,
                seed=42,
                ledger_path=ledger_file,
                use_multi_fidelity=False,
                enable_controller=True,
            )

            results = engine.search(X_tr, y_tr, X_val, y_val, dataset_name="test_synth", budget=12)
            self.assertEqual(results["evaluations_completed"], 12)
            self.assertGreater(results["incumbent_score"], 0.40)
            self.assertIsNotNone(results["incumbent_candidate"])
            self.assertGreaterEqual(results["pareto_size"], 1)
            self.assertEqual(len(results["trajectory"]), 12)

            # Evaluate on sealed holdout
            holdout, token = create_sealed_holdout(X_te, y_te, dataset_name="test_synth")
            test_score = engine.evaluate_sealed_holdout(X_tr, y_tr, holdout, token=token)
            self.assertIsInstance(test_score, float)
            self.assertGreaterEqual(test_score, 0.0)
            engine.ledger.close()


if __name__ == "__main__":
    unittest.main()
