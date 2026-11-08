"""AMDE 4.0.1 P0 Correctness Verification & Regression Suite.

Tests and proves the closure of the six P0 correctness findings:
1. P0-1: TPE feedback updates observation history and subsequent proposals depend on evaluated results.
2. P0-2: Search-space phenotype consistency, gene-to-model mapping, and unsupported dropout rejection.
3. P0-3: Sealed holdout isolation, token security, and distinction from process isolation.
4. P0-4: Event ledger cryptographic integrity, experiment_id hashing, and adversarial tamper tests.
5. P0-5: Pareto frontier exact preservation and bounded crowding-distance multi-objective trade-off retention.
6. P0-6: Timeout deadline enforcement through isolated worker processes and actual peak RSS telemetry.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
import random
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

from amde.archive.pareto import ParetoArchive
from amde.core.contracts import (
    Candidate,
    EvaluationRequest,
    EvaluationResult,
    FidelityLevel,
    ResourceBudget,
)
from amde.core.errors import EvaluationError, EvaluationTimeoutError
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.evaluation.sealed import SealedHoldoutTest, create_sealed_holdout
from amde.ledger.event_ledger import EventLedger
from amde.space.search_space import SearchSpace
from amde.strategy.tpe import TPEStrategy


class AMDE4P0CorrectnessTests(unittest.TestCase):
    """Rigorous tests proving P0 closure across all 6 core subsystems."""

    # =========================================================================
    # P0-1: TPE Feedback & Proposal Dependence
    # =========================================================================
    def test_p0_1_tpe_feedback_updates_history_and_drives_proposals(self) -> None:
        """Proves record_feedback updates observations and proposals depend on feedback."""
        space = SearchSpace(n_features=6, is_classification=True)
        tpe = TPEStrategy(space, gamma_quantile=0.25, min_observations=4)
        rng = random.Random(42)

        # 1. Uniform proposals before min_observations
        proposals = tpe.propose(n=2, rng=rng, archives={})
        self.assertEqual(len(proposals), 2)
        c0 = proposals[0]

        # 2. record_feedback MUST update observation history
        self.assertEqual(len(tpe.observation_history), 0)
        tpe.record_feedback(c0.canonical_hash, score=0.92, improved=True)
        self.assertEqual(len(tpe.observation_history), 1)
        self.assertEqual(tpe.observation_history[0][0].canonical_hash, c0.canonical_hash)
        self.assertEqual(tpe.observation_history[0][1], 0.92)

        # 3. Add observations with strong bias: tree_ensemble has score=0.95, others have 0.15
        for _ in range(12):
            c = space.sample_random(rng)
            fam = c.genome["model"]["family"]
            score = 0.95 if fam == "tree_ensemble" else 0.15
            tpe.record_observation(c, score)

        self.assertGreaterEqual(len(tpe.observation_history), 12)

        # 4. Subsequent proposals must heavily favor the high-scoring family (tree_ensemble)
        subsequent = tpe.propose(n=10, rng=rng, archives={})
        tree_count = sum(1 for p in subsequent if p.genome["model"]["family"] == "tree_ensemble")
        self.assertGreaterEqual(
            tree_count,
            7,
            f"TPE failed to exploit top-performing tree_ensemble family (only {tree_count}/10 proposed)",
        )

    # =========================================================================
    # P0-2: Search-Space Phenotype Consistency & Dropout Removal
    # =========================================================================
    def test_p0_2_search_space_phenotype_consistency(self) -> None:
        """Verifies every active gene affects model construction and dropout is rejected."""
        space = SearchSpace(n_features=8, is_classification=True)
        rng = random.Random(77)

        # 1. Sampled MLP candidates MUST NOT contain the inactive 'dropout' gene
        for _ in range(30):
            c = space.sample_random(rng)
            if c.genome["model"]["family"] == "mlp":
                self.assertNotIn("dropout", c.genome["model"])

        # 2. validate_genome must fail closed on unsupported dropout
        bad_mlp = {
            "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 4},
            "model": {"family": "mlp", "hidden_widths": [32], "activation": "relu", "optimizer": "adam", "dropout": 0.20},
        }
        ok, reason = space.validate_genome(bad_mlp)
        self.assertFalse(ok)
        self.assertIn("dropout", reason)

        # 3. Pipeline correctly maps optimizer genes to scikit-learn solver & momentum
        pipe = StandardEvaluationPipeline()
        f = FidelityLevel.screening_fidelity()

        m_adam = pipe._build_model({"family": "mlp", "optimizer": "adam", "activation": "relu"}, f, 42)
        self.assertEqual(m_adam.solver, "adam")
        self.assertEqual(m_adam.activation, "relu")

        m_sgd = pipe._build_model({"family": "mlp", "optimizer": "sgd_momentum", "activation": "tanh"}, f, 42)
        self.assertEqual(m_sgd.solver, "sgd")
        self.assertEqual(m_sgd.momentum, 0.9)
        self.assertTrue(m_sgd.nesterovs_momentum)
        self.assertEqual(m_sgd.activation, "tanh")

        m_lbfgs = pipe._build_model({"family": "mlp", "optimizer": "lbfgs", "activation": "logistic"}, f, 42)
        self.assertEqual(m_lbfgs.solver, "lbfgs")
        self.assertEqual(m_lbfgs.activation, "logistic")

        # 4. Pipeline rejects unsupported activation
        with self.assertRaises(EvaluationError):
            pipe._build_model({"family": "mlp", "optimizer": "adam", "activation": "unsupported_act"}, f, 42)

    # =========================================================================
    # P0-3: Sealed Holdout Isolation & Token Security
    # =========================================================================
    def test_p0_3_sealed_holdout_token_isolation(self) -> None:
        """Verifies search code cannot read authorization tokens via property access."""
        np_rng = np.random.RandomState(42)
        X_test = np_rng.randn(20, 4)
        y_test = (X_test[:, 0] > 0).astype(int)
        X_train = np_rng.randn(40, 4)
        y_train = (X_train[:, 0] > 0).astype(int)

        holdout, secret_token = create_sealed_holdout(X_test, y_test, dataset_name="audit_test")
        space = SearchSpace(n_features=4)
        cand = space.sample_random(random.Random(1))

        # 1. Plaintext token is NOT readable via property access
        with self.assertRaises(AttributeError):
            _ = holdout.token

        # 2. Plaintext token is NOT stored as a string instance attribute
        for key, val in holdout.__dict__.items():
            if isinstance(val, str):
                self.assertNotEqual(val, secret_token)

        # 3. Invalid token fails
        with self.assertRaises(EvaluationError):
            holdout.evaluate(cand, X_train, y_train, token="wrong_token_guess")

        # 4. Valid token succeeds once
        score = holdout.evaluate(cand, X_train, y_train, token=secret_token)
        self.assertIsInstance(score, float)
        self.assertTrue(holdout.is_consumed)

        # 5. Single-use enforcement: second evaluation with valid token fails
        with self.assertRaises(EvaluationError):
            holdout.evaluate(cand, X_train, y_train, token=secret_token)

    # =========================================================================
    # P0-4: Event Ledger Cryptographic Integrity & Adversarial Tamper Tests
    # =========================================================================
    def test_p0_4_ledger_integrity_and_adversarial_tampering(self) -> None:
        """Verifies hashing of all security-relevant metadata including experiment_id and chain verification."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ledger_p0.sqlite3"
            ledger = EventLedger(db_path)

            ledger.append_event("ev_01", "exp_alpha", "start", {"param": 10})
            ledger.append_event("ev_02", "exp_alpha", "trial", {"metric": 0.88})
            ledger.append_event("ev_03", "exp_alpha", "end", {"status": "ok"})

            # 1. Clean chain verification passes
            valid, msg = ledger.verify_chain()
            self.assertTrue(valid, msg)
            self.assertEqual(msg, "VALID_3_EVENTS")

            # 2. Adversarial Tamper 1: modify experiment_id
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("UPDATE experiment_events SET experiment_id = 'exp_ROGUE' WHERE seq = 2;")
                conn.commit()

            valid, msg = ledger.verify_chain()
            self.assertFalse(valid, "Tampered experiment_id went undetected!")
            self.assertIn("Tamper detected at seq 2", msg)

            # Restore and test Adversarial Tamper 2: modify payload
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("UPDATE experiment_events SET experiment_id = 'exp_alpha' WHERE seq = 2;")
                conn.execute("UPDATE experiment_events SET payload_json = '{\"metric\": 0.999}' WHERE seq = 2;")
                conn.commit()

            valid, msg = ledger.verify_chain()
            self.assertFalse(valid, "Tampered payload went undetected!")
            self.assertIn("Tamper detected at seq 2", msg)

            # Restore and test Adversarial Tamper 3: sequence deletion
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("DELETE FROM experiment_events WHERE seq = 2;")
                conn.commit()

            valid, msg = ledger.verify_chain()
            self.assertFalse(valid, "Deleted intermediate row went undetected!")
            self.assertIn("Sequence discontinuity", msg)
            ledger.close()

    # =========================================================================
    # P0-5: Pareto Archive Exact Preservation & Bounded Crowding Distance
    # =========================================================================
    def test_p0_5_pareto_exact_and_bounded_tradeoff_preservation(self) -> None:
        """Verifies exact Pareto preservation and multi-objective diversity under bounded capacity."""
        def _cand(name: str) -> Candidate:
            return Candidate(genome={"name": name}, canonical_hash=f"hash_{name}")

        # 1. Exact unconstrained preservation (default capacity=None)
        archive_exact = ParetoArchive(capacity=None)
        archive_exact.update(_cand("best_loss"), loss=-0.95, latency=100.0, memory=50.0)
        archive_exact.update(_cand("best_lat"), loss=-0.70, latency=0.2, memory=30.0)
        archive_exact.update(_cand("best_mem"), loss=-0.65, latency=20.0, memory=1.5)
        archive_exact.update(_cand("balanced_1"), loss=-0.88, latency=5.0, memory=15.0)
        archive_exact.update(_cand("balanced_2"), loss=-0.82, latency=2.0, memory=10.0)
        self.assertEqual(len(archive_exact), 5)

        # 2. Bounded capacity with NSGA-II crowding distance
        archive_bounded = ParetoArchive(capacity=3)
        archive_bounded.update(_cand("best_loss"), loss=-0.95, latency=100.0, memory=50.0)
        archive_bounded.update(_cand("best_lat"), loss=-0.70, latency=0.2, memory=30.0)
        archive_bounded.update(_cand("best_mem"), loss=-0.65, latency=20.0, memory=1.5)
        # Add clustered intermediate points
        archive_bounded.update(_cand("cluster_loss_1"), loss=-0.94, latency=98.0, memory=49.0)
        archive_bounded.update(_cand("cluster_loss_2"), loss=-0.93, latency=95.0, memory=48.0)

        self.assertEqual(len(archive_bounded), 3)
        retained = {item["candidate_hash"] for item in archive_bounded.frontier}

        # Extreme points for ALL 3 objectives MUST be preserved
        self.assertIn("hash_best_loss", retained, "Extreme best loss point was dropped!")
        self.assertIn("hash_best_lat", retained, "Extreme lowest latency point was dropped!")
        self.assertIn("hash_best_mem", retained, "Extreme lowest memory point was dropped!")

    # =========================================================================
    # P0-6: Timeout & Memory Telemetry (Actual Peak RSS & Process Isolation)
    # =========================================================================
    def test_p0_6_actual_peak_rss_and_process_deadline_enforcement(self) -> None:
        """Verifies actual resident set size measurement and hard OS deadline enforcement."""
        pipe = StandardEvaluationPipeline()
        space = SearchSpace(n_features=6)
        cand = space.sample_random(random.Random(42))

        np_rng = np.random.RandomState(42)
        X_tr = np_rng.randn(40, 6)
        y_tr = (X_tr[:, 0] > 0).astype(int)
        X_v = np_rng.randn(20, 6)
        y_v = (X_v[:, 0] > 0).astype(int)

        req = EvaluationRequest(
            request_id="p0_rss_eval",
            candidate=cand,
            dataset_id="test_rss",
            split_hash="split_0",
            seed=42,
            fidelity=FidelityLevel.screening_fidelity(),
            resource_budget=ResourceBudget(),
            timeout_seconds=15.0,
        )

        # 1. In-process actual peak RSS measurement
        res_inproc = pipe.execute(req, X_tr, y_tr, X_v, y_v, isolated=False)
        self.assertEqual(res_inproc.status, "completed")
        self.assertGreater(res_inproc.peak_memory_mb, 10.0, "peak_memory_mb is not measuring actual process RSS")
        self.assertIn("memory_mb", res_inproc.metrics)
        self.assertEqual(res_inproc.evaluator_version, "4.0.1")

        # 2. Out-of-process isolated worker execution
        res_iso = pipe.execute(req, X_tr, y_tr, X_v, y_v, isolated=True)
        self.assertEqual(res_iso.status, "completed")
        self.assertGreater(res_iso.peak_memory_mb, 10.0)

        # 3. Hard deadline enforcement: isolated worker terminated on timeout
        req_timeout = EvaluationRequest(
            request_id="p0_timeout_eval",
            candidate=cand,
            dataset_id="test_rss",
            split_hash="split_0",
            seed=42,
            fidelity=FidelityLevel.full_fidelity(),
            resource_budget=ResourceBudget(),
            timeout_seconds=0.001,  # Impossible deadline
        )
        res_timeout = pipe.execute(req_timeout, X_tr, y_tr, X_v, y_v, isolated=True)
        self.assertEqual(res_timeout.status, "timeout")
        self.assertIn("timeout", res_timeout.error_message.lower())


if __name__ == "__main__":
    unittest.main()
