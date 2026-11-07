"""Unit Tests for AMDE 4.0 SOTA Tabular Performance Upgrades.

Verifies:
1. LightGBM integration into SearchSpace, EvaluationPipeline, and TPE.
2. Novelty enforcement loop dropping duplicate rate below 3%.
3. EventLedger async batching, high throughput, and SHA-256 chain verification.
"""

from __future__ import annotations

import random
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

from amde.archive.dedup import CandidateDeduplicator
from amde.core.contracts import EvaluationRequest, FidelityLevel, ResourceBudget
from amde.engine import AMDE4Engine
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.ledger.event_ledger import EventLedger
from amde.space.search_space import SearchSpace
from amde.strategy.tpe import TPEStrategy


class AMDE4SOTAUpgradesTests(unittest.TestCase):
    """Test suite for AMDE 4.0 production SOTA performance upgrades."""

    def test_lightgbm_search_space_and_pipeline(self) -> None:
        """Task 1: Verifies LightGBM model family integration and evaluation execution."""
        ss = SearchSpace(n_features=6, is_classification=True)
        self.assertIn("lgbm", ss.MODEL_FAMILIES)
        self.assertIn("n_estimators", ss.lgbm_params)
        self.assertIn("max_depth", ss.lgbm_params)
        self.assertIn("learning_rate", ss.lgbm_params)
        self.assertIn("num_leaves", ss.lgbm_params)
        self.assertIn("subsample", ss.lgbm_params)

        rng = random.Random(42)
        # Verify sampling
        samples = [ss.sample_random(rng) for _ in range(50)]
        lgbm_samples = [c for c in samples if c.genome["model"]["family"] == "lgbm"]
        self.assertGreater(len(lgbm_samples), 0, "No LightGBM candidate sampled from search space")

        cand = lgbm_samples[0]
        # Verify mutation
        mut = ss.mutate_candidate(cand, rng, scale=0.25)
        self.assertEqual(mut.genome["model"]["family"], "lgbm")

        # Verify pipeline execution
        pipe = StandardEvaluationPipeline()
        X_tr = np.random.randn(40, 6).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=40)
        X_val = np.random.randn(20, 6).astype(np.float32)
        y_val = np.random.randint(0, 2, size=20)

        req = EvaluationRequest(
            request_id="test_lgbm_eval",
            candidate=cand,
            dataset_id="test_synth",
            split_hash="fake_split_hash",
            seed=42,
            fidelity=FidelityLevel.full_fidelity(),
            resource_budget=ResourceBudget(max_evaluations=5),
        )
        res = pipe.execute(req, X_tr, y_tr, X_val, y_val)
        self.assertEqual(res.status, "completed")
        self.assertGreaterEqual(res.primary_metric, 0.0)
        self.assertGreater(res.metrics.get("accuracy", 0.0), 0.0)
        self.assertGreater(res.inference_latency_ms, 0.0)

    def test_novelty_enforcement_duplicate_rate(self) -> None:
        """Task 2: Verifies CandidateDeduplicator.has_seen and < 3% duplicate rate in search."""
        dedup = CandidateDeduplicator()
        self.assertFalse(dedup.has_seen("hash_abc"))
        self.assertFalse(dedup.is_duplicate("hash_abc"))
        self.assertEqual(dedup._total_proposals, 1)
        self.assertEqual(dedup._duplicate_attempts, 0)

        # Register hash_abc
        dedup.register("hash_abc")
        self.assertTrue(dedup.has_seen("hash_abc"))
        # has_seen should NOT increment proposal counters
        self.assertEqual(dedup._total_proposals, 1)

        # Run AMDE engine search over 30 evaluations and assert duplicate rate < 3%
        X_tr = np.random.randn(60, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=60)
        X_val = np.random.randn(20, 4).astype(np.float32)
        y_val = np.random.randint(0, 2, size=20)

        engine = AMDE4Engine(seed=42)
        result = engine.search(X_tr, y_tr, X_val, y_val, budget=30, dataset_name="dedup_test")
        self.assertLess(result["duplicate_rate"], 0.03)
        self.assertEqual(result["total_unique"], 30)

    def test_event_ledger_async_batching_and_chain(self) -> None:
        """Task 3: Verifies high-throughput async batch writes and cryptographic chain verification."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "async_ledger.db"
            ledger = EventLedger(db_path)

            t0 = time.perf_counter()
            for i in range(200):
                ledger.append_event(
                    event_id=f"evt_{i}",
                    experiment_id="exp_async",
                    event_type="test_step",
                    payload={"step": i, "val": i * 0.1},
                )
            append_duration = time.perf_counter() - t0

            # 200 writes must complete in under 500ms
            self.assertLess(append_duration, 0.5)

            # Flush and verify
            ledger.flush()
            events = ledger.get_events("exp_async")
            self.assertEqual(len(events), 200)

            valid, msg = ledger.verify_chain()
            self.assertTrue(valid, f"Chain verification failed: {msg}")
            self.assertEqual(msg, "VALID_200_EVENTS")

            sha = ledger.compute_ledger_sha256()
            self.assertIsInstance(sha, str)
            self.assertEqual(len(sha), 64)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
