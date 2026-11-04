"""AMDE 4.0 Hardening Regression Test Suite.

Verifies fixes for DEF-11 through DEF-16:
- DEF-11: canonical_json NaN/Inf validation, -0.0 normalization, Unicode normalization, numpy scalars.
- DEF-12: Full SHA-256 hash in EvaluationCacheKey, complete-array data split hashing.
- DEF-13: EGADA zero stagnation reward, online empirical variance updating.
- DEF-14: ParetoArchive 3D hypervolume calculation, out-of-bounds boundary point accounting.
- DEF-15: EventLedger SQLite connection cleanup without ResourceWarning, busy_timeout.
- DEF-16: Version string and packaging integrity.
"""

from __future__ import annotations

import math
import random
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np

import amde
from amde.archive.pareto import ParetoArchive
from amde.controller.egada import EGADAController
from amde.core.contracts import Candidate, EvaluationCacheKey
from amde.core.errors import ValidationError
from amde.core.identity import (
    canonical_json,
    compute_candidate_hash,
    compute_data_split_hash,
    make_evaluation_cache_key,
)
from amde.ledger.event_ledger import EventLedger


class TestAMDE4HardeningRegressions(unittest.TestCase):
    """Regression test suite for AMDE 4.0 forensic hardening."""

    def test_def11_nan_and_inf_raise_validation_error(self):
        """DEF-11: canonical_json must reject NaN and Inf rather than silently mapping to null."""
        with self.assertRaises(ValidationError):
            canonical_json({"alpha": float("nan")})

        with self.assertRaises(ValidationError):
            canonical_json({"beta": float("inf")})

        with self.assertRaises(ValidationError):
            canonical_json({"gamma": float("-inf")})

        with self.assertRaises(ValidationError):
            canonical_json({"nested": [1.0, 2.0, float("nan")]})

    def test_def11_negative_zero_normalization(self):
        """DEF-11: -0.0 and 0.0 must serialize to identical canonical JSON."""
        str_neg_zero = canonical_json({"val": -0.0})
        str_pos_zero = canonical_json({"val": 0.0})
        self.assertEqual(str_neg_zero, str_pos_zero)
        self.assertEqual(str_pos_zero, '{"val":0.0000000}')

    def test_def11_unicode_nfc_normalization(self):
        """DEF-11: Composed and decomposed Unicode strings must produce identical canonical representation."""
        # 'e' with acute accent: combined (NFC) vs decomposed (NFD)
        s_nfc = "\u00e9"
        s_nfd = "e\u0301"
        self.assertNotEqual(s_nfc, s_nfd)
        self.assertEqual(canonical_json({"key": s_nfc}), canonical_json({"key": s_nfd}))
        self.assertEqual(canonical_json({s_nfc: 1}), canonical_json({s_nfd: 1}))

    def test_def11_numpy_scalar_handling(self):
        """DEF-11: NumPy integer, floating, and bool scalars serialize as JSON literals, not quoted strings."""
        payload = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159265),
            "bool_val": np.bool_(True),
        }
        res = canonical_json(payload)
        self.assertIn('"int_val":42', res)
        self.assertIn('"bool_val":true', res)
        self.assertIn('"float_val":3.1415927', res)

    def test_def12_evaluation_cache_key_full_hash(self):
        """DEF-12: EvaluationCacheKey preserves full 64-char hashes for collision resistance."""
        cand_hash = "a" * 64
        split_hash = "b" * 64
        key = make_evaluation_cache_key(
            candidate_hash=cand_hash,
            dataset_id="test_ds",
            split_hash=split_hash,
            seed=42,
            fidelity_rung=2,
            metric_id="accuracy",
            evaluator_version="4.0.1",
        )
        s = key.to_string()
        self.assertIn(cand_hash, s)
        self.assertIn(split_hash, s)
        self.assertEqual(len(cand_hash), 64)
        self.assertEqual(len(split_hash), 64)

    def test_def12_compute_data_split_hash_full_array_sensitivity(self):
        """DEF-12: Differences beyond row 10 must produce different split hashes."""
        X_tr_1 = np.zeros((100, 4), dtype=np.float64)
        X_tr_2 = np.zeros((100, 4), dtype=np.float64)
        # Modify row 50 (beyond original 10-row slice)
        X_tr_2[50, 0] = 1.0

        y_tr = np.zeros(100, dtype=np.int64)
        X_v = np.zeros((20, 4), dtype=np.float64)
        y_v = np.zeros(20, dtype=np.int64)

        h1 = compute_data_split_hash(X_tr_1, y_tr, X_v, y_v)
        h2 = compute_data_split_hash(X_tr_2, y_tr, X_v, y_v)
        self.assertNotEqual(h1, h2, "compute_data_split_hash failed to detect change beyond first 10 rows")

    def test_def13_egada_zero_reward_on_stagnation(self):
        """DEF-13: Valid candidates with zero improvement over incumbent receive 0.0 reward."""
        controller = EGADAController(strategy_names=["strat_a", "strat_b"])
        # First valid candidate establishes incumbent = 0.85
        r1 = controller.update_feedback("strat_a", val_score=0.85, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertEqual(controller.incumbent_score, 0.85)

        # Subsequent candidate with lower score (0.80) does NOT improve
        r2 = controller.update_feedback("strat_a", val_score=0.80, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertEqual(r2, 0.0, "Stagnant/inferior candidate must receive 0.0 reward, not positive bonus")

        # Candidate with equal score (0.85) does NOT improve
        r3 = controller.update_feedback("strat_a", val_score=0.85, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertEqual(r3, 0.0, "Equal score candidate must receive 0.0 reward")

        # Candidate with higher score (0.90) receives positive improvement reward
        r4 = controller.update_feedback("strat_a", val_score=0.90, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)
        self.assertGreater(r4, 0.0)

    def test_def13_egada_online_variance_update(self):
        """DEF-13: Bandit arm posterior variance updates dynamically with evidence."""
        controller = EGADAController(strategy_names=["strat_a"], prior_variance=0.25)
        initial_var = controller.stats["strat_a"]["variance"]
        self.assertEqual(initial_var, 0.25)

        # Feed several diverse observations
        for s in [0.70, 0.95, 0.60, 0.99]:
            controller.update_feedback("strat_a", val_score=s, train_duration=0.5, is_novel=True, is_duplicate=False, is_failure=False)

        updated_var = controller.stats["strat_a"]["variance"]
        self.assertNotEqual(initial_var, updated_var, "Posterior variance should update from empirical evidence")

    def test_def14_pareto_3d_hypervolume(self):
        """DEF-14: ParetoArchive exact 3D hypervolume calculation across (loss, latency, memory)."""
        archive = ParetoArchive(capacity=10)
        candA = Candidate(genome={"id": "A"}, canonical_hash="hash_a")
        candB = Candidate(genome={"id": "B"}, canonical_hash="hash_b")
        # Point A: loss=-0.5, latency=10.0, memory=20.0
        # Point B: loss=-0.8, latency=30.0, memory=40.0
        # Reference point: ref_loss=0.0, ref_latency=50.0, ref_memory=100.0
        archive.update(candA, loss=-0.5, latency=10.0, memory=20.0)
        archive.update(candB, loss=-0.8, latency=30.0, memory=40.0)

        hv3d = archive.calculate_3d_hypervolume(ref_loss=0.0, ref_latency=50.0, ref_memory=100.0)
        # Analytical solution verified earlier: 1960.0
        self.assertAlmostEqual(hv3d, 1960.0, places=2)

    def test_def14_out_of_bounds_accounting(self):
        """DEF-14: Out of bounds points relative to reference points are accounted for."""
        archive = ParetoArchive(capacity=10)
        cand1 = Candidate(genome={"id": "valid"}, canonical_hash="h1")
        cand2 = Candidate(genome={"id": "oob_latency"}, canonical_hash="h2")
        cand3 = Candidate(genome={"id": "oob_memory"}, canonical_hash="h3")

        # Provide points that are non-dominated (better loss, but worse latency/memory)
        archive.update(cand1, loss=-0.5, latency=20.0, memory=30.0)
        archive.update(cand2, loss=-0.7, latency=80.0, memory=30.0)
        archive.update(cand3, loss=-0.9, latency=20.0, memory=150.0)

        oob_count = archive.get_out_of_bounds_count(ref_loss=0.0, ref_latency=50.0, ref_memory=100.0)
        self.assertEqual(oob_count, 2)

    def test_def15_sqlite_connection_no_resource_warnings(self):
        """DEF-15: EventLedger operations cleanly close SQLite connections without ResourceWarning."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_ledger.db"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ResourceWarning)
                ledger = EventLedger(db_path)
                for i in range(10):
                    ledger.append_event(
                        event_id=f"evt_{i}",
                        experiment_id="exp_01",
                        event_type="test_event",
                        payload={"step": i},
                    )
                events = ledger.get_events("exp_01")
                sha = ledger.compute_ledger_sha256()
                self.assertEqual(len(events), 10)
                self.assertTrue(bool(sha))

                # Check for ResourceWarning specifically related to sqlite3
                sqlite_warnings = [
                    item for item in w
                    if issubclass(item.category, ResourceWarning) and "sqlite" in str(item.message).lower()
                ]
                self.assertEqual(len(sqlite_warnings), 0, f"Encountered unclosed SQLite warnings: {sqlite_warnings}")

    def test_def16_version_alignment(self):
        """DEF-16: Release version must be 4.0.2 in amde.__version__."""
        self.assertEqual(amde.__version__, "4.0.2")


if __name__ == "__main__":
    unittest.main()
