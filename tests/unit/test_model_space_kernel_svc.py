"""Tests for kernel_svc model space and legacy compatibility alias."""

import unittest
import warnings
from sklearn.svm import SVC

from amde import Candidate, FidelityLevel, SearchSpace, StandardEvaluationPipeline, TPEStrategy
from amde.core.identity import compute_candidate_hash


class ModelSpaceKernelSVCTests(unittest.TestCase):
    """Unit tests for kernel_svc model space and legacy kernel_ridge alias."""

    def test_search_space_contains_kernel_svc(self):
        space = SearchSpace(n_features=64, is_classification=True)
        self.assertIn("kernel_svc", space.MODEL_FAMILIES)
        self.assertNotIn("kernel_ridge", space.MODEL_FAMILIES)
        self.assertEqual(space.DEPRECATED_ALIASES.get("kernel_ridge"), "kernel_svc")

    def test_search_space_validation(self):
        space = SearchSpace(n_features=64, is_classification=True)

        # Valid kernel_svc
        genome_svc = {
            "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
            "model": {"family": "kernel_svc", "alpha": 0.1, "kernel": "rbf", "gamma": 0.05},
        }
        is_valid, msg = space.validate_genome(genome_svc)
        self.assertTrue(is_valid, msg)

        # Deprecated alias kernel_ridge accepted
        genome_legacy = {
            "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
            "model": {"family": "kernel_ridge", "alpha": 0.1, "kernel": "rbf", "gamma": 0.05},
        }
        is_valid, msg = space.validate_genome(genome_legacy)
        self.assertTrue(is_valid, msg)

        # Unknown family rejected
        genome_unknown = {
            "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
            "model": {"family": "unknown_xgboost"},
        }
        is_valid, msg = space.validate_genome(genome_unknown)
        self.assertFalse(is_valid)

    def test_pipeline_builds_svc_from_kernel_svc(self):
        pipeline = StandardEvaluationPipeline()
        model_spec = {"family": "kernel_svc", "alpha": 0.05, "kernel": "rbf", "gamma": 0.01}
        fidelity = FidelityLevel.full_fidelity()
        model = pipeline._build_model(model_spec, fidelity=fidelity, seed=42)

        self.assertIsInstance(model, SVC)
        self.assertAlmostEqual(model.C, 1.0 / 0.05)
        self.assertEqual(model.kernel, "rbf")
        self.assertAlmostEqual(model.gamma, 0.01)
        self.assertEqual(model.random_state, 42)

    def test_pipeline_builds_svc_from_legacy_kernel_ridge_with_warning(self):
        pipeline = StandardEvaluationPipeline()
        model_spec = {"family": "kernel_ridge", "alpha": 0.2, "kernel": "poly", "gamma": 0.5}
        fidelity = FidelityLevel.full_fidelity()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = pipeline._build_model(model_spec, fidelity=fidelity, seed=42)
            self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in w))

        self.assertIsInstance(model, SVC)
        self.assertAlmostEqual(model.C, 1.0 / 0.2)
        self.assertEqual(model.kernel, "poly")
        self.assertAlmostEqual(model.gamma, 0.5)

    def test_tpe_strategy_proposes_and_scores_kernel_svc(self):
        import random
        space = SearchSpace(n_features=64, is_classification=True)
        tpe = TPEStrategy(space, min_observations=2)

        genome = {
            "preprocessing": {"imputer": "none", "scaler": "minmax", "feature_selection": "none", "k_features": 32},
            "model": {"family": "kernel_svc", "alpha": 0.5, "kernel": "rbf", "gamma": 0.1},
        }
        c_hash = compute_candidate_hash(genome)
        cand = Candidate(genome=genome, canonical_hash=c_hash)

        tpe.record_feedback(c_hash, 0.95, improved=True, candidate=cand)
        props = tpe.propose(n=1, rng=random.Random(42), archives={}, generation=1)
        self.assertEqual(len(props), 1)
        self.assertIsNotNone(props[0].genome)


if __name__ == "__main__":
    unittest.main()
