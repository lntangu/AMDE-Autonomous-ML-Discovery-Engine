"""Tests for kernel_svc model space and legacy compatibility alias."""

import warnings
import pytest
from sklearn.svm import SVC

from amde import Candidate, FidelityLevel, SearchSpace, StandardEvaluationPipeline, TPEStrategy
from amde.core.identity import compute_candidate_hash


def test_search_space_contains_kernel_svc():
    space = SearchSpace(n_features=64, is_classification=True)
    assert "kernel_svc" in space.MODEL_FAMILIES
    assert "kernel_ridge" not in space.MODEL_FAMILIES
    assert space.DEPRECATED_ALIASES.get("kernel_ridge") == "kernel_svc"


def test_search_space_validation():
    space = SearchSpace(n_features=64, is_classification=True)

    # Valid kernel_svc
    genome_svc = {
        "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
        "model": {"family": "kernel_svc", "alpha": 0.1, "kernel": "rbf", "gamma": 0.05},
    }
    is_valid, msg = space.validate_genome(genome_svc)
    assert is_valid, msg

    # Deprecated alias kernel_ridge accepted
    genome_legacy = {
        "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
        "model": {"family": "kernel_ridge", "alpha": 0.1, "kernel": "rbf", "gamma": 0.05},
    }
    is_valid, msg = space.validate_genome(genome_legacy)
    assert is_valid, msg

    # Unknown family rejected
    genome_unknown = {
        "preprocessing": {"imputer": "mean", "scaler": "standard", "feature_selection": "none", "k_features": 32},
        "model": {"family": "unknown_xgboost"},
    }
    is_valid, msg = space.validate_genome(genome_unknown)
    assert not is_valid


def test_pipeline_builds_svc_from_kernel_svc():
    pipeline = StandardEvaluationPipeline()
    model_spec = {"family": "kernel_svc", "alpha": 0.05, "kernel": "rbf", "gamma": 0.01}
    fidelity = FidelityLevel.full_fidelity()
    model = pipeline._build_model(model_spec, fidelity=fidelity, seed=42)

    assert isinstance(model, SVC)
    assert model.C == pytest.approx(1.0 / 0.05)
    assert model.kernel == "rbf"
    assert model.gamma == pytest.approx(0.01)
    assert model.random_state == 42


def test_pipeline_builds_svc_from_legacy_kernel_ridge_with_warning():
    pipeline = StandardEvaluationPipeline()
    model_spec = {"family": "kernel_ridge", "alpha": 0.2, "kernel": "poly", "gamma": 0.5}
    fidelity = FidelityLevel.full_fidelity()

    with pytest.deprecated_call():
        model = pipeline._build_model(model_spec, fidelity=fidelity, seed=42)

    assert isinstance(model, SVC)
    assert model.C == pytest.approx(1.0 / 0.2)
    assert model.kernel == "poly"
    assert model.gamma == pytest.approx(0.5)


def test_tpe_strategy_proposes_and_scores_kernel_svc():
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
    assert len(props) == 1
    assert props[0].genome is not None
