"""AMDE 4.0 Core Domain Contracts & Type-Safe Abstractions.

Defines versioned, immutable dataclasses and protocols governing:
- Experiment specifications
- Candidates and pipeline representations
- Evaluation requests and results
- Resource budgets and fidelity levels
- Controller decisions and event ledgers
- Archives and failure records
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union


def utc_now_iso() -> str:
    """Returns current UTC timestamp in strict ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResourceBudget:
    """Explicit resource constraints allocated to an experiment or trial."""
    max_evaluations: int = 50
    max_wallclock_seconds: float = 300.0
    max_cpu_seconds: float = 600.0
    max_memory_mb: float = 2048.0
    max_gpu_memory_mb: Optional[float] = 3072.0  # Bounded for Quadro P2000 (4GB)

    def is_evaluations_exhausted(self, completed_evaluations: int) -> bool:
        return completed_evaluations >= self.max_evaluations

    def is_time_exhausted(self, elapsed_seconds: float) -> bool:
        return elapsed_seconds >= self.max_wallclock_seconds


@dataclass(frozen=True)
class FidelityLevel:
    """Defines a discrete computational evaluation fidelity rung."""
    rung_id: int
    name: str
    subsample_ratio: float  # Fraction of training samples [0.0, 1.0]
    max_iterations: int     # Optimization iterations / epochs
    cost_multiplier: float  # Relative compute cost proxy

    @classmethod
    def full_fidelity(cls) -> FidelityLevel:
        return cls(rung_id=2, name="full_fidelity", subsample_ratio=1.0, max_iterations=50, cost_multiplier=1.0)

    @classmethod
    def screening_fidelity(cls) -> FidelityLevel:
        return cls(rung_id=0, name="screening", subsample_ratio=0.20, max_iterations=10, cost_multiplier=0.04)

    @classmethod
    def intermediate_fidelity(cls) -> FidelityLevel:
        return cls(rung_id=1, name="intermediate", subsample_ratio=0.50, max_iterations=25, cost_multiplier=0.25)


@dataclass(frozen=True)
class EvaluationCacheKey:
    """Cryptographically collision-resistant cache key strictly separating Candidate Identity

    from Evaluation Identity. Ensures cached evaluations are never reused under incompatible
    datasets, splits, random seeds, fidelity rungs, or evaluator versions.
    """
    candidate_hash: str
    dataset_id: str
    split_hash: str
    seed: int
    fidelity_rung: int
    metric_id: str
    evaluator_version: str = "4.0.1"

    def to_string(self) -> str:
        return (
            f"eval_{self.candidate_hash}_{self.dataset_id}_"
            f"{self.split_hash}_s{self.seed}_r{self.fidelity_rung}_"
            f"{self.metric_id}_v{self.evaluator_version}"
        )


@dataclass(frozen=True)
class Candidate:
    """An immutable candidate machine-learning discovery pipeline."""
    genome: Mapping[str, Any]
    canonical_hash: str
    generation: int = 0
    strategy_used: str = "random_search"
    parent_hashes: Tuple[str, ...] = field(default_factory=tuple)
    creation_timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_hash": self.canonical_hash,
            "genome": copy.deepcopy(dict(self.genome)),
            "generation": self.generation,
            "strategy_used": self.strategy_used,
            "parent_hashes": list(self.parent_hashes),
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass(frozen=True)
class EvaluationRequest:
    """An immutable, versioned request to evaluate a Candidate pipeline."""
    request_id: str
    candidate: Candidate
    dataset_id: str
    split_hash: str
    seed: int
    fidelity: FidelityLevel
    resource_budget: ResourceBudget
    metric_names: Tuple[str, ...] = ("accuracy", "macro_f1", "inference_latency_ms", "memory_mb")
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class EvaluationResult:
    """The verifiable outcome of a Candidate evaluation."""
    request_id: str
    candidate_hash: str
    status: Literal["completed", "failed", "pruned", "timeout"]
    primary_metric: float
    metrics: Mapping[str, float]
    fidelity: FidelityLevel
    train_duration_seconds: float
    inference_latency_ms: float
    peak_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    evaluator_version: str = "4.0.1"
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_hash": self.candidate_hash,
            "status": self.status,
            "primary_metric": self.primary_metric,
            "metrics": copy.deepcopy(dict(self.metrics)),
            "fidelity": dataclasses.asdict(self.fidelity),
            "train_duration_seconds": self.train_duration_seconds,
            "inference_latency_ms": self.inference_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "evaluator_version": self.evaluator_version,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ControllerDecision:
    """Records an adaptive strategy allocation decision by the meta-controller."""
    step: int
    selected_strategy: str
    bandit_probabilities: Mapping[str, float]
    budget_fraction: float
    exploration_mode: bool
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable specification defining a reproducible AMDE 4.0 experiment."""
    experiment_id: str
    dataset_name: str
    search_budget: ResourceBudget
    random_seed: int
    target_metric: str = "accuracy"
    multi_objective_metrics: Tuple[str, ...] = ("accuracy", "inference_latency_ms", "memory_mb")
    use_multi_fidelity: bool = False
    multi_fidelity_policy: str = "disabled"
    enable_controller: bool = True
    enable_deduplication: bool = True
    enable_pareto_archive: bool = True
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ExperimentEvent:
    """An idempotent event recorded in the append-only evidence ledger."""
    event_id: str
    experiment_id: str
    event_type: Literal[
        "experiment_started",
        "candidate_proposed",
        "evaluation_started",
        "evaluation_completed",
        "evaluation_pruned",
        "evaluation_failed",
        "incumbent_updated",
        "pareto_frontier_updated",
        "controller_feedback_updated",
        "experiment_completed",
    ]
    payload: Mapping[str, Any]
    sequence_number: int
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class RunManifest:
    """Provenance and execution manifest for a completed experiment."""
    experiment_id: str
    spec: ExperimentSpec
    git_commit: str
    git_branch: str
    hardware_profile: Mapping[str, Any]
    start_time: str
    end_time: str
    total_evaluations: int
    unique_candidates: int
    duplicate_attempts: int
    best_candidate_hash: str
    best_validation_score: float
    sealed_test_score: Optional[float]
    pareto_frontier_size: int
    manifest_sha256: Optional[str] = None
