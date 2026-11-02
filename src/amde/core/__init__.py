"""AMDE 4.0 Core Contracts, Identity, and Error Taxonomy."""

from amde.core.contracts import (
    Candidate,
    ControllerDecision,
    EvaluationCacheKey,
    EvaluationRequest,
    EvaluationResult,
    ExperimentEvent,
    ExperimentSpec,
    FidelityLevel,
    ResourceBudget,
    RunManifest,
    utc_now_iso,
)
from amde.core.errors import (
    AMDEError,
    EvaluationError,
    EvaluationTimeoutError,
    IncompatibleCacheError,
    LedgerError,
    ResourceExhaustedError,
    SearchSpaceError,
    ValidationError,
)
from amde.core.identity import (
    canonical_json,
    compute_candidate_hash,
    compute_data_split_hash,
    make_evaluation_cache_key,
)

__all__ = [
    "Candidate",
    "ControllerDecision",
    "EvaluationCacheKey",
    "EvaluationRequest",
    "EvaluationResult",
    "ExperimentEvent",
    "ExperimentSpec",
    "FidelityLevel",
    "ResourceBudget",
    "RunManifest",
    "utc_now_iso",
    "AMDEError",
    "ValidationError",
    "SearchSpaceError",
    "EvaluationError",
    "EvaluationTimeoutError",
    "ResourceExhaustedError",
    "LedgerError",
    "IncompatibleCacheError",
    "canonical_json",
    "compute_candidate_hash",
    "compute_data_split_hash",
    "make_evaluation_cache_key",
]
