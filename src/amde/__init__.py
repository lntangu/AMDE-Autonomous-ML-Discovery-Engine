"""AMDE 4.0: Autonomous Machine Learning Discovery Engine.

Principal ML Systems Architecture for Evidence-Guided Discovery.
"""

from amde.archive.cache import EvaluationCache
from amde.archive.dedup import CandidateDeduplicator
from amde.archive.elite import EliteArchive
from amde.archive.pareto import ParetoArchive
from amde.controller.baselines import RandomController, UniformController
from amde.controller.egada import EGADAController
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
from amde.engine import AMDE4Engine
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.evaluation.sealed import SealedHoldoutTest
from amde.fidelity.policy import DelayedPruningPolicy, FidelityPolicy, FullFidelityPolicy
from amde.hardware.profile import HardwareProfile
from amde.ledger.event_ledger import EventLedger
from amde.space.parameter import (
    BooleanParameter,
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    Parameter,
)
from amde.space.search_space import SearchSpace
from amde.strategy.evolutionary import EvolutionaryStrategy
from amde.strategy.local_refinement import LocalRefinementStrategy
from amde.strategy.portfolio import StrategyPortfolio
from amde.strategy.random_search import RandomSearchStrategy
from amde.strategy.tpe import TPEStrategy

__version__ = "4.0.0"

__all__ = [
    "AMDE4Engine",
    "Candidate",
    "ExperimentSpec",
    "ResourceBudget",
    "FidelityLevel",
    "EvaluationRequest",
    "EvaluationResult",
    "RunManifest",
    "SearchSpace",
    "HardwareProfile",
    "SealedHoldoutTest",
    "EventLedger",
    "CandidateDeduplicator",
    "EliteArchive",
    "ParetoArchive",
    "EvaluationCache",
    "StrategyPortfolio",
    "RandomSearchStrategy",
    "EvolutionaryStrategy",
    "LocalRefinementStrategy",
    "TPEStrategy",
    "EGADAController",
    "UniformController",
    "RandomController",
    "StandardEvaluationPipeline",
    "FidelityPolicy",
    "FullFidelityPolicy",
    "DelayedPruningPolicy",
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
    "__version__",
]
