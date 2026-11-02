"""AMDE 4.0 Archive, Memory, and Multi-Objective Frontier Subsystems."""

from amde.archive.cache import EvaluationCache
from amde.archive.dedup import CandidateDeduplicator
from amde.archive.elite import EliteArchive
from amde.archive.pareto import ParetoArchive

__all__ = [
    "CandidateDeduplicator",
    "EliteArchive",
    "ParetoArchive",
    "EvaluationCache",
]
