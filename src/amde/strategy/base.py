"""AMDE 4.0 Search Strategy Interface and Contract.

Defines the pluggable strategy contract with explicit metadata:
- Required state/archive inputs
- Computational overhead estimation
- Determinism guarantees
- Decoupled candidate proposal and feedback loops
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

from amde.core.contracts import Candidate
from amde.space.search_space import SearchSpace


class SearchStrategy(abc.ABC):
    """Abstract base class for all AMDE 4.0 search strategies."""

    def __init__(self, name: str, search_space: SearchSpace):
        self.name = name
        self.search_space = search_space
        self.proposals_count: int = 0
        self.feedback_history: list[dict[str, Any]] = []

    @property
    @abc.abstractmethod
    def required_information(self) -> Sequence[str]:
        """Names of archives or data structures required by this strategy (e.g. 'elite_archive')."""
        pass

    @property
    @abc.abstractmethod
    def computational_overhead_ms(self) -> float:
        """Estimated CPU proposal overhead in milliseconds per candidate."""
        pass

    @abc.abstractmethod
    def propose(
        self,
        n: int,
        rng: random.Random,
        archives: Mapping[str, Any],
        budget_fraction: float = 0.0,
        generation: int = 0,
    ) -> List[Candidate]:
        """Proposes up to n candidate pipelines."""
        pass

    def record_feedback(self, candidate_hash: str, score: float, improved: bool) -> None:
        """Incorporate feedback from trial evaluation."""
        self.feedback_history.append({
            "candidate_hash": candidate_hash,
            "score": score,
            "improved": improved,
        })
