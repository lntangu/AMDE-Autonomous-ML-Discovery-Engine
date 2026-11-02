"""AMDE 4.0 Uniform Random Search Strategy."""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Sequence

from amde.core.contracts import Candidate
from amde.space.search_space import SearchSpace
from amde.strategy.base import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Generates uniform, unbiased candidate pipelines across the search space."""

    def __init__(self, search_space: SearchSpace):
        super().__init__(name="random_search", search_space=search_space)

    @property
    def required_information(self) -> Sequence[str]:
        return ()  # Stateless, requires no archives

    @property
    def computational_overhead_ms(self) -> float:
        return 0.1  # Fast sampling

    def propose(
        self,
        n: int,
        rng: random.Random,
        archives: Mapping[str, Any],
        budget_fraction: float = 0.0,
        generation: int = 0,
    ) -> List[Candidate]:
        candidates = []
        for _ in range(n):
            cand = self.search_space.sample_random(rng)
            cand = Candidate(
                genome=cand.genome,
                canonical_hash=cand.canonical_hash,
                generation=generation,
                strategy_used=self.name,
                parent_hashes=(),
            )
            candidates.append(cand)
            self.proposals_count += 1
        return candidates
