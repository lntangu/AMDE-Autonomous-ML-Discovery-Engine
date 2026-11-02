"""AMDE 4.0 Coordinate Descent Local Refinement Strategy.

Applies fine-grained perturbations to continuous parameters and hyperparameters
of the incumbent elite pipeline to locate local optima in late-stage search.
"""

from __future__ import annotations

import copy
import random
from typing import Any, List, Mapping, Sequence

from amde.core.contracts import Candidate
from amde.core.identity import compute_candidate_hash
from amde.space.search_space import SearchSpace
from amde.strategy.base import SearchStrategy


class LocalRefinementStrategy(SearchStrategy):
    """Fine-polishing coordinate descent on top-ranked incumbent candidates."""

    def __init__(self, search_space: SearchSpace, step_scale: float = 0.08):
        super().__init__(name="local_refinement", search_space=search_space)
        self.step_scale = step_scale

    @property
    def required_information(self) -> Sequence[str]:
        return ("elite_archive",)

    @property
    def computational_overhead_ms(self) -> float:
        return 0.25

    def propose(
        self,
        n: int,
        rng: random.Random,
        archives: Mapping[str, Any],
        budget_fraction: float = 0.0,
        generation: int = 0,
    ) -> List[Candidate]:
        elite_archive = archives.get("elite_archive")
        candidates = []
        elite_entries = elite_archive.entries if elite_archive is not None else []

        for _ in range(n):
            if not elite_entries:
                child = self.search_space.sample_random(rng)
            else:
                # Target the best candidate in elite archive
                best_cand = elite_entries[0]["candidate"]
                child = self.search_space.mutate_candidate(
                    best_cand, rng, generation=generation, scale=self.step_scale
                )
                child = Candidate(
                    genome=child.genome,
                    canonical_hash=child.canonical_hash,
                    generation=generation,
                    strategy_used=self.name,
                    parent_hashes=(best_cand.canonical_hash,),
                )
            candidates.append(child)
            self.proposals_count += 1

        return candidates
