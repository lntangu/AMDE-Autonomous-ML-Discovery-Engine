"""AMDE 4.0 Quality-Preserving Evolutionary Search Strategy.

Performs:
1. Elitist selection from EliteArchive.
2. Calibrated, bounded mutation preserving pipeline validity (BOT-01).
3. Chromosomal crossover across compatible preprocessing and model blocks.
"""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Sequence

from amde.core.contracts import Candidate
from amde.space.search_space import SearchSpace
from amde.strategy.base import SearchStrategy


class EvolutionaryStrategy(SearchStrategy):
    """Quality-preserving evolutionary search strategy utilizing mutation and crossover."""

    def __init__(self, search_space: SearchSpace, crossover_rate: float = 0.35):
        super().__init__(name="evolutionary_elitist", search_space=search_space)
        self.crossover_rate = crossover_rate

    @property
    def required_information(self) -> Sequence[str]:
        return ("elite_archive",)

    @property
    def computational_overhead_ms(self) -> float:
        return 0.35

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

        # If elite archive is empty or has fewer than 2 candidates, fall back to random sampling
        elite_entries = elite_archive.entries if elite_archive is not None else []

        for _ in range(n):
            if len(elite_entries) >= 2 and rng.random() < self.crossover_rate:
                # Perform crossover
                parents = rng.sample(elite_entries, 2)
                cand_a = parents[0]["candidate"]
                cand_b = parents[1]["candidate"]
                child = self.search_space.crossover_candidates(cand_a, cand_b, rng, generation=generation)
                candidates.append(child)
            elif len(elite_entries) >= 1:
                # Perform calibrated mutation
                parent = rng.choice(elite_entries)["candidate"]
                child = self.search_space.mutate_candidate(parent, rng, generation=generation, scale=0.15)
                candidates.append(child)
            else:
                child = self.search_space.sample_random(rng)
                candidates.append(child)
            self.proposals_count += 1

        return candidates
