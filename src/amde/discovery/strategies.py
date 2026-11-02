"""AMDE 3.0 Multi-Strategy Search Portfolio.

Implements the BaseDiscoveryStrategy interface and concrete strategies:
- RandomStrategy
- TPEStrategy
- QualityPreservingEvolutionStrategy (fixes BOT-01, BOT-03)
- LocalRefinementStrategy
- SurrogateAcquisitionStrategy
"""

from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from amde.discovery.memory import EliteArchive, NegativeMemory
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash
from amde.discovery.surrogate import CalibratedSurrogate


class BaseDiscoveryStrategy(ABC):
    """Abstract interface for all discovery search strategies."""

    def __init__(self, name: str, search_space: SearchSpace):
        self.name = name
        self.search_space = search_space
        self.proposals_count = 0
        self.improvements_count = 0

    @abstractmethod
    def propose(
        self,
        n: int,
        rng: random.Random,
        elite_archive: EliteArchive,
        surrogate: CalibratedSurrogate,
        negative_memory: NegativeMemory,
        generation: int = 1,
    ) -> list[Candidate]:
        pass

    def record_feedback(self, candidate_hash: str, score: float, improved: bool) -> None:
        if improved:
            self.improvements_count += 1


class RandomStrategy(BaseDiscoveryStrategy):
    """Unbiased global random explorer."""

    def __init__(self, search_space: SearchSpace):
        super().__init__("random_search", search_space)

    def propose(
        self,
        n: int,
        rng: random.Random,
        elite_archive: EliteArchive,
        surrogate: CalibratedSurrogate,
        negative_memory: NegativeMemory,
        generation: int = 1,
    ) -> list[Candidate]:
        candidates = []
        for _ in range(n):
            genome = self.search_space.sample_random(rng)
            c_hash = compute_genome_hash(genome)
            c = Candidate(
                genome=genome,
                canonical_hash=c_hash,
                parents=[],
                operator="sample_random",
                strategy=self.name,
                generation=generation,
                is_novel=True,
            )
            candidates.append(c)
            self.proposals_count += 1
        return candidates


class QualityPreservingEvolutionStrategy(BaseDiscoveryStrategy):
    """Elitist evolutionary strategy with calibrated, non-destructive mutation.
    
    Fixes BOT-01 (Generational Degradation) and BOT-03 (Unguided Mutation):
    - Tournament selection exclusively from EliteArchive.
    - Gaussian log-normal hyperparameter perturbation.
    - Negative memory rejection prior to evaluation.
    """

    def __init__(self, search_space: SearchSpace):
        super().__init__("evolutionary_elitist", search_space)

    def propose(
        self,
        n: int,
        rng: random.Random,
        elite_archive: EliteArchive,
        surrogate: CalibratedSurrogate,
        negative_memory: NegativeMemory,
        generation: int = 1,
    ) -> list[Candidate]:
        candidates = []
        for _ in range(n):
            parent = elite_archive.sample_parent(rng)
            if parent is None:
                # Cold start: sample random candidate
                genome = self.search_space.sample_random(rng)
                parent_hash = []
            else:
                parent_genome = parent["genome"]
                parent_hash = [parent["canonical_hash"]]
                # Attempt up to 5 calibrated mutations that don't land in negative memory
                genome = parent_genome
                for _ in range(5):
                    candidate_genome = self.search_space.mutate_calibrated(
                        parent_genome, rng, generation=generation
                    )
                    if negative_memory is None or not negative_memory.is_in_failure_basin(candidate_genome):
                        genome = candidate_genome
                        break

            c_hash = compute_genome_hash(genome)
            c = Candidate(
                genome=genome,
                canonical_hash=c_hash,
                parents=parent_hash,
                operator="calibrated_mutation",
                strategy=self.name,
                generation=generation,
                is_novel=True,
            )
            candidates.append(c)
            self.proposals_count += 1
        return candidates


class LocalRefinementStrategy(BaseDiscoveryStrategy):
    """Coordinate descent fine-polishing on incumbent elite candidates."""

    def __init__(self, search_space: SearchSpace):
        super().__init__("local_refinement", search_space)

    def propose(
        self,
        n: int,
        rng: random.Random,
        elite_archive: EliteArchive,
        surrogate: CalibratedSurrogate,
        negative_memory: NegativeMemory,
        generation: int = 1,
    ) -> list[Candidate]:
        candidates = []
        best_entry = elite_archive.get_best()
        if best_entry is None:
            # Fallback to random if archive is empty
            genome = self.search_space.sample_random(rng)
            return [Candidate(genome=genome, canonical_hash=compute_genome_hash(genome), strategy=self.name)]

        base_genome = best_entry["candidate"]["genome"]
        for _ in range(n):
            mutated = copy.deepcopy(base_genome)
            model = mutated["model"]
            family = model.get("family")

            # Small 5% to 10% local perturbation
            if family == "mlp" and "learning_rate" in model:
                factor = rng.choice([0.9, 0.95, 1.05, 1.1])
                model["learning_rate"] = round(max(1e-5, min(0.5, model["learning_rate"] * factor)), 6)
            elif family == "tree_ensemble" and "learning_rate" in model:
                factor = rng.choice([0.9, 0.95, 1.05, 1.1])
                model["learning_rate"] = round(max(1e-3, min(0.3, model["learning_rate"] * factor)), 5)
            elif family == "kernel_ridge" and "alpha" in model:
                factor = rng.choice([0.8, 1.25])
                model["alpha"] = round(max(1e-5, min(10.0, model["alpha"] * factor)), 6)

            c_hash = compute_genome_hash(mutated)
            candidates.append(Candidate(
                genome=mutated,
                canonical_hash=c_hash,
                parents=[best_entry["canonical_hash"]],
                operator="local_coordinate_step",
                strategy=self.name,
                generation=generation,
                is_novel=True,
            ))
            self.proposals_count += 1
        return candidates


class SurrogateAcquisitionStrategy(BaseDiscoveryStrategy):
    """Acquires candidates that maximize Expected Improvement on the surrogate model."""

    def __init__(self, search_space: SearchSpace):
        super().__init__("surrogate_ei", search_space)

    def propose(
        self,
        n: int,
        rng: random.Random,
        elite_archive: EliteArchive,
        surrogate: CalibratedSurrogate,
        negative_memory: NegativeMemory,
        generation: int = 1,
    ) -> list[Candidate]:
        candidates = []
        for _ in range(n):
            if not surrogate.fitted:
                # If surrogate is not fitted yet, sample random candidate
                genome = self.search_space.sample_random(rng)
            else:
                # Generate 20 candidate pool proposals and pick argmax EI
                pool = [self.search_space.sample_random(rng) for _ in range(20)]
                scores = [surrogate.acquisition_ei(g) for g in pool]
                best_idx = int(scores.index(max(scores)))
                genome = pool[best_idx]

            c_hash = compute_genome_hash(genome)
            candidates.append(Candidate(
                genome=genome,
                canonical_hash=c_hash,
                parents=[],
                operator="surrogate_acquisition_ei",
                strategy=self.name,
                generation=generation,
                is_novel=True,
            ))
            self.proposals_count += 1
        return candidates
