"""Safe proposal strategies over the validated genome space."""

from __future__ import annotations

import random
from typing import Any

from ..discovery.factory import random_genome, upgrade_genome
from ..discovery.genome import Genome
from ..discovery.operators import crossover, mutate
from .population import ArchiveManager, Individual, Population
from .open_ended import generate_open_ended_genome

STRATEGIES = ("random", "evolutionary", "crossover", "novelty", "local", "open_ended")
_LEVEL_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


class Proposer:
    def __init__(
        self,
        *,
        population: Population,
        archives: ArchiveManager,
        rng: random.Random,
        levels: list[str],
        safety: dict[str, Any],
    ) -> None:
        self.population = population
        self.archives = archives
        self.rng = rng
        self.levels = [level for level in levels if level in _LEVEL_ORDER] or ["A"]
        self.safety = safety

    def _choose_level(self, *, minimum: str | None = None) -> str:
        candidates = self.levels
        if minimum is not None:
            allowed = [level for level in candidates if _LEVEL_ORDER[level] >= _LEVEL_ORDER[minimum]]
            if allowed:
                candidates = allowed
        return self.rng.choice(candidates)

    def _individual(
        self,
        genome: Genome,
        *,
        generation: int,
        operator: str,
        parents: list[str],
        island: int,
    ) -> Individual:
        return Individual(
            genome=genome,
            generation=generation,
            operator=operator,
            parents=parents,
            island=island % self.population.n_islands,
        )

    def propose(
        self,
        *,
        generation: int,
        strategy: str,
        level_override: str | None = None,
        island: int | None = None,
    ) -> Individual:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown search strategy: {strategy}")
        island_index = int(island if island is not None else self.rng.randrange(self.population.n_islands))
        for attempt in range(96):
            individual = self._propose_once(
                generation=generation + (attempt // 24),
                strategy=strategy,
                level_override=level_override,
                island=island_index,
            )
            if individual.hash not in self.population.seen:
                return individual
        # When a conventional local operator repeatedly hits known structures,
        # escalate to the open-ended grammar. Never silently return a duplicate.
        if "D" in self.levels:
            for attempt in range(256):
                genome, _ = generate_open_ended_genome(
                    self.rng, safety=self.safety, generation=generation + 1 + attempt
                )
                individual = self._individual(
                    genome,
                    generation=generation,
                    operator="open_ended_fallback",
                    parents=[],
                    island=island_index,
                )
                if individual.hash not in self.population.seen:
                    return individual
        raise RuntimeError(
            "no novel candidate found within the configured search budget; "
            "the engine stopped instead of evaluating a duplicate"
        )

    def _propose_once(
        self,
        *,
        generation: int,
        strategy: str,
        level_override: str | None,
        island: int,
    ) -> Individual:
        if strategy == "open_ended" and level_override not in (None, "D"):
            # Respect the configured A->D curriculum. The open-ended DSL arm
            # becomes active only after the caller has opened level D.
            return self._propose_once(
                generation=generation,
                strategy="novelty",
                level_override=level_override,
                island=island,
            )

        if strategy == "open_ended":
            genome, budget = generate_open_ended_genome(
                self.rng,
                safety=self.safety,
                generation=generation,
            )
            return self._individual(
                genome,
                generation=generation,
                operator=f"open_ended_grammar_n{budget.max_nodes}_d{budget.max_depth}",
                parents=[],
                island=island,
            )

        if strategy == "random":
            level = level_override or self._choose_level()
            genome = random_genome(level, self.rng, safety=self.safety)
            return self._individual(genome, generation=generation, operator="random", parents=[], island=island)

        if strategy == "crossover":
            left = self.population.select_parent(self.rng, island=island)
            right = self.population.select_parent(self.rng, island=(island + 1) % self.population.n_islands)
            if left is None or right is None:
                # First-generation bootstrap: crossover two validated random
                # parents. This is a real crossover event, not a random relabel.
                level = level_override or self._choose_level()
                left = random_genome(level, self.rng, safety=self.safety)
                right = random_genome(level, self.rng, safety=self.safety)
            left_genome = left.genome if isinstance(left, Individual) else left
            right_genome = right.genome if isinstance(right, Individual) else right
            if right.hash == left.hash:
                right = self.population.select_parent(self.rng)
                right_genome = right.genome if right is not None and isinstance(right, Individual) else right
            if right is None or right_genome is None or right.hash == left.hash:
                # A true crossover needs two distinct parents. When the
                # population is too small (or both selections landed on the
                # same genome), bootstrap a fresh validated partner rather
                # than relabelling a mutation as crossover.
                level = level_override or self._choose_level(minimum=left.level)
                partner = random_genome(level, self.rng, safety=self.safety)
                for _ in range(3):
                    if partner.hash != left.hash:
                        break
                    partner = random_genome(level, self.rng, safety=self.safety)
                child = crossover(
                    left_genome,
                    partner,
                    self.rng,
                    target_level=level,
                    safety=self.safety,
                )
                return self._individual(
                    child,
                    generation=generation,
                    operator="crossover",
                    parents=sorted({left.hash, partner.hash}),
                    island=island,
                )
            minimum = max((left.level, right.level), key=lambda level: _LEVEL_ORDER[level])
            target = level_override or self._choose_level(minimum=minimum)
            child = crossover(left_genome, right_genome, self.rng, target_level=target, safety=self.safety)
            return self._individual(
                child,
                generation=generation,
                operator="crossover",
                parents=[left.hash, right.hash],
                island=island,
            )

        parent = self.population.select_parent(self.rng, island=island)
        if parent is None:
            # Bootstrap a fresh validated seed, then apply the requested
            # operator. This keeps generation-zero labels honest: local,
            # novelty, and evolutionary proposals are still mutations.
            level = level_override or self._choose_level()
            seed = random_genome(level, self.rng, safety=self.safety)
            if strategy == "local":
                child = mutate(seed, self.rng, target_level=level, safety=self.safety, small=True)
                operator = "local_mutation"
            elif strategy == "novelty":
                child = mutate(seed, self.rng, target_level=level, safety=self.safety)
                operator = "novelty_mutation"
            else:
                child = mutate(seed, self.rng, target_level=level, safety=self.safety)
                operator = "evolutionary_mutation"
            return self._individual(
                child,
                generation=generation,
                operator=operator,
                parents=[seed.hash],
                island=island,
            )

        if strategy == "local":
            target = level_override or (
                parent.level if parent.level in self.levels else self._choose_level(minimum=parent.level)
            )
            child = mutate(parent.genome, self.rng, target_level=target, safety=self.safety, small=True)
            return self._individual(
                child,
                generation=generation,
                operator="local_mutation",
                parents=[parent.hash],
                island=island,
            )

        if strategy == "novelty":
            members = self.population.all()
            if members:
                # Prefer a parent that is not already in a crowded region.
                ranked = sorted(members, key=lambda item: item.novelty, reverse=True)
                parent = self.rng.choice(ranked[: max(1, len(ranked) // 2)])
            target = level_override or self._choose_level(minimum=parent.level)
            child = mutate(parent.genome, self.rng, target_level=target, safety=self.safety)
            return self._individual(
                child,
                generation=generation,
                operator="novelty_mutation",
                parents=[parent.hash],
                island=island,
            )

        # evolutionary
        target = level_override or self._choose_level(minimum=parent.level)
        child = mutate(parent.genome, self.rng, target_level=target, safety=self.safety)
        return self._individual(
            child,
            generation=generation,
            operator="evolutionary_mutation",
            parents=[parent.hash],
            island=island,
        )


def bootstrap_genome(level: str, rng: random.Random, safety: dict[str, Any]) -> Genome:
    """Small helper used by tests and recovery paths."""

    return random_genome(level, rng, safety=safety)


def upgrade(genome: Genome, level: str, rng: random.Random, safety: dict[str, Any]) -> Genome:
    return upgrade_genome(genome, level, rng, safety=safety)
