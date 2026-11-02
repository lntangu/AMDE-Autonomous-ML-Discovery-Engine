"""Populations, Pareto archive, novelty archive, and MAP-Elites cells."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..discovery.genome import Genome
from ..util import clamp


def _slim_aggregate(aggregate: dict[str, Any] | None) -> dict[str, Any] | None:
    if aggregate is None:
        return None
    allowed = (
        "mean",
        "median",
        "std_tasks",
        "std_families",
        "lower_quartile_family_score",
        "std_seeds",
        "ci_low",
        "ci_high",
        "ci_level",
        "worst_family",
        "worst_family_score",
        "family_scores",
        "per_task",
        "seed_aggregates",
        "n_tasks",
        "n_families",
        "n_seeds",
        "metric_version",
        "metric_hash",
    )
    return {key: aggregate[key] for key in allowed if key in aggregate}


@dataclass
class Individual:
    genome: Genome
    generation: int
    operator: str = "seed"
    parents: list[str] = field(default_factory=list)
    island: int = 0
    quick_score: float | None = None
    confirm_score: float | None = None
    quick_aggregate: dict[str, Any] | None = None
    confirm_aggregate: dict[str, Any] | None = None
    status: str = "proposed"
    novelty: float = 0.0

    @property
    def hash(self) -> str:
        return self.genome.hash

    @property
    def level(self) -> str:
        return self.genome.level

    @property
    def complexity(self) -> float:
        return self.genome.complexity()

    @property
    def rank_score(self) -> float:
        if self.confirm_score is not None:
            return float(self.confirm_score)
        if self.quick_score is not None:
            return float(self.quick_score)
        return -1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "candidate_hash": self.hash,
            "generation": int(self.generation),
            "operator": self.operator,
            "parents": list(self.parents),
            "island": int(self.island),
            "quick_score": self.quick_score,
            "confirm_score": self.confirm_score,
            "quick_aggregate": _slim_aggregate(self.quick_aggregate),
            "confirm_aggregate": _slim_aggregate(self.confirm_aggregate),
            "status": self.status,
            "novelty": float(self.novelty),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, safety: dict[str, Any]) -> "Individual":
        from ..discovery.factory import _validate_kwargs

        genome = Genome(data["genome"], **_validate_kwargs(safety))
        if genome.hash != data["candidate_hash"]:
            raise ValueError("candidate hash does not match genome")
        return cls(
            genome=genome,
            generation=int(data["generation"]),
            operator=str(data.get("operator", "unknown")),
            parents=[str(item) for item in data.get("parents", [])],
            island=int(data.get("island", 0)),
            quick_score=data.get("quick_score"),
            confirm_score=data.get("confirm_score"),
            quick_aggregate=data.get("quick_aggregate"),
            confirm_aggregate=data.get("confirm_aggregate"),
            status=str(data.get("status", "proposed")),
            novelty=float(data.get("novelty", 0.0)),
        )


class Population:
    def __init__(
        self,
        *,
        capacity: int,
        islands: int,
        elite_size: int,
        safety: dict[str, Any],
    ) -> None:
        self.capacity = max(2, int(capacity))
        self.n_islands = max(1, int(islands))
        self.elite_size = max(1, min(int(elite_size), self.capacity))
        self.safety = safety
        self.islands: list[list[Individual]] = [[] for _ in range(self.n_islands)]
        self.members: dict[str, Individual] = {}
        self.seen: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, individual: Individual) -> bool:
        if individual.hash in self.members:
            return False
        self.seen.add(individual.hash)
        island = self.islands[individual.island % self.n_islands]
        island.append(individual)
        self._sort_island(island)
        self.members[individual.hash] = individual
        if len(island) > self.capacity:
            removed = island[self.capacity :]
            del island[self.capacity :]
            for item in removed:
                if self.members.get(item.hash) is item:
                    self.members.pop(item.hash, None)
        return True

    def update(self, individual: Individual) -> None:
        if individual.hash not in self.members:
            self.add(individual)
            return
        self.members[individual.hash] = individual
        for island in self.islands:
            for idx, current in enumerate(island):
                if current.hash == individual.hash:
                    island[idx] = individual
                    break
            self._sort_island(island)

    def update_confirm(self, candidate_hash: str, aggregate: dict[str, Any]) -> None:
        individual = self.members.get(candidate_hash)
        if individual is None:
            return
        individual.confirm_score = float(aggregate["mean"])
        individual.confirm_aggregate = _slim_aggregate(aggregate)
        individual.status = "confirmed"
        self.update(individual)

    def update_quick(self, candidate_hash: str, aggregate: dict[str, Any], novelty: float = 0.0) -> None:
        individual = self.members.get(candidate_hash)
        if individual is None:
            return
        individual.quick_score = float(aggregate["mean"])
        individual.quick_aggregate = _slim_aggregate(aggregate)
        individual.novelty = float(novelty)
        individual.status = "quick_ok"
        self.update(individual)

    def _sort_island(self, island: list[Individual]) -> None:
        island.sort(key=lambda item: (-item.rank_score, item.complexity, item.hash))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def all(self) -> list[Individual]:
        by_hash = {}
        for island in self.islands:
            for individual in island:
                by_hash[individual.hash] = individual
        return list(by_hash.values())

    def best(self) -> Individual | None:
        members = self.all()
        if not members:
            return None
        return max(members, key=lambda item: (item.rank_score, -item.complexity))

    def select_parent(self, rng: random.Random, *, island: int | None = None) -> Individual | None:
        if island is None:
            pool = self.all()
        else:
            pool = self.islands[island % self.n_islands]
        if not pool:
            pool = self.all()
        if not pool:
            return None
        if rng.random() < 0.15:
            return rng.choice(pool)
        contenders = rng.sample(pool, k=min(3, len(pool)))
        return max(contenders, key=lambda item: (item.rank_score, -item.complexity))

    def elites(self, count: int | None = None) -> list[Individual]:
        count = int(count or self.elite_size)
        return sorted(self.all(), key=lambda item: (-item.rank_score, item.complexity))[:count]

    def island_members(self, island: int) -> list[Individual]:
        return list(self.islands[island % self.n_islands])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "islands": self.n_islands,
            "elite_size": self.elite_size,
            "seen": sorted(self.seen),
            "members": [item.to_dict() for item in self.all()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, safety: dict[str, Any]) -> "Population":
        population = cls(
            capacity=int(data["capacity"]),
            islands=int(data["islands"]),
            elite_size=int(data["elite_size"]),
            safety=safety,
        )
        population.seen = set(str(item) for item in data.get("seen", []))
        for raw in data.get("members", []):
            individual = Individual.from_dict(raw, safety=safety)
            island = population.islands[individual.island % population.n_islands]
            island.append(individual)
            population.members[individual.hash] = individual
        for island in population.islands:
            population._sort_island(island)
        return population


class ParetoArchive:
    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(2, int(capacity))
        self.entries: dict[str, dict[str, Any]] = {}

    def add(self, individual: Individual) -> bool:
        score = individual.rank_score
        if score < 0:
            return False
        payload = {
            "candidate_hash": individual.hash,
            "score": float(score),
            "complexity": float(individual.complexity),
            "level": individual.level,
            "generation": int(individual.generation),
        }
        for current in self.entries.values():
            if (
                current["score"] >= payload["score"] - 1e-12
                and current["complexity"] <= payload["complexity"] + 1e-12
                and (
                    current["score"] > payload["score"] + 1e-12
                    or current["complexity"] < payload["complexity"] - 1e-12
                )
            ):
                return False
        dominated = [
            key
            for key, current in self.entries.items()
            if payload["score"] >= current["score"] - 1e-12
            and payload["complexity"] <= current["complexity"] + 1e-12
            and (
                payload["score"] > current["score"] + 1e-12
                or payload["complexity"] < current["complexity"] - 1e-12
            )
        ]
        for key in dominated:
            self.entries.pop(key, None)
        self.entries[individual.hash] = payload
        if len(self.entries) > self.capacity:
            ordered = sorted(
                self.entries.values(),
                key=lambda item: (-item["score"], item["complexity"], item["candidate_hash"]),
            )
            keep: list[dict[str, Any]] = []
            n = len(ordered)
            for idx in range(self.capacity):
                position = round(idx * (n - 1) / max(1, self.capacity - 1))
                keep.append(ordered[position])
            self.entries = {item["candidate_hash"]: item for item in keep}
        return True

    def hashes(self) -> list[str]:
        return sorted(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "entries": [self.entries[key] for key in sorted(self.entries)]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParetoArchive":
        archive = cls(capacity=int(data.get("capacity", 64)))
        for entry in data.get("entries", []):
            archive.entries[str(entry["candidate_hash"])] = dict(entry)
        return archive


def behavior_descriptor(genome: Genome) -> list[float]:
    data = genome.data
    model = data["model"]
    recipe = data["recipe"]
    optimizer = data["optimizer"]
    dsl_nodes = sum(
        _dsl_nodes(data["dsl"][key])
        for key in ("w_update", "b_update", "m_update", "v_update")
        if data["dsl"][key] is not None
    )
    # Keep descriptor coordinates normalized as the candidate schema expands.
    opt_index = ("sgd", "momentum", "adagrad", "rmsprop", "adam", "nadam", "adamw", "lion", "dsl").index(optimizer["kind"])
    init_index = ("zeros", "uniform", "normal").index(recipe["init"])
    return [
        {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}[genome.level] / 3.0,
        clamp((math.log10(max(1e-12, float(recipe["learning_rate"]))) + 4.0) / 4.0, 0.0, 1.0),
        clamp((float(recipe["epochs"]) - 2.0) / 78.0, 0.0, 1.0),
        clamp(float(recipe["batch_size"]) / 128.0, 0.0, 1.0),
        len(model["hidden"]) / 3.0,
        max(model["hidden"], default=0) / 48.0,
        opt_index / 8.0,
        clamp(dsl_nodes / 128.0, 0.0, 1.0),
        1.0 if recipe["standardize"] else 0.0,
        init_index / 2.0,
        clamp(float(recipe["l2"]) / 0.25, 0.0, 1.0),
    ]


def _dsl_nodes(expr: Any) -> int:
    if not isinstance(expr, dict):
        return 0
    if expr.get("op") in ("const", "var"):
        return 1
    return 1 + sum(_dsl_nodes(arg) for arg in expr.get("args", []))


def behavior_cell(descriptor: Iterable[float]) -> str:
    values = list(descriptor)
    bins = [
        min(3, int(values[0] * 3.999)),
        min(3, int(values[1] * 3.999)),
        min(3, int(values[2] * 3.999)),
        min(3, int(values[5] * 3.999)),
        min(4, int(values[6] * 4.999)),
        min(2, int(values[7] * 2.999)),
        1 if values[8] >= 0.5 else 0,
        min(2, int(values[9] * 2.999)),
    ]
    return "cell-" + "-".join(str(value) for value in bins)


class NoveltyArchive:
    def __init__(self, capacity: int = 128, k: int = 5) -> None:
        self.capacity = max(2, int(capacity))
        self.k = max(1, int(k))
        self.hashes: list[str] = []
        self.descriptors: dict[str, list[float]] = {}
        self.cells: dict[str, dict[str, Any]] = {}

    def novelty(self, descriptor: list[float], *, exclude_hash: str | None = None) -> float:
        if not self.descriptors:
            return 1.0
        distances = sorted(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptor, other)))
            for other_hash, other in self.descriptors.items()
            if other_hash != exclude_hash
        )
        if not distances:
            return 1.0
        nearest = distances[: self.k]
        return sum(nearest) / len(nearest)

    def add(self, candidate_hash: str, descriptor: list[float], score: float) -> dict[str, Any]:
        novelty = self.novelty(descriptor, exclude_hash=candidate_hash)
        if candidate_hash not in self.descriptors:
            self.hashes.append(candidate_hash)
        self.descriptors[candidate_hash] = list(descriptor)
        while len(self.hashes) > self.capacity:
            evicted = self.hashes.pop(0)
            if evicted != candidate_hash:
                self.descriptors.pop(evicted, None)
        cell = behavior_cell(descriptor)
        previous = self.cells.get(cell)
        improved = previous is None or float(score) > float(previous["score"])
        if improved:
            self.cells[cell] = {
                "candidate_hash": candidate_hash,
                "score": float(score),
                "descriptor": list(descriptor),
            }
        return {
            "novelty": float(novelty),
            "cell": cell,
            "new_cell": previous is None,
            "cell_improved": bool(improved),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "k": self.k,
            "hashes": list(self.hashes),
            "descriptors": {key: list(value) for key, value in self.descriptors.items()},
            "cells": {key: dict(value) for key, value in self.cells.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoveltyArchive":
        archive = cls(capacity=int(data.get("capacity", 128)), k=int(data.get("k", 5)))
        archive.hashes = [str(item) for item in data.get("hashes", [])]
        archive.descriptors = {
            str(key): [float(value) for value in values]
            for key, values in data.get("descriptors", {}).items()
        }
        archive.cells = {str(key): dict(value) for key, value in data.get("cells", {}).items()}
        return archive


class ArchiveManager:
    def __init__(self, *, pareto_capacity: int = 64, novelty_capacity: int = 128, k: int = 5) -> None:
        self.pareto = ParetoArchive(pareto_capacity)
        self.novelty = NoveltyArchive(novelty_capacity, k)

    def observe(self, individual: Individual) -> dict[str, Any]:
        descriptor = behavior_descriptor(individual.genome)
        map_result = self.novelty.add(individual.hash, descriptor, max(0.0, individual.rank_score))
        individual.novelty = map_result["novelty"]
        pareto_added = self.pareto.add(individual)
        return {
            "descriptor": descriptor,
            "novelty": map_result["novelty"],
            "cell": map_result["cell"],
            "new_cell": map_result["new_cell"],
            "cell_improved": map_result["cell_improved"],
            "pareto_added": pareto_added,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"pareto": self.pareto.to_dict(), "novelty": self.novelty.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveManager":
        manager = cls()
        manager.pareto = ParetoArchive.from_dict(data.get("pareto", {}))
        manager.novelty = NoveltyArchive.from_dict(data.get("novelty", {}))
        return manager
