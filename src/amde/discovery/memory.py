"""AMDE 3.0 Multi-Tier Memory & Archive System.

Includes:
- Elite Archive: Preserves top-performing candidates (Guarantees Elitism, fixes BOT-01)
- Pareto Archive: Tracks non-dominated frontier across multiple objectives
- Novelty Archive: Preserves behavioral diversity
- Negative Memory: Spatial index of failed parameter regions (fixes BOT-03)
- Candidate Deduplicator: Cryptographic hash ledger preventing recycling (fixes BOT-02)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Set, Tuple


class CandidateDeduplicator:
    """Tracks evaluated candidate hashes to prevent static recycling (BOT-02)."""

    def __init__(self):
        self._seen_hashes: set[str] = set()
        self._duplicate_attempts: int = 0
        self._total_proposals: int = 0

    def is_duplicate(self, candidate_hash: str) -> bool:
        self._total_proposals += 1
        if candidate_hash in self._seen_hashes:
            self._duplicate_attempts += 1
            return True
        return False

    def has_seen(self, candidate_hash: str) -> bool:
        """Checks if candidate_hash has already been registered without modifying counters."""
        return candidate_hash in self._seen_hashes

    def register(self, candidate_hash: str) -> None:
        self._seen_hashes.add(candidate_hash)

    @property
    def duplicate_rate(self) -> float:
        if self._total_proposals == 0:
            return 0.0
        return self._duplicate_attempts / self._total_proposals

    @property
    def total_unique(self) -> int:
        return len(self._seen_hashes)


class EliteArchive:
    """Preserves the top-K highest-performing candidates, guaranteeing elitism."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.entries: list[dict[str, Any]] = []

    def add(self, candidate_dict: dict[str, Any], score: float) -> bool:
        """Adds a candidate if it improves upon the archive or capacity allows."""
        candidate_hash = candidate_dict["canonical_hash"]
        for entry in self.entries:
            if entry["canonical_hash"] == candidate_hash:
                if score > entry["score"]:
                    entry["score"] = score
                    entry["candidate"] = copy.deepcopy(candidate_dict)
                return False

        item = {
            "canonical_hash": candidate_hash,
            "score": score,
            "candidate": copy.deepcopy(candidate_dict),
        }
        self.entries.append(item)
        self.entries.sort(key=lambda x: x["score"], reverse=True)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[: self.capacity]
        return True

    def get_best(self) -> Optional[dict[str, Any]]:
        return self.entries[0] if self.entries else None

    def get_top_k(self, k: int) -> list[dict[str, Any]]:
        return self.entries[:k]

    def sample_parent(self, rng) -> Optional[dict[str, Any]]:
        """Tournament selection among elites."""
        if not self.entries:
            return None
        pool_size = min(4, len(self.entries))
        pool = rng.sample(self.entries[:pool_size], min(2, pool_size))
        pool.sort(key=lambda x: x["score"], reverse=True)
        return pool[0]["candidate"]


class ParetoArchive:
    """Maintains the non-dominated multi-objective frontier (Accuracy, Latency, Memory)."""

    def __init__(self):
        # Each item: {"candidate": dict, "objectives": [loss, latency, memory]}
        # All objectives are to be minimized.
        self.frontier: list[dict[str, Any]] = []

    def update(self, candidate_dict: dict[str, Any], loss: float, latency: float, memory: float) -> bool:
        """Inserts candidate if it is non-dominated, and prunes newly dominated entries."""
        new_point = [loss, latency, memory]

        # Check if new point is dominated by any existing point
        for item in self.frontier:
            if self._dominates(item["objectives"], new_point):
                return False

        # Remove points dominated by the new point
        self.frontier = [item for item in self.frontier if not self._dominates(new_point, item["objectives"])]
        self.frontier.append({
            "canonical_hash": candidate_dict["canonical_hash"],
            "candidate": copy.deepcopy(candidate_dict),
            "objectives": new_point,
        })
        return True

    @staticmethod
    def _dominates(a: list[float], b: list[float]) -> bool:
        """Returns True if point a dominates point b (minimization)."""
        all_le = all(x <= y for x, y in zip(a, b))
        any_lt = any(x < y for x, y in zip(a, b))
        return all_le and any_lt


class NegativeMemory:
    """Spatial index of parameter failure basins (BOT-03 fix).
    
    Prevents allocating compute to parameter regions known to produce
    numerical instability, gradient explosions, or severe underfitting.
    """

    def __init__(self, radius: float = 0.15):
        self.radius = radius
        self.failure_points: list[list[float]] = []

    def extract_features(self, genome: dict[str, Any]) -> list[float]:
        """Extracts normalized continuous parameters for distance indexing."""
        model = genome.get("model", {})
        family_idx = {"mlp": 0.0, "tree_ensemble": 0.33, "kernel_ridge": 0.66, "linear": 1.0}.get(model.get("family", ""), 0.0)
        lr = math.log10(model.get("learning_rate", 1e-3))
        # Normalize lr between -5 and 0 -> [0, 1]
        norm_lr = (lr + 5.0) / 5.0
        return [family_idx, norm_lr]

    def record_failure(self, genome: dict[str, Any]) -> None:
        feats = self.extract_features(genome)
        self.failure_points.append(feats)

    def is_in_failure_basin(self, genome: dict[str, Any]) -> bool:
        if not self.failure_points:
            return False
        feats = self.extract_features(genome)
        for point in self.failure_points:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(feats, point)))
            if dist < self.radius:
                return True
        return False
