"""AMDE 4.0 Elite Candidate Archive.

Preserves top-performing candidate pipelines to guarantee monotonic generational elitism (BOT-01)
and provide seed candidates for mutation and crossover operators.
"""

from __future__ import annotations

import copy
from typing import Any, List, Mapping, Optional

from amde.core.contracts import Candidate


class EliteArchive:
    """Bounded priority archive retaining top-k evaluated candidates."""

    def __init__(self, capacity: int = 16):
        self.capacity = max(1, capacity)
        self.entries: list[dict[str, Any]] = []

    def add(self, candidate: Candidate, score: float) -> bool:
        """Adds a candidate if it qualifies for the top-k archive."""
        import math
        try:
            fscore = float(score)
        except (ValueError, TypeError):
            return False
        if not math.isfinite(fscore):
            return False

        # Check if already present by hash
        for item in self.entries:
            if item["candidate"].canonical_hash == candidate.canonical_hash:
                # Update score if better
                if fscore > item["score"]:
                    item["score"] = fscore
                    self._sort_and_trim()
                    return True
                return False

        # Add candidate
        self.entries.append({
            "candidate": candidate,
            "score": fscore,
            "hash": candidate.canonical_hash,
        })
        self._sort_and_trim()
        return any(item["hash"] == candidate.canonical_hash for item in self.entries)

    def _sort_and_trim(self) -> None:
        """Sorts entries descending by score and clamps to capacity."""
        self.entries.sort(key=lambda x: x["score"], reverse=True)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[:self.capacity]

    @property
    def best_candidate(self) -> Optional[Candidate]:
        if not self.entries:
            return None
        return self.entries[0]["candidate"]

    @property
    def best_score(self) -> float:
        if not self.entries:
            return -float("inf")
        return self.entries[0]["score"]

    def __len__(self) -> int:
        return len(self.entries)
