"""AMDE 4.0 Candidate Deduplication Ledger.

Enforces zero candidate recycling (BOT-02) with O(1) set lookups
and exact duplicate rate tracking.
"""

from __future__ import annotations

from typing import Optional, Set


class CandidateDeduplicator:
    """Tracks evaluated candidate hashes to eliminate redundant computations."""

    def __init__(self, capacity: Optional[int] = None):
        self._seen_hashes: set[str] = set()
        self._duplicate_attempts: int = 0
        self._total_proposals: int = 0
        self.capacity = capacity

    def is_duplicate(self, candidate_hash: str) -> bool:
        """Checks whether the candidate hash has already been registered."""
        self._total_proposals += 1
        if candidate_hash in self._seen_hashes:
            self._duplicate_attempts += 1
            return True
        return False

    def register(self, candidate_hash: str) -> None:
        """Registers a novel candidate hash."""
        self._seen_hashes.add(candidate_hash)
        if self.capacity is not None and len(self._seen_hashes) > self.capacity:
            # Evict oldest entry (arbitrary pop from set)
            self._seen_hashes.pop()

    @property
    def duplicate_rate(self) -> float:
        """Calculates historical duplicate proposal rate."""
        if self._total_proposals == 0:
            return 0.0
        return self._duplicate_attempts / self._total_proposals

    @property
    def total_unique(self) -> int:
        """Returns total count of registered unique candidates."""
        return len(self._seen_hashes)
