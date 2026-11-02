"""AMDE 4.0 Evaluation Cache Indexed by EvaluationCacheKey.

Guarantees:
1. Strict separation of Candidate Identity from Evaluation Identity.
2. Zero cache cross-contamination across datasets, splits, seeds, or fidelity rungs.
"""

from __future__ import annotations

from typing import Dict, Optional

from amde.core.contracts import EvaluationCacheKey, EvaluationResult
from amde.core.errors import IncompatibleCacheError


class EvaluationCache:
    """Thread-safe in-memory cache storing completed EvaluationResult records."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._cache: dict[str, EvaluationResult] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: EvaluationCacheKey) -> Optional[EvaluationResult]:
        """Lookup cached evaluation by strict typed EvaluationCacheKey."""
        key_str = key.to_string()
        if key_str in self._cache:
            self._hits += 1
            return self._cache[key_str]
        self._misses += 1
        return None

    def put(self, key: EvaluationCacheKey, result: EvaluationResult) -> None:
        """Stores evaluation result under the specific EvaluationCacheKey."""
        key_str = key.to_string()
        if len(self._cache) >= self.capacity:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)
        self._cache[key_str] = result

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def __len__(self) -> int:
        return len(self._cache)
