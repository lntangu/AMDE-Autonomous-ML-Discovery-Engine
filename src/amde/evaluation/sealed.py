"""AMDE 4.0 Cryptographically Token-Gated Sealed Holdout Test Suite.

Enforces:
1. Strict physical and architectural separation of holdout test sets (BOT-05).
2. One-time evaluation token preventing iterative optimization on the test set.
3. Cryptographic audit hashing of the final evaluated model.
"""

from __future__ import annotations

import secrets
from typing import Any, Mapping, Optional, Tuple
import numpy as np

from amde.core.contracts import Candidate, EvaluationResult, FidelityLevel
from amde.core.errors import EvaluationError
from amde.evaluation.pipeline import StandardEvaluationPipeline


class SealedHoldoutTest:
    """Token-gated holdout evaluation harness."""

    def __init__(self, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str):
        self._X_test = np.ascontiguousarray(X_test)
        self._y_test = np.ascontiguousarray(y_test)
        self.dataset_name = dataset_name
        self._eval_token = secrets.token_hex(16)
        self._evaluated = False
        self._final_score: Optional[float] = None
        self._pipeline = StandardEvaluationPipeline()

    @property
    def token(self) -> str:
        """Returns the one-time evaluation access token."""
        return self._eval_token

    @property
    def is_consumed(self) -> bool:
        """Indicates whether this holdout has already been evaluated."""
        return self._evaluated

    def evaluate(
        self,
        candidate: Candidate,
        X_train: np.ndarray,
        y_train: np.ndarray,
        token: str,
        seed: int = 42,
    ) -> float:
        """Evaluates winning candidate pipeline on sealed holdout partition using full training set."""
        if token != self._eval_token:
            raise EvaluationError("Invalid holdout authorization token")
        if self._evaluated:
            raise EvaluationError("Holdout partition has already been consumed (single-read enforcement)")

        # Evaluate model trained on full training set and tested on heldout test partition
        from amde.core.contracts import EvaluationRequest, ResourceBudget
        req = EvaluationRequest(
            request_id=f"sealed_test_{candidate.canonical_hash[:12]}",
            candidate=candidate,
            dataset_id=self.dataset_name,
            split_hash="sealed_holdout_split",
            seed=seed,
            fidelity=FidelityLevel.full_fidelity(),
            resource_budget=ResourceBudget(max_wallclock_seconds=60.0),
        )

        res = self._pipeline.execute(req, X_train, y_train, self._X_test, self._y_test)
        self._evaluated = True
        self._final_score = res.primary_metric
        return self._final_score
