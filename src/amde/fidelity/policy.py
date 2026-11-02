"""AMDE 4.0 Fidelity Policies, Calibration Safeguards, and Rank-Inversion Guards."""

from __future__ import annotations

import abc
import math
from typing import Any, List, Optional, Tuple
import numpy as np
from scipy import stats

from amde.core.contracts import FidelityLevel


class FidelityPolicy(abc.ABC):
    """Abstract interface governing multi-fidelity resource allocation."""

    @abc.abstractmethod
    def get_initial_fidelity(self, evaluation_count: int) -> FidelityLevel:
        """Determines the fidelity level for initial candidate screening."""
        pass

    @abc.abstractmethod
    def should_promote(
        self,
        score_r0: float,
        evaluation_count: int,
        elite_scores: Sequence[float],
    ) -> bool:
        """Determines whether a candidate passes early screening to Rung 2."""
        pass

    @abc.abstractmethod
    def record_dual_evaluation(self, score_r0: float, score_r2: float) -> None:
        """Registers a paired evaluation to calibrate rank correlation."""
        pass


class FullFidelityPolicy(FidelityPolicy):
    """Production research default: evaluates all candidates at full fidelity (Rung 2)."""

    def get_initial_fidelity(self, evaluation_count: int) -> FidelityLevel:
        return FidelityLevel.full_fidelity()

    def should_promote(
        self, score_r0: float, evaluation_count: int, elite_scores: Sequence[float]
    ) -> bool:
        return True  # Already at full fidelity

    def record_dual_evaluation(self, score_r0: float, score_r2: float) -> None:
        pass


class DelayedPruningPolicy(FidelityPolicy):
    """Calibrated multi-fidelity policy with delayed pruning and automatic rank-inversion fallback.

    Safeguards:
    1. Warmup: First 15 evaluations execute at full fidelity (Rung 2) to build an unbiased elite archive.
    2. Rank-Inversion Guard: If Spearman rank correlation drops below 0.30 on >= 8 dual-evaluated pairs,
       the policy automatically disables early pruning and falls back to full fidelity.
    3. Minimum Promotion Protection: Promotes candidates above median seen performance.
    """

    def __init__(self, warmup_evaluations: int = 15, fallback_rho_threshold: float = 0.30):
        self.warmup_evaluations = warmup_evaluations
        self.fallback_rho_threshold = fallback_rho_threshold
        self.dual_r0: list[float] = []
        self.dual_r2: list[float] = []
        self.fallback_active: bool = False
        self.latest_rho: Optional[float] = None

    def get_initial_fidelity(self, evaluation_count: int) -> FidelityLevel:
        if self.fallback_active or evaluation_count < self.warmup_evaluations:
            return FidelityLevel.full_fidelity()
        return FidelityLevel.screening_fidelity()

    def should_promote(
        self, score_r0: float, evaluation_count: int, elite_scores: Sequence[float]
    ) -> bool:
        if self.fallback_active or evaluation_count < self.warmup_evaluations:
            return True

        if not elite_scores:
            return True

        # Promote if candidate achieves at least the 40th percentile of current elite archive
        threshold = float(np.percentile(list(elite_scores), 40.0))
        return score_r0 >= threshold

    def record_dual_evaluation(self, score_r0: float, score_r2: float) -> None:
        self.dual_r0.append(float(score_r0))
        self.dual_r2.append(float(score_r2))

        # Check rank correlation when sufficient dual pairs exist
        if len(self.dual_r0) >= 8 and not self.fallback_active:
            rho, p_val = stats.spearmanr(self.dual_r0, self.dual_r2)
            if not np.isnan(rho):
                self.latest_rho = float(rho)
                if rho < self.fallback_rho_threshold:
                    # Ranking inversion detected! Automatically fall back to full fidelity
                    self.fallback_active = True
