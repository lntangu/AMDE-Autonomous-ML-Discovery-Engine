"""AMDE 4.0 Controller Contract and Abstract Base Class."""

from __future__ import annotations

import abc
import random
from typing import Any, Mapping, Sequence

from amde.core.contracts import ControllerDecision


class MetaController(abc.ABC):
    """Abstract interface for adaptive strategy allocation controllers."""

    def __init__(self, strategy_names: Sequence[str]):
        self.strategy_names = list(strategy_names)
        self.step_count: int = 0
        self.history: list[ControllerDecision] = []

    @abc.abstractmethod
    def select_strategy(self, rng: random.Random, budget_fraction: float = 0.0) -> str:
        """Selects the next strategy to generate candidates."""
        pass

    @abc.abstractmethod
    def update_feedback(
        self,
        strategy_name: str,
        val_score: float,
        train_duration: float,
        is_novel: bool,
        is_duplicate: bool,
        is_failure: bool,
    ) -> float:
        """Incorporates evaluation feedback and returns the calculated bounded reward."""
        pass

    def get_allocation_shares(self) -> dict[str, float]:
        """Calculates relative historical allocation proportions of each strategy."""
        counts = {name: 0 for name in self.strategy_names}
        for decision in self.history:
            counts[decision.selected_strategy] += 1
        total = max(1, len(self.history))
        return {k: round(v / total, 4) for k, v in counts.items()}
