"""AMDE 4.0 Controller Baselines (Uniform, Random, Static) for Controlled Ablation."""

from __future__ import annotations

import random
from typing import Sequence

from amde.controller.base import MetaController
from amde.core.contracts import ControllerDecision


class UniformController(MetaController):
    """Deterministic round-robin strategy allocation."""

    def select_strategy(self, rng: random.Random, budget_fraction: float = 0.0) -> str:
        self.step_count += 1
        chosen = self.strategy_names[(self.step_count - 1) % len(self.strategy_names)]
        self.history.append(
            ControllerDecision(
                step=self.step_count,
                selected_strategy=chosen,
                bandit_probabilities={name: 1.0 / len(self.strategy_names) for name in self.strategy_names},
                budget_fraction=budget_fraction,
                exploration_mode=True,
            )
        )
        return chosen

    def update_feedback(
        self,
        strategy_name: str,
        val_score: float,
        train_duration: float,
        is_novel: bool,
        is_duplicate: bool,
        is_failure: bool,
    ) -> float:
        return 0.0  # Feedback is ignored in static uniform allocation


class RandomController(MetaController):
    """Stochastic uniform random strategy allocation."""

    def select_strategy(self, rng: random.Random, budget_fraction: float = 0.0) -> str:
        self.step_count += 1
        chosen = rng.choice(self.strategy_names)
        self.history.append(
            ControllerDecision(
                step=self.step_count,
                selected_strategy=chosen,
                bandit_probabilities={name: 1.0 / len(self.strategy_names) for name in self.strategy_names},
                budget_fraction=budget_fraction,
                exploration_mode=True,
            )
        )
        return chosen

    def update_feedback(
        self,
        strategy_name: str,
        val_score: float,
        train_duration: float,
        is_novel: bool,
        is_duplicate: bool,
        is_failure: bool,
    ) -> float:
        return 0.0
