"""AMDE 3.0 Evidence-Guided Autonomous Discovery Architecture (EGADA) Controller.

Maintains non-stationary contextual multi-armed bandit state, reward-per-compute
weighting, duplicate penalties, and dynamic strategy allocation.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple


class EGADAController:
    """Evidence-Guided Autonomous Discovery Controller using Discounted Thompson Sampling."""

    def __init__(self, strategy_names: list[str], discount_factor: float = 0.95):
        self.strategy_names = strategy_names
        self.gamma = discount_factor

        # Bandit priors: mu_0 = 0.5, sigma_0 = 0.5
        self.stats: dict[str, dict[str, float]] = {
            name: {"discounted_n": 1.0, "discounted_reward": 0.5, "variance": 0.25}
            for name in strategy_names
        }
        self.incumbent_score: float = -float("inf")
        self.step_count: int = 0
        self.history: list[dict[str, Any]] = []

    def select_strategy(self, rng: random.Random, budget_fraction: float = 0.0) -> str:
        """Selects the next strategy using Discounted Thompson Sampling with phase priors."""
        self.step_count += 1

        # Phase 1: Pure exploration warm-up (first 5 steps)
        if self.step_count <= len(self.strategy_names):
            return self.strategy_names[(self.step_count - 1) % len(self.strategy_names)]

        # Phase 3: Late-stage polishing (budget > 85%)
        if budget_fraction > 0.85 and "local_refinement" in self.strategy_names:
            if rng.random() < 0.60:
                return "local_refinement"

        # Thompson Sampling draw:
        samples = {}
        for name in self.strategy_names:
            st = self.stats[name]
            # Standard error of mean: sigma / sqrt(n)
            se = math.sqrt(st["variance"] / max(1.0, st["discounted_n"]))
            sample_val = rng.gauss(st["discounted_reward"], max(0.05, se))
            samples[name] = sample_val

        best_strategy = max(samples.keys(), key=lambda k: samples[k])
        return best_strategy

    def update_feedback(
        self,
        strategy_name: str,
        val_score: float,
        train_duration: float,
        is_novel: bool,
        is_duplicate: bool,
        is_failure: bool,
    ) -> float:
        """Updates bandit posterior based on observed trial outcome."""
        if strategy_name not in self.stats:
            return 0.0

        # Calculate reward
        if is_duplicate:
            reward = -1.0
        elif is_failure:
            reward = -0.5
        else:
            improvement = max(0.0, val_score - self.incumbent_score)
            novelty_bonus = 0.10 if is_novel else 0.0
            # Reward normalized by square root of compute duration
            time_norm = math.sqrt(max(0.05, train_duration))
            reward = (improvement / time_norm) + novelty_bonus

            if val_score > self.incumbent_score:
                self.incumbent_score = val_score

        # Apply exponential discount to past observations
        for name in self.strategy_names:
            self.stats[name]["discounted_n"] *= self.gamma
            self.stats[name]["discounted_reward"] *= self.gamma

        # Incorporate new sample
        st = self.stats[strategy_name]
        st["discounted_n"] += 1.0
        # Incremental online update of discounted mean reward
        alpha = 1.0 / st["discounted_n"]
        st["discounted_reward"] = (1.0 - alpha) * st["discounted_reward"] + alpha * reward

        self.history.append({
            "step": self.step_count,
            "strategy": strategy_name,
            "reward": round(reward, 4),
            "incumbent": round(self.incumbent_score, 4),
        })

        return reward

    def get_strategy_shares(self) -> dict[str, float]:
        """Returns relative historical allocation proportions of each strategy."""
        counts = {name: 0 for name in self.strategy_names}
        for item in self.history:
            counts[item["strategy"]] += 1
        total = max(1, len(self.history))
        return {k: round(v / total, 3) for k, v in counts.items()}
