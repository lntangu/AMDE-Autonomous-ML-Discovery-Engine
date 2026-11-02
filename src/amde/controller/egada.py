"""AMDE 4.0 Evidence-Guided Autonomous Discovery Architecture (EGADA) Controller.

Implements non-stationary continuous-reward multi-armed bandit allocation using
Discounted Thompson Sampling with Gaussian posteriors, cost-normalized rewards,
and provably bounded finite states.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from amde.controller.base import MetaController
from amde.core.contracts import ControllerDecision, utc_now_iso


class EGADAController(MetaController):
    """Discounted Thompson Sampling meta-controller with calibrated reward-per-compute weighting."""

    def __init__(
        self,
        strategy_names: Sequence[str],
        discount_factor: float = 0.95,
        prior_mean: float = 0.5,
        prior_variance: float = 0.25,
        min_exploration_steps: Optional[int] = None,
    ):
        super().__init__(strategy_names)
        self.gamma = max(0.50, min(1.0, discount_factor))
        self.prior_var = max(0.01, prior_variance)
        self.min_exploration_steps = min_exploration_steps or len(self.strategy_names)

        # Bandit priors: sum_0 = prior_mean, n_0 = 1.0 -> prior mean = prior_mean, se = sqrt(prior_var)
        self.stats: dict[str, dict[str, float]] = {
            name: {
                "discounted_n": 1.0,
                "discounted_sum": prior_mean,
                "variance": self.prior_var,
            }
            for name in self.strategy_names
        }
        self.incumbent_score: Optional[float] = None

    def select_strategy(self, rng: random.Random, budget_fraction: float = 0.0) -> str:
        """Selects the next strategy using Discounted Thompson Sampling with phase priors."""
        self.step_count += 1

        # Phase 1: Pure exploration warm-up (first len(strategies) steps round-robin)
        if self.step_count <= self.min_exploration_steps and self.strategy_names:
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

        # Phase 3: Late-stage polishing bias (budget > 85%)
        if budget_fraction > 0.85 and "local_refinement" in self.strategy_names:
            if rng.random() < 0.50:
                self.history.append(
                    ControllerDecision(
                        step=self.step_count,
                        selected_strategy="local_refinement",
                        bandit_probabilities={"local_refinement": 0.50},
                        budget_fraction=budget_fraction,
                        exploration_mode=False,
                    )
                )
                return "local_refinement"

        # Thompson Sampling draws:
        samples: dict[str, float] = {}
        for name in self.strategy_names:
            st = self.stats[name]
            eff_n = max(0.10, st["discounted_n"])
            mean_est = st["discounted_sum"] / eff_n
            # Standard error: sqrt(var / n)
            se = math.sqrt(st["variance"] / eff_n)
            # Sample from posterior Normal distribution
            sample_val = rng.gauss(mean_est, max(0.05, se))
            samples[name] = sample_val

        best_strategy = max(samples.keys(), key=lambda k: samples[k])

        # Compute empirical softmax probabilities for logging
        max_s = max(samples.values())
        exp_vals = {k: math.exp(v - max_s) for k, v in samples.items()}
        sum_exp = sum(exp_vals.values())
        probs = {k: round(v / sum_exp, 4) for k, v in exp_vals.items()}

        self.history.append(
            ControllerDecision(
                step=self.step_count,
                selected_strategy=best_strategy,
                bandit_probabilities=probs,
                budget_fraction=budget_fraction,
                exploration_mode=False,
            )
        )
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
        """Calculates bounded reward and updates posterior statistics."""
        if strategy_name not in self.stats:
            return 0.0

        # 1. Calculate bounded reward
        if is_duplicate:
            reward = -1.0
        elif is_failure:
            reward = -0.5
        else:
            if self.incumbent_score is None:
                # First valid candidate establishes incumbent baseline
                improvement = 0.0
                self.incumbent_score = float(val_score)
            else:
                improvement = max(0.0, float(val_score) - self.incumbent_score)
                if float(val_score) > self.incumbent_score:
                    self.incumbent_score = float(val_score)

            # Cost-normalized improvement with soft baseline quality bonus
            time_norm = math.sqrt(max(0.01, train_duration))
            reward = (improvement / time_norm) + (0.05 * max(0.0, float(val_score)))

        # Clamp reward to [-1.0, 5.0] to mathematically eliminate infinity/NaN
        reward = max(-1.0, min(5.0, reward))

        # 2. Apply exponential discount to past observations across ALL strategies
        for name in self.strategy_names:
            self.stats[name]["discounted_n"] *= self.gamma
            self.stats[name]["discounted_sum"] *= self.gamma

        # 3. Incorporate new observation for the pulled strategy
        st = self.stats[strategy_name]
        st["discounted_n"] += 1.0
        st["discounted_sum"] += reward

        return reward
