"""Deterministic UCB1 search-strategy portfolio."""

from __future__ import annotations

import math
import random
from typing import Any


class Portfolio:
    def __init__(self, strategies: list[str], *, exploration_floor: float = 0.15, c: float = 0.7) -> None:
        if not strategies:
            raise ValueError("portfolio needs at least one strategy")
        self.strategies = list(strategies)
        self.exploration_floor = max(0.0, min(1.0, float(exploration_floor)))
        self.c = float(c)
        self.pulls = {name: 0 for name in self.strategies}
        self.reward_sum = {name: 0.0 for name in self.strategies}
        self.total_pulls = 0

    def select(self, rng: random.Random) -> str:
        if rng.random() < self.exploration_floor:
            return rng.choice(self.strategies)
        untried = [name for name in self.strategies if self.pulls[name] == 0]
        if untried:
            return untried[0]
        log_total = math.log(max(1, self.total_pulls))
        scored = []
        for name in self.strategies:
            mean_reward = self.reward_sum[name] / self.pulls[name]
            bonus = self.c * math.sqrt(log_total / self.pulls[name])
            scored.append((mean_reward + bonus, name))
        return max(scored)[1]

    def update(self, strategy: str, reward: float) -> None:
        if strategy not in self.pulls:
            return
        self.pulls[strategy] += 1
        self.reward_sum[strategy] += max(0.0, min(1.0, float(reward)))
        self.total_pulls += 1

    def best(self) -> str:
        if self.total_pulls == 0:
            return self.strategies[0]
        return max(
            self.strategies,
            key=lambda name: (self.reward_sum[name] / max(1, self.pulls[name]), -self.pulls[name]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": list(self.strategies),
            "exploration_floor": self.exploration_floor,
            "c": self.c,
            "pulls": dict(self.pulls),
            "reward_sum": dict(self.reward_sum),
            "total_pulls": self.total_pulls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        portfolio = cls(
            [str(item) for item in data["strategies"]],
            exploration_floor=float(data.get("exploration_floor", 0.15)),
            c=float(data.get("c", 0.7)),
        )
        portfolio.pulls = {name: int(data.get("pulls", {}).get(name, 0)) for name in portfolio.strategies}
        portfolio.reward_sum = {
            name: float(data.get("reward_sum", {}).get(name, 0.0)) for name in portfolio.strategies
        }
        portfolio.total_pulls = int(data.get("total_pulls", sum(portfolio.pulls.values())))
        return portfolio
