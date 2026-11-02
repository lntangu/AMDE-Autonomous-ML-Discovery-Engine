"""AMDE 4.0 Strategy Portfolio Registry & Lifecycle Manager."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional

from amde.core.contracts import Candidate
from amde.space.search_space import SearchSpace
from amde.strategy.base import SearchStrategy
from amde.strategy.evolutionary import EvolutionaryStrategy
from amde.strategy.local_refinement import LocalRefinementStrategy
from amde.strategy.random_search import RandomSearchStrategy
from amde.strategy.tpe import TPEStrategy


class StrategyPortfolio:
    """Manages active search strategies with explicit lifecycle and isolation guarantees."""

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.strategies: dict[str, SearchStrategy] = {}

    def register(self, strategy: SearchStrategy) -> None:
        """Add a strategy to the active portfolio."""
        self.strategies[strategy.name] = strategy

    def unregister(self, name: str) -> Optional[SearchStrategy]:
        """Remove a strategy by name without disrupting remaining strategies."""
        return self.strategies.pop(name, None)

    def get(self, name: str) -> Optional[SearchStrategy]:
        return self.strategies.get(name)

    @property
    def strategy_names(self) -> list[str]:
        return list(self.strategies.keys())

    @classmethod
    def standard_portfolio(cls, search_space: SearchSpace) -> "StrategyPortfolio":
        """Constructs the standard qualified portfolio for AMDE 4.0."""
        portfolio = cls(search_space)
        portfolio.register(RandomSearchStrategy(search_space))
        portfolio.register(EvolutionaryStrategy(search_space))
        portfolio.register(LocalRefinementStrategy(search_space))
        portfolio.register(TPEStrategy(search_space))
        return portfolio
