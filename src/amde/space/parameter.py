"""AMDE 4.0 Typed Hyperparameter Domain & Constraint Abstractions."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

from amde.core.errors import ValidationError


@dataclass(frozen=True)
class Parameter(ABC):
    """Abstract base class for typed search space parameters."""
    name: str

    @abstractmethod
    def sample(self, rng: random.Random) -> Any:
        """Sample a valid random value from the parameter domain."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if a value conforms to domain boundaries and typing."""
        pass

    @abstractmethod
    def perturb(self, value: Any, rng: random.Random, scale: float = 0.20) -> Any:
        """Perturb an existing value smoothly within domain boundaries."""
        pass


@dataclass(frozen=True)
class ContinuousParameter(Parameter):
    """Continuous real-valued parameter with optional logarithmic scaling."""
    lower: float
    upper: float
    log_scale: bool = False
    default: Optional[float] = None

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValidationError(f"Invalid bounds for {self.name}: lower ({self.lower}) >= upper ({self.upper})")
        if self.log_scale and self.lower <= 0.0:
            raise ValidationError(f"Log scale parameter {self.name} must have strictly positive lower bound, got {self.lower}")

    def sample(self, rng: random.Random) -> float:
        if self.log_scale:
            log_low = math.log10(self.lower)
            log_high = math.log10(self.upper)
            val = 10 ** rng.uniform(log_low, log_high)
        else:
            val = rng.uniform(self.lower, self.upper)
        return round(val, 7)

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        return self.lower <= float(value) <= self.upper

    def perturb(self, value: Any, rng: random.Random, scale: float = 0.20) -> float:
        val = float(value)
        if self.log_scale:
            current_log = math.log10(max(self.lower, min(self.upper, val)))
            span = math.log10(self.upper) - math.log10(self.lower)
            delta = rng.gauss(0.0, span * scale)
            new_val = 10 ** (current_log + delta)
        else:
            span = self.upper - self.lower
            delta = rng.gauss(0.0, span * scale)
            new_val = val + delta
        clamped = max(self.lower, min(self.upper, new_val))
        return round(clamped, 7)


@dataclass(frozen=True)
class IntegerParameter(Parameter):
    """Discrete integer-valued parameter."""
    lower: int
    upper: int
    default: Optional[int] = None

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValidationError(f"Invalid integer bounds for {self.name}: lower ({self.lower}) > upper ({self.upper})")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.lower, self.upper)

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, np.integer)):
            return False
        return self.lower <= int(value) <= self.upper

    def perturb(self, value: Any, rng: random.Random, scale: float = 0.20) -> int:
        val = int(value)
        span = max(1, self.upper - self.lower)
        step = int(round(rng.gauss(0.0, max(1.0, span * scale))))
        if step == 0:
            step = rng.choice([-1, 1])
        new_val = val + step
        return max(self.lower, min(self.upper, new_val))


@dataclass(frozen=True)
class CategoricalParameter(Parameter):
    """Categorical parameter choosing from a discrete set of choices."""
    choices: Tuple[Any, ...]
    default: Optional[Any] = None

    def __post_init__(self):
        if not self.choices:
            raise ValidationError(f"Categorical parameter {self.name} must have non-empty choices")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def validate(self, value: Any) -> bool:
        return value in self.choices

    def perturb(self, value: Any, rng: random.Random, scale: float = 0.20) -> Any:
        # With probability (1 - scale), retain; otherwise choose a different category
        if rng.random() > scale or len(self.choices) == 1:
            return value
        alternatives = [c for c in self.choices if c != value]
        return rng.choice(alternatives) if alternatives else value


@dataclass(frozen=True)
class BooleanParameter(Parameter):
    """Boolean flag parameter."""
    default: bool = True

    def sample(self, rng: random.Random) -> bool:
        return rng.choice([True, False])

    def validate(self, value: Any) -> bool:
        return isinstance(value, bool)

    def perturb(self, value: Any, rng: random.Random, scale: float = 0.20) -> bool:
        if rng.random() < scale:
            return not bool(value)
        return bool(value)
