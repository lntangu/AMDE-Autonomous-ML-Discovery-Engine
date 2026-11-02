"""Real, persisted baselines evaluated on the same protocol as candidates."""

from .registry import BASELINE_NAMES, make_baseline_genome

__all__ = ["BASELINE_NAMES", "make_baseline_genome"]
