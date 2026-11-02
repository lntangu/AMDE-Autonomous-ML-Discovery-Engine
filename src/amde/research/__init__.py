"""Preregistered hypothesis and controlled-ablation research tools."""

from .ablation import build_ablation_plan, audit_ablation_results
from .hypotheses import HypothesisRegistry, evaluate_hypothesis, validate_hypothesis

__all__ = [
    "HypothesisRegistry", "validate_hypothesis", "evaluate_hypothesis",
    "build_ablation_plan", "audit_ablation_results",
]
