"""AMDE 4.0 Multi-Objective Pareto Archive & Hypervolume Evaluation.

Maintains non-dominated frontier across:
1. Loss = -Accuracy (Minimize)
2. Inference Latency ms (Minimize)
3. Memory Consumption MB (Minimize)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from amde.core.contracts import Candidate


class ParetoArchive:
    """Bounded non-dominated archive for multi-objective discovery."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        # Each entry: {"candidate": Candidate, "objectives": (loss, latency, memory), "dict": dict}
        self.frontier: list[dict[str, Any]] = []

    def dominates(self, obj_a: Sequence[float], obj_b: Sequence[float]) -> bool:
        """Determines whether objective vector a Pareto-dominates vector b (all objectives minimized)."""
        at_least_as_good = all(a <= b for a, b in zip(obj_a, obj_b))
        strictly_better = any(a < b for a, b in zip(obj_a, obj_b))
        return at_least_as_good and strictly_better

    def update(
        self, candidate: Candidate, loss: float, latency: float, memory: float
    ) -> bool:
        """Updates the frontier with a new candidate. Returns True if accepted into the non-dominated set."""
        new_obj = (float(loss), float(latency), float(memory))

        # 1. Check if the candidate is dominated by any existing member
        for item in self.frontier:
            if self.dominates(item["objectives"], new_obj):
                return False  # Dominated by existing point

        # 2. Candidate is non-dominated; prune existing members that the new candidate dominates
        pruned_frontier = [
            item for item in self.frontier if not self.dominates(new_obj, item["objectives"])
        ]

        # 3. Add novel point
        pruned_frontier.append({
            "candidate": candidate,
            "candidate_hash": candidate.canonical_hash,
            "objectives": new_obj,
            "loss": float(loss),
            "latency": float(latency),
            "memory": float(memory),
        })

        # 4. Enforce capacity if needed
        if len(pruned_frontier) > self.capacity:
            # Sort by loss ascending
            pruned_frontier.sort(key=lambda x: x["loss"])
            pruned_frontier = pruned_frontier[:self.capacity]

        self.frontier = pruned_frontier
        return True

    def calculate_2d_hypervolume(self, ref_loss: float = 0.0, ref_latency: float = 50.0) -> float:
        """Calculates 2D hypervolume for loss and latency relative to explicit reference point.

        loss is in [-1.0, 0.0] (where -1.0 is 100% accuracy, 0.0 is 0% accuracy).
        ref_loss = 0.0, ref_latency = 50.0 ms.
        """
        if not self.frontier:
            return 0.0

        # Filter points within reference bounds
        valid_points = [
            (item["loss"], item["latency"])
            for item in self.frontier
            if item["loss"] <= ref_loss and item["latency"] <= ref_latency
        ]
        if not valid_points:
            return 0.0

        # Sort by loss ascending
        sorted_points = sorted(valid_points, key=lambda p: p[0])

        hv = 0.0
        current_latency = ref_latency
        for l, lat in sorted_points:
            if lat < current_latency:
                width = ref_loss - l
                height = current_latency - lat
                hv += width * height
                current_latency = lat

        return round(hv, 4)

    def __len__(self) -> int:
        return len(self.frontier)
