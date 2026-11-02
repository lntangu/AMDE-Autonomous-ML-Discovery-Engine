"""Open-ended, progressively growing proposal grammar for AMDE.

The expression language contains all finite ASTs over a finite typed grammar.
That set is countably infinite in principle because expression trees may have
arbitrary finite size. Any concrete campaign still applies explicit node,
depth, byte, time, memory and parameter budgets. This module never removes
those safety bounds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..discovery import dsl
from ..discovery.factory import _validate_kwargs, random_genome
from ..discovery.genome import Genome


@dataclass(frozen=True)
class GrowthBudget:
    generation: int
    max_nodes: int
    max_depth: int


def growth_budget(generation: int, safety: dict[str, Any]) -> GrowthBudget:
    """Increase representational complexity over generations, within caps."""
    generation = max(0, int(generation))
    node_cap = max(8, int(safety.get("max_dsl_nodes", 128)))
    depth_cap = max(3, int(safety.get("max_dsl_depth", 16)))
    # Geometric early growth explores increasingly expressive programs;
    # after reaching the configured cap, diversity comes from recombination
    # and continuous constants rather than exceeding resource policy.
    growth_step = min(8, generation // 3)
    nodes = min(node_cap, 16 * (2 ** growth_step))
    depth = min(depth_cap, 3 + generation // 2)
    return GrowthBudget(generation=generation, max_nodes=nodes, max_depth=depth)


def _make_expression(
    rng: random.Random,
    *,
    budget: GrowthBudget,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    # Generate multiple smaller branches rather than always filling the full
    # budget; this avoids systematically creating pathological, slow updates.
    target_depth = rng.randint(2, max(2, budget.max_depth))
    target_nodes = rng.randint(min(5, budget.max_nodes), budget.max_nodes)
    raw = dsl.random_expr(
        rng,
        depth=target_depth,
        max_nodes=target_nodes,
        variables=variables,
    )
    return dsl.validate_dsl(
        raw,
        max_nodes=budget.max_nodes,
        max_depth=budget.max_depth,
        version="dsl-v2",
    )


def generate_open_ended_genome(
    rng: random.Random,
    *,
    safety: dict[str, Any],
    generation: int,
) -> tuple[Genome, GrowthBudget]:
    """Generate a valid DSL genome at the current grammar growth frontier."""
    budget = growth_budget(generation, safety)
    seed = random_genome("D", rng, safety=safety)
    data = seed.to_dict()
    data["dsl"]["version"] = "dsl-v2"
    data["optimizer"]["kind"] = "dsl"

    roles = {
        "w_update": ("g", "p", "w", "m", "v", "t", "lr", "gn"),
        "b_update": ("g", "p", "b", "m", "v", "t", "lr", "gn"),
        "m_update": ("g", "p", "m", "v", "t", "lr", "gn"),
        "v_update": ("g", "p", "m", "v", "t", "lr", "gn"),
    }

    # Preserve some known-stable subexpressions as a grammar prior, while
    # ensuring the proposal genuinely grows beyond a fixed template.
    for key, variables in roles.items():
        if key in {"w_update", "b_update"} or rng.random() < 0.55:
            data["dsl"][key] = _make_expression(
                rng, budget=budget, variables=variables
            )
        else:
            data["dsl"][key] = None

    # The base random D genome can contain an optimizer-specific expression
    # created under a narrower limit; all expressions are revalidated against
    # the current campaign safety policy before leaving this function.
    kwargs = _validate_kwargs(safety)
    genome = Genome(data, **kwargs)
    return genome, budget


class NoveltyLedger:
    """Small exact-identity archive for proposed genome serializations.

    SHA-256 is used as an index, while the canonical JSON payload is retained
    to detect even a theoretical digest collision. Behavioural signatures are
    stored as diagnostics only: finite probes cannot prove two programs
    equivalent, so they are never used to discard candidates automatically.
    """

    def __init__(self) -> None:
        self._payload_by_digest: dict[str, str] = {}
        self._behaviour_by_digest: dict[str, str] = {}

    def add(self, genome: Genome, behaviour_signature: str | None = None) -> bool:
        digest = genome.hash
        payload = genome.canonical_json()
        existing = self._payload_by_digest.get(digest)
        if existing is not None:
            if existing != payload:
                raise RuntimeError("SHA-256 collision detected for distinct canonical genomes")
            return False
        self._payload_by_digest[digest] = payload
        if behaviour_signature:
            self._behaviour_by_digest[digest] = str(behaviour_signature)
        return True

    def probable_behavioural_collisions(self, behaviour_signature: str) -> list[str]:
        signature = str(behaviour_signature)
        return sorted(
            digest for digest, prior in self._behaviour_by_digest.items()
            if prior == signature
        )

    def __len__(self) -> int:
        return len(self._payload_by_digest)


def behaviour_signature(
    genome: Genome,
    *,
    probes: int = 64,
    seed: int = 0xA4D3,
) -> str:
    """Hash deterministic numeric probe responses for near-duplicate triage.

    This is intentionally a diagnostic fingerprint, not a proof of semantic
    equivalence. Any match should be reviewed or tested on a richer probe set.
    """
    import hashlib
    import json

    rng = random.Random(seed)
    updates = genome.data["dsl"]
    outputs: list[Any] = []
    for _ in range(max(1, int(probes))):
        variables = {name: rng.uniform(-2.0, 2.0) for name in dsl.ALLOWED_VARS}
        row = {}
        for key in ("w_update", "b_update", "m_update", "v_update"):
            expression = updates.get(key)
            if expression is None:
                row[key] = None
                continue
            try:
                value = dsl.evaluate(expression, variables)
                row[key] = round(value, 10) if math.isfinite(value) else "nonfinite"
            except Exception as exc:
                row[key] = f"error:{type(exc).__name__}"
        outputs.append(row)
    encoded = json.dumps(outputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
