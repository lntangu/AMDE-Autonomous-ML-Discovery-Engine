"""Shared benchmark types and deterministic split generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..util import hash_json, seed_from


@dataclass(frozen=True)
class Example:
    x: tuple[float, ...]
    y: float

    def to_dict(self) -> dict[str, Any]:
        return {"x": list(self.x), "y": self.y}


@dataclass
class SplitData:
    task_id: str
    suite: str
    family: str
    kind: str
    n_features: int
    train: list[Example]
    val: list[Example]
    dev: list[Example]
    challenge: list[Example] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    version: str = "generated-v1"

    def split(self, name: str) -> list[Example]:
        if name == "challenge":
            if self.challenge is None:
                raise ValueError("challenge split was not generated")
            return self.challenge
        value = getattr(self, name, None)
        if not isinstance(value, list):
            raise ValueError(f"unknown split: {name}")
        return value

    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "dev": len(self.dev),
            "challenge": len(self.challenge) if self.challenge is not None else 0,
        }

    def split_hashes(self, include_challenge: bool = True) -> dict[str, str]:
        result = {}
        for name in ("train", "val", "dev"):
            result[name] = hash_json([example.to_dict() for example in self.split(name)])
        if include_challenge and self.challenge is not None:
            result["challenge"] = hash_json([example.to_dict() for example in self.challenge])
        return result


ExampleFactory = Callable[[random.Random, dict[str, Any], int], Example]


def _feature_signature(example: Example, rounding: int) -> tuple[Any, ...]:
    return tuple(round(value, rounding) for value in example.x)


def _make_split(
    spec: dict[str, Any],
    split_name: str,
    count: int,
    factory: ExampleFactory,
    rng_seed: int,
    rounding: int,
    forbidden: set[tuple[Any, ...]],
) -> list[Example]:
    rng = random.Random(seed_from(rng_seed, spec["task_id"], split_name, "examples"))
    result: list[Example] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = count * 80 + 500
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        example = factory(rng, spec, count)
        sig = _feature_signature(example, rounding)
        if sig in forbidden or sig in seen:
            continue
        # Reject non-finite values rather than silently generating data that
        # makes every candidate look invalid.
        if not all(math.isfinite(value) for value in example.x) or not math.isfinite(example.y):
            continue
        seen.add(sig)
        result.append(example)
    if len(result) != count:
        raise RuntimeError(
            f"could not generate a disjoint split for {spec['task_id']}/{split_name} "
            f"({len(result)}/{count})"
        )
    return result


def generate_task(
    spec: dict[str, Any],
    *,
    split_config: dict[str, Any],
    generators: dict[str, ExampleFactory],
    challenge_generators: dict[str, ExampleFactory] | None = None,
    include_challenge: bool = False,
) -> SplitData:
    """Generate deterministic splits for one task.

    The default path deliberately does *not* generate challenge examples. The
    sealed evaluator is the only component that requests them.
    """

    try:
        factory = generators[spec["generator"]]
    except KeyError as exc:
        raise ValueError(f"unknown generator: {spec['generator']}") from exc
    challenge_factory = (challenge_generators or {}).get(spec["generator"], factory)
    rounding = int(split_config.get("duplicate_rounding", 8))
    base_seed = int(spec["seed"])
    train = _make_split(spec, "train", int(split_config["train"]), factory, base_seed, rounding, set())
    forbidden = {_feature_signature(ex, rounding) for ex in train}
    val = _make_split(spec, "val", int(split_config["val"]), factory, base_seed, rounding, forbidden)
    forbidden.update(_feature_signature(ex, rounding) for ex in val)
    dev = _make_split(spec, "dev", int(split_config["dev"]), factory, base_seed, rounding, forbidden)
    challenge: list[Example] | None = None
    if include_challenge:
        forbidden.update(_feature_signature(ex, rounding) for ex in dev)
        challenge = _make_split(
            spec,
            "challenge",
            int(split_config["challenge"]),
            challenge_factory,
            base_seed ^ 0x5EEDC0DE,
            rounding,
            forbidden,
        )
    meta = _task_meta(spec, train, val, dev)
    return SplitData(
        task_id=str(spec["task_id"]),
        suite=str(spec["suite"]),
        family=str(spec["family"]),
        kind=str(spec["kind"]),
        n_features=int(spec["n_features"]),
        train=train,
        val=val,
        dev=dev,
        challenge=challenge,
        meta=meta,
    )


def _task_meta(spec: dict[str, Any], train: list[Example], val: list[Example], dev: list[Example]) -> dict[str, Any]:
    labels = [example.y for example in train]
    if spec["kind"] == "classification":
        positive = sum(1 for y in labels if y >= 0.5)
        balance = positive / max(1, len(labels))
        target = {"positive_rate": balance, "n_classes": 2}
    else:
        mu = sum(labels) / len(labels)
        var = sum((y - mu) ** 2 for y in labels) / len(labels)
        target = {"mean": mu, "variance": var}
    return {
        "suite": spec["suite"],
        "family": spec["family"],
        "kind": spec["kind"],
        "n_features": int(spec["n_features"]),
        "n_train": len(train),
        "n_val": len(val),
        "n_dev": len(dev),
        "target": target,
    }


def task_specs_for_quick(specs: Iterable[dict[str, Any]], quick_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(quick_ids)
    selected = [dict(spec) for spec in specs if spec["task_id"] in wanted]
    if not selected:
        raise ValueError("quick task selection is empty")
    return selected
