"""Benchmark registry, split audit, and manifest construction."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..config import Config, PROTOCOL_GENERATOR_VERSION
from ..errors import ValidationError
from ..util import canonical_json, hash_json, sha256_text
from . import BENCHMARK_VERSION
from .base import SplitData, generate_task
from .generators import CHALLENGE_GENERATORS, GENERATORS


def generate_tasks(
    config: Config,
    *,
    task_specs: Iterable[dict[str, Any]] | None = None,
    include_challenge: bool = False,
) -> list[SplitData]:
    specs = list(task_specs) if task_specs is not None else config.task_specs()
    split_config = dict(config.get("benchmarks.split"))
    tasks: list[SplitData] = []
    for spec in specs:
        tasks.append(
            generate_task(
                spec,
                split_config=split_config,
                generators=GENERATORS,
                challenge_generators=CHALLENGE_GENERATORS,
                include_challenge=include_challenge,
            )
        )
    return tasks


def tasks_for_stage(
    config: Config,
    stage: str,
    *,
    task_specs: Iterable[dict[str, Any]] | None = None,
) -> list[SplitData]:
    if stage not in {"quick", "confirm", "challenge"}:
        raise ValueError(f"unknown stage: {stage}")
    specs = list(task_specs) if task_specs is not None else config.task_specs()
    if stage == "challenge":
        allowed_ids = {str(spec["task_id"]) for spec in config.challenge_task_specs()}
        specs = [spec for spec in specs if str(spec["task_id"]) in allowed_ids]
    else:
        # Explicit task_specs must not bypass the holdout boundary.
        specs = [spec for spec in specs if not bool(spec.get("holdout_only", False))]
    if stage == "quick":
        wanted = set(config.quick_task_ids())
        specs = [spec for spec in specs if spec["task_id"] in wanted]
    if not specs:
        raise ValueError(f"no tasks selected for stage {stage}")
    return generate_tasks(config, task_specs=specs, include_challenge=(stage == "challenge"))


def _split_audit(tasks: list[SplitData], rounding: int) -> dict[str, Any]:
    issues: list[str] = []
    for task in tasks:
        seen_signatures: dict[tuple[float, ...], str] = {}
        split_names = ["train", "val", "dev"]
        if task.challenge is not None:
            split_names.append("challenge")
        for split_name in split_names:
            for example in task.split(split_name):
                signature = tuple(round(v, rounding) for v in example.x)
                previous = seen_signatures.get(signature)
                if previous is not None and previous != split_name:
                    issues.append(f"{task.task_id}: near-duplicate feature vector across {previous}/{split_name}")
                seen_signatures[signature] = split_name
        if any(not task.split(split_name) for split_name in split_names):
            issues.append(f"{task.task_id}: empty split")
        if task.kind == "classification":
            for split_name in split_names:
                labels = {int(round(ex.y)) for ex in task.split(split_name)}
                if labels != {0, 1}:
                    issues.append(f"{task.task_id}/{split_name}: missing a class")
    task_hashes: dict[str, str] = {}
    for task in tasks:
        task_hash = hash_json({name: [ex.to_dict() for ex in task.split(name)] for name in ("train", "val", "dev")})
        if task_hash in task_hashes:
            issues.append(f"duplicate task content: {task.task_id} and {task_hashes[task_hash]}")
        task_hashes[task_hash] = task.task_id
    return {"issues": issues, "n_tasks": len(tasks), "rounding": rounding}


def build_manifests(config: Config) -> dict[str, Any]:
    """Build audit manifests without exposing challenge examples.

    Challenge data are generated in this trusted setup function solely to
    compute hashes. They are not written to disk, logged, or returned in any
    search-facing result. Normal evaluator requests never include this output.
    """

    rounding = int(config.get("benchmarks.split.duplicate_rounding", 8))
    specs_by_id = {str(spec["task_id"]): spec for spec in config.task_specs()}
    challenge_ids = {str(spec["task_id"]) for spec in config.challenge_task_specs()}
    tasks = generate_tasks(config, include_challenge=True)
    audit = _split_audit(tasks, rounding)
    if audit["issues"]:
        raise ValidationError(
            "split audit failed: " + "; ".join(audit["issues"][:8]),
            code="split_overlap",
        )
    task_records = []
    challenge_hashes = []
    for task in tasks:
        hashes = task.split_hashes(include_challenge=True)
        if task.task_id in challenge_ids:
            challenge_hashes.append(hashes.get("challenge", ""))
        task_records.append(
            {
                "task_id": task.task_id,
                "suite": task.suite,
                "family": task.family,
                "kind": task.kind,
                "n_features": task.n_features,
                "holdout_only": bool(specs_by_id.get(task.task_id, {}).get("holdout_only", False)),
                "meta": task.meta,
                "splits": {
                    name: {"n": task.split_counts()[name], "sha256": hashes[name]}
                    for name in ("train", "val", "dev", "challenge")
                },
            }
        )
    cohort = sha256_text("|".join(sorted(challenge_hashes)))[:24]
    data_manifest = {
        "schema_version": "1.0.0",
        "dataset_version": config.get("benchmarks.dataset_version"),
        "benchmark_version": BENCHMARK_VERSION,
        "generator_protocol": PROTOCOL_GENERATOR_VERSION,
        "provenance": "deterministically generated locally; no downloaded data",
        "task_count": len(task_records),
        "tasks": task_records,
        "split_policy": {
            "kind": "independent deterministic example splits",
            "rounding": rounding,
            "sizes": config.get("benchmarks.split"),
            "challenge_cohort": cohort,
        },
        "audit": audit,
    }
    data_manifest["manifest_hash"] = hash_json(data_manifest)
    split_manifest = {
        "schema_version": "1.0.0",
        "protocol_hash": config.protocol_hash,
        "dataset_version": config.get("benchmarks.dataset_version"),
        "challenge_cohort": cohort,
        "challenge_policy": {
            "sealed": True,
            "search_uses_challenge": False,
            "openings_per_run": config.get("challenge.max_openings_per_run"),
        },
        "tasks": {
            task.task_id: {
                "family": task.family,
                "suite": task.suite,
                "kind": task.kind,
                "splits": task.split_hashes(include_challenge=True),
            }
            for task in tasks
        },
    }
    split_manifest["manifest_hash"] = hash_json(split_manifest)
    
    # Clear challenge examples from tasks before returning to search process
    # to avoid materializing challenge data in the search/orchestrator process.
    for task in tasks:
        task.challenge = None

    return {
        "data_manifest": data_manifest,
        "split_manifest": split_manifest,
        "tasks": tasks,
        "cohort": cohort,
    }


def manifest_summary(manifests: dict[str, Any]) -> dict[str, Any]:
    data = manifests["data_manifest"]
    return {
        "data_manifest_hash": data["manifest_hash"],
        "split_manifest_hash": manifests["split_manifest"]["manifest_hash"],
        "challenge_cohort": manifests["cohort"],
        "task_count": data["task_count"],
        "dataset_version": data["dataset_version"],
        "benchmark_version": data["benchmark_version"],
    }
