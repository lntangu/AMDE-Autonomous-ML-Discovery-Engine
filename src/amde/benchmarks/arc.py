"""Read-only local ARC adapter.

This module never downloads data and never modifies an ARC repository. If a
user places ARC task JSON files under `data/arc/`, the adapter validates them,
splits tasks (not examples) deterministically, and can run simple exact-match
baselines. The current numeric/MLP candidate trainer cannot emit 2-D ARC
grids, so the profile is explicitly baseline-only and must not be reported as
a candidate-search benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ValidationError
from ..util import hash_json, read_json


@dataclass
class ArcTask:
    task_id: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    source_file: str
    family: str
    task_hash: str


@dataclass
class ArcProfile:
    status: str
    reason: str
    path: str
    n_files: int = 0
    n_tasks: int = 0
    dev_tasks: list[ArcTask] = field(default_factory=list)
    challenge_tasks: list[ArcTask] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] | None = None


def _validate_grid(grid: Any, where: str) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise ValidationError(f"{where}: grid must be a non-empty list", code="arc_grid")
    if len(grid) > 30:
        raise ValidationError(f"{where}: grid too tall", code="arc_grid")
    width = None
    for row in grid:
        if not isinstance(row, list) or not row:
            raise ValidationError(f"{where}: grid row must be non-empty", code="arc_grid")
        if width is None:
            width = len(row)
        if len(row) != width or len(row) > 30:
            raise ValidationError(f"{where}: grid rows must be rectangular and <=30", code="arc_grid")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ValidationError(f"{where}: cell must be an integer 0..9", code="arc_grid")
    return grid


def _validate_pairs(pairs: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(pairs, list) or not pairs:
        raise ValidationError(f"{where}: must be a non-empty list", code="arc_pairs")
    out = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
            raise ValidationError(f"{where}[{idx}]: pair needs input/output", code="arc_pairs")
        inp = _validate_grid(pair["input"], f"{where}[{idx}].input")
        out_grid = _validate_grid(pair["output"], f"{where}[{idx}].output")
        out.append({"input": inp, "output": out_grid})
    return out


def _family_for(task: dict[str, Any]) -> str:
    # A weak, transparent heuristic: size-change vs same-size, and colour count.
    same_shape = sum(
        1 for pair in task["train"] if len(pair["input"]) == len(pair["output"]) and len(pair["input"][0]) == len(pair["output"][0])
    )
    total = max(1, len(task["train"]))
    shape_mode = "same_size" if same_shape * 2 >= total else "resize"
    colors = sorted({cell for pair in task["train"] for grid in (pair["input"], pair["output"]) for row in grid for cell in row})
    return f"{shape_mode}_colors_{len(colors)}"


class ArcAdapter:
    def __init__(self, path: Path, *, max_files: int = 1000, max_bytes: int = 5 * 1024 * 1024) -> None:
        self.path = path
        self.max_files = max_files
        self.max_bytes = max_bytes

    def discover(self) -> ArcProfile:
        if not self.path.exists():
            return ArcProfile(
                status="BLOCKED",
                reason=f"ARC data path does not exist: {self.path}",
                path=str(self.path),
            )
        files = sorted(
            [
                candidate
                for candidate in self.path.rglob("*.json")
                if candidate.is_file() and ".git" not in candidate.parts
            ]
        )[: self.max_files]
        if not files:
            return ArcProfile(
                status="BLOCKED",
                reason=f"no ARC task JSON files found under {self.path}",
                path=str(self.path),
                n_files=0,
            )
        tasks: list[ArcTask] = []
        errors: list[str] = []
        for file_path in files:
            try:
                if file_path.stat().st_size > self.max_bytes:
                    raise ValidationError("file is larger than the ARC adapter limit", code="arc_size")
                raw = read_json(file_path)
                if not isinstance(raw, dict):
                    raise ValidationError("ARC task file must contain an object", code="arc_schema")
                train = _validate_pairs(raw.get("train"), "train")
                test = _validate_pairs(raw.get("test"), "test")
                task_hash = hash_json(raw)
                tasks.append(
                    ArcTask(
                        task_id=file_path.stem,
                        train=train,
                        test=test,
                        source_file=str(file_path.relative_to(self.path)),
                        family=_family_for({"train": train}),
                        task_hash=task_hash,
                    )
                )
            except Exception as exc:
                errors.append(f"{file_path.name}: {exc}")
        if not tasks:
            return ArcProfile(
                status="BLOCKED",
                reason="ARC files were found but none passed validation: " + "; ".join(errors[:5]),
                path=str(self.path),
                n_files=len(files),
            )
        # Split at task level, never by individual example, to reduce leakage.
        ordered = sorted(tasks, key=lambda task: hashlib.sha256(task.task_hash.encode()).hexdigest())
        split_at = max(1, int(math.floor(len(ordered) * 0.7)))
        dev = ordered[:split_at]
        challenge = ordered[split_at:]
        manifest = {
            "schema_version": "1.0.0",
            "status": "READY_BASELINE_ONLY",
            "mode": "baseline_only",
            "path": str(self.path),
            "n_files": len(files),
            "n_tasks": len(tasks),
            "n_dev_tasks": len(dev),
            "n_challenge_tasks": len(challenge),
            "split_policy": "deterministic 70/30 split by task hash; never by examples",
            "candidate_search_supported": False,
            "candidate_search_reason": "the current candidate trainer emits scalar predictions, not ARC grids",
            "validation_errors": errors,
            "task_ids_dev": [task.task_id for task in dev],
            "task_ids_challenge": [task.task_id for task in challenge],
            "dev_manifest_hash": hash_json([{"task_id": t.task_id, "hash": t.task_hash, "family": t.family} for t in dev]),
            "challenge_manifest_hash": hash_json([{"task_id": t.task_id, "hash": t.task_hash, "family": t.family} for t in challenge]),
        }
        manifest["manifest_hash"] = hash_json(manifest)
        return ArcProfile(
            status="READY_BASELINE_ONLY",
            reason="local ARC tasks found; baseline-only adapter is ready; candidate search unsupported for grids",
            path=str(self.path),
            n_files=len(files),
            n_tasks=len(tasks),
            dev_tasks=dev,
            challenge_tasks=challenge,
            manifest=manifest,
        )

    def run_baseline(self, *, max_tasks: int = 40) -> dict[str, Any]:
        """Run transparent exact-match baselines on dev tasks.

        These are real, if deliberately weak, baselines. No candidate training
        claim is made for ARC.
        """

        profile = self.discover()
        if profile.status != "READY_BASELINE_ONLY":
            return {"status": profile.status, "reason": profile.reason}
        task_results = []
        for task in profile.dev_tasks[:max_tasks]:
            identity = 0
            palette = 0
            for pair in task.test:
                inp = pair["input"]
                out = pair["output"]
                if inp == out:
                    identity += 1
                counts: dict[int, int] = {}
                for row in out:
                    for cell in row:
                        counts[cell] = counts.get(cell, 0) + 1
                most_common = max(counts, key=counts.get) if counts else 0
                guessed = [[most_common] * len(out[0]) for _ in range(len(out))]
                if guessed == out:
                    palette += 1
            n = max(1, len(task.test))
            task_results.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "n_test": len(task.test),
                    "identity_exact_match": identity / n,
                    "palette_fill_exact_match": palette / n,
                }
            )
        if not task_results:
            return {"status": "READY_BASELINE_ONLY", "reason": "no dev tasks"}
        summary = {
            "status": "READY_BASELINE_ONLY",
            "n_tasks": len(task_results),
            "identity_exact_match": sum(t["identity_exact_match"] for t in task_results) / len(task_results),
            "palette_fill_exact_match": sum(t["palette_fill_exact_match"] for t in task_results) / len(task_results),
            "tasks": task_results,
            "warning": "Exact-match baseline scores are not a claim of ARC solving ability.",
        }
        return summary


def arc_status(config_path: Path) -> dict[str, Any]:
    profile = ArcAdapter(config_path).discover()
    return {
        "status": profile.status,
        "reason": profile.reason,
        "path": profile.path,
        "n_files": profile.n_files,
        "n_tasks": profile.n_tasks,
        "manifest": profile.manifest,
    }
