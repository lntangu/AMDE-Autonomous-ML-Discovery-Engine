"""Benchmark splits, manifests, leakage audit, and ARC adapter state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from amde.benchmarks.arc import ArcAdapter, arc_status
from amde.benchmarks.registry import build_manifests, generate_tasks, tasks_for_stage
from amde.errors import ValidationError
from amde.config import load_config
from amde.util import sha256_text
from tests.support import PROJECT_ROOT, make_config


class BenchmarkTests(unittest.TestCase):
    def test_default_stage_routing_keeps_holdout_out_of_open_stages(self) -> None:
        config = load_config(project_root=PROJECT_ROOT)
        # Explicitly passing all specs must not bypass stage-level holdout routing.
        confirm_ids = {task.task_id for task in tasks_for_stage(config, "confirm", task_specs=config.task_specs())}
        challenge_ids = {task.task_id for task in tasks_for_stage(config, "challenge", task_specs=config.task_specs())}
        self.assertEqual(
            challenge_ids,
            {"tab_circles", "tab_imbalanced", "ood_spiral", "ood_checkerboard", "ood_sparse_parity"},
        )
        self.assertTrue(challenge_ids.isdisjoint(confirm_ids))
        self.assertEqual(len(confirm_ids), 13)

    def test_default_challenge_cohort_hash_covers_only_heldout_task_splits(self) -> None:
        config = load_config(project_root=PROJECT_ROOT)
        manifests = build_manifests(config)
        task_hashes = manifests["split_manifest"]["tasks"]
        expected_hashes = sorted(
            task_hashes[task_id]["splits"]["challenge"]
            for task_id in ("tab_circles", "tab_imbalanced", "ood_spiral", "ood_checkerboard", "ood_sparse_parity")
        )
        expected_cohort = sha256_text("|".join(expected_hashes))[:24]
        self.assertEqual(manifests["cohort"], expected_cohort)

    def test_research_task_zoo_contains_diverse_dev_and_heldout_families(self) -> None:
        config = load_config(project_root=PROJECT_ROOT)
        specs = config.task_specs()
        families = {str(spec["family"]) for spec in specs}
        generators = {str(spec["generator"]) for spec in specs}
        self.assertEqual(len(specs), 18)
        self.assertGreaterEqual(len(families), 17)
        self.assertTrue({"friedman_regression", "sparse_linear_regression", "moons_classification",
                         "parity_classification", "spiral_classification", "checkerboard_classification"} <= generators)
        quick = set(config.quick_task_ids())
        self.assertEqual(len(quick), 4)
        self.assertFalse(quick & {str(spec["task_id"]) for spec in specs if spec.get("holdout_only")})

    def test_new_generator_outputs_are_finite_and_dimensionally_valid(self) -> None:
        config = load_config(project_root=PROJECT_ROOT)
        tasks = generate_tasks(config)
        checked = {task.task_id: task for task in tasks}
        for task_id in ("tab_sine", "tab_friedman", "tab_sparse_linear", "tab_heteroscedastic",
                        "tab_moons", "tab_parity", "ood_spiral", "ood_checkerboard", "ood_sparse_parity"):
            task = checked[task_id]
            for split in ("train", "val", "dev"):
                for row in task.split(split):
                    self.assertEqual(len(row.x), task.n_features)
                    self.assertTrue(all(__import__("math").isfinite(value) for value in row.x))
                    self.assertTrue(__import__("math").isfinite(row.y))

    def test_challenge_examples_are_not_generated_by_default(self) -> None:
        config = make_config(self)
        tasks = generate_tasks(config)
        self.assertTrue(tasks)
        for task in tasks:
            self.assertIsNone(task.challenge)
            with self.assertRaises(ValueError):
                task.split("challenge")

    def test_splits_are_disjoint_and_deterministic(self) -> None:
        config = make_config(self)
        first = generate_tasks(config)
        second = generate_tasks(config)
        for left, right in zip(first, second):
            self.assertEqual(left.split_hashes(include_challenge=False), right.split_hashes(include_challenge=False))
            seen: set[tuple[float, ...]] = set()
            for split_name in ("train", "val", "dev"):
                for example in left.split(split_name):
                    signature = tuple(round(value, 8) for value in example.x)
                    self.assertNotIn(signature, seen)
                    seen.add(signature)

    def test_sealed_challenge_split_is_disjoint_and_generated_on_request(self) -> None:
        config = make_config(self, challenge=True)
        tasks = generate_tasks(config, include_challenge=True)
        for task in tasks:
            self.assertIsNotNone(task.challenge)
            seen: set[tuple[float, ...]] = set()
            for split_name in ("train", "val", "dev", "challenge"):
                for example in task.split(split_name):
                    signature = (split_name, tuple(round(value, 8) for value in example.x))
                    features = signature[1]
                    self.assertNotIn(features, seen)
                    seen.add(features)

    def test_manifest_contains_hashes_not_challenge_examples(self) -> None:
        config = make_config(self, challenge=True)
        manifests = build_manifests(config)
        self.assertIn("tasks", manifests)
        self.assertTrue(manifests["cohort"])
        data_manifest = manifests["data_manifest"]
        for task in data_manifest["tasks"]:
            for split in ("train", "val", "dev", "challenge"):
                self.assertEqual(set(task["splits"][split]), {"n", "sha256"})
        encoded = json.dumps(data_manifest)
        self.assertNotIn('"x"', encoded)
        self.assertNotIn('"y"', encoded)
        self.assertNotIn("examples", encoded)
        self.assertNotIn("input", encoded)
        self.assertNotIn("output", encoded)
        self.assertEqual(data_manifest["audit"]["issues"], [])

    def test_manifest_hashes_are_stable_for_same_protocol(self) -> None:
        config = make_config(self)
        first = build_manifests(config)
        second = build_manifests(config)
        self.assertEqual(first["data_manifest"]["manifest_hash"], second["data_manifest"]["manifest_hash"])
        self.assertEqual(first["cohort"], second["cohort"])

    def test_arc_is_blocked_without_local_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = ArcAdapter(Path(tmp) / "arc").discover()
            self.assertEqual(profile.status, "BLOCKED")
            status = arc_status(Path(tmp) / "arc")
            self.assertEqual(status["status"], "BLOCKED")
            self.assertEqual(status["n_tasks"], 0)

    def test_arc_adapter_validates_and_never_mutates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arc = Path(tmp) / "arc"
            arc.mkdir()
            task = {
                "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
                "test": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
            }
            path = arc / "task_a.json"
            original = json.dumps(task)
            path.write_text(original, encoding="utf-8")
            profile = ArcAdapter(arc).discover()
            self.assertEqual(profile.status, "READY_BASELINE_ONLY")
            self.assertEqual(profile.n_tasks, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            baseline = ArcAdapter(arc).run_baseline()
            self.assertEqual(baseline["status"], "READY_BASELINE_ONLY")
            self.assertIn("identity_exact_match", baseline)
            self.assertIn("warning", baseline)

    def test_arc_adapter_rejects_invalid_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arc = Path(tmp) / "arc"
            arc.mkdir()
            (arc / "bad.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0, True]], "output": [[0, 1]]}],
                        "test": [{"input": [[0, 1]], "output": [[0, 1]]}],
                    }
                ),
                encoding="utf-8",
            )
            profile = ArcAdapter(arc).discover()
            self.assertEqual(profile.status, "BLOCKED")
            self.assertIn("none passed validation", profile.reason)

    def test_split_audit_detects_challenge_overlap(self) -> None:
        config = make_config(self, challenge=True)
        tasks = generate_tasks(config, include_challenge=True)
        tasks[0].challenge[0] = tasks[0].dev[0]
        from amde.benchmarks.registry import _split_audit

        audit = _split_audit(tasks, 8)
        self.assertTrue(audit["issues"])
        self.assertTrue(any("dev/challenge" in issue for issue in audit["issues"]))

    def test_split_audit_detects_leakage(self) -> None:
        config = make_config(self)
        tasks = generate_tasks(config)
        # Manufacture leakage in a copy; the audit is a real check, not a label.
        tasks[0].val[0] = tasks[0].train[0]
        from amde.benchmarks.registry import _split_audit

        audit = _split_audit(tasks, 8)
        self.assertTrue(audit["issues"])
        self.assertIn("near-duplicate", audit["issues"][0])


if __name__ == "__main__":
    unittest.main()
