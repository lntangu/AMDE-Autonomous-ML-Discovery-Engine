"""Configuration validation and resource-plan honesty."""

from __future__ import annotations

import copy
import json
import unittest

from amde.config import ResourcePolicy, apply_cli_overrides, load_config
from amde.orchestrator import SMOKE_BASELINE_NAMES, smoke_config
from amde.errors import ConfigError
from amde.search.meta_policy import MetaPolicy
from tests.support import (
    PROJECT_ROOT,
    fake_hardware,
    make_config,
    make_project,
    small_config_data,
    write_project,
)


class ConfigTests(unittest.TestCase):
    def test_shared_test_fixture_forces_cpu_independent_of_host_gpu(self) -> None:
        # The integration gate verifies engine lifecycle, not optional CUDA availability.
        self.assertEqual(small_config_data()["runtime"]["device"], "cpu")

    def test_smoke_profile_is_bounded_without_changing_full_campaign_config(self) -> None:
        full = load_config(project_root=PROJECT_ROOT)
        smoke = smoke_config(full)
        quick_ids = set(full.quick_task_ids())
        smoke_dev_ids = {str(item["task_id"]) for item in smoke.dev_task_specs()}
        self.assertEqual(smoke_dev_ids, quick_ids)
        self.assertEqual(len(smoke.challenge_task_specs()), len(full.challenge_task_specs()))
        self.assertEqual(smoke.eval_seeds(int(smoke.get("evaluation.confirm.seed_count"))), [1, 2])
        self.assertEqual(int(smoke.get("search.proposals_per_generation")), 6)
        self.assertEqual(float(smoke.get("evaluation.quick.timeout_seconds")), 45.0)
        self.assertEqual(float(smoke.get("evaluation.confirm.timeout_seconds")), 60.0)
        self.assertEqual(int(smoke.get("runtime.max_transient_retries")), 0)
        self.assertEqual(len(SMOKE_BASELINE_NAMES), 4)
        self.assertNotIn("mlp_medium", SMOKE_BASELINE_NAMES)
        self.assertEqual(len(full.dev_task_specs()), 13)
        self.assertEqual(len(smoke.dev_task_specs()), len(quick_ids))

    def test_canonical_json_protocol_version_is_part_of_protocol_hash(self) -> None:
        from unittest.mock import patch
        import amde.config as config_module

        base = load_config(project_root=PROJECT_ROOT)
        with patch.object(config_module, "CANONICAL_JSON_VERSION", "test-canonical-json-v999"):
            changed = load_config(project_root=PROJECT_ROOT)
        self.assertNotEqual(base.protocol_hash, changed.protocol_hash)

    def test_default_config_loads_and_hashes_are_stable(self) -> None:
        first = load_config(project_root=PROJECT_ROOT)
        second = load_config(project_root=PROJECT_ROOT)
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(first.protocol_hash, second.protocol_hash)
        self.assertGreaterEqual(len(first.task_specs()), 9)

    def test_path_escape_is_rejected(self) -> None:
        data = json.loads((PROJECT_ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        data["paths"]["runs_dir"] = "../escape"
        with self.assertRaises(ConfigError):
            make_config(self, config_data=data)

    def test_duplicate_task_ids_are_rejected(self) -> None:
        data = json.loads((PROJECT_ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        data["benchmarks"]["task_specs"].append(copy.deepcopy(data["benchmarks"]["task_specs"][0]))
        with self.assertRaises(ConfigError):
            make_config(self, config_data=data)

    def test_unknown_generator_is_rejected(self) -> None:
        data = json.loads((PROJECT_ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        data["benchmarks"]["task_specs"][0]["generator"] = "python_eval"
        with self.assertRaises(ConfigError):
            make_config(self, config_data=data)

    def test_invalid_levels_and_seed_counts_are_rejected(self) -> None:
        base = json.loads((PROJECT_ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        bad_levels = copy.deepcopy(base)
        bad_levels["search"]["levels"] = ["A", "E"]
        with self.assertRaises(ConfigError):
            make_config(self, config_data=bad_levels)
        bad_seeds = copy.deepcopy(base)
        bad_seeds["evaluation"]["seeds"] = [1, 1, 2]
        with self.assertRaises(ConfigError):
            make_config(self, config_data=bad_seeds)

    def test_holdout_only_tasks_are_excluded_from_dev_and_quick(self) -> None:
        base = json.loads((PROJECT_ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        for spec in base["benchmarks"]["task_specs"]:
            spec.pop("holdout_only", None)
        base["benchmarks"]["task_specs"][-1]["holdout_only"] = True
        cfg = make_config(self, config_data=base)
        self.assertEqual(len(cfg.dev_task_specs()), len(cfg.task_specs()) - 1)
        self.assertEqual([spec["task_id"] for spec in cfg.challenge_task_specs()], [base["benchmarks"]["task_specs"][-1]["task_id"]])

        bad_quick = copy.deepcopy(base)
        bad_quick["benchmarks"]["quick_task_ids"].append(bad_quick["benchmarks"]["task_specs"][-1]["task_id"])
        with self.assertRaises(ConfigError):
            make_config(self, config_data=bad_quick)

    def test_meta_features_ignore_holdout_task_metadata(self) -> None:
        cfg = load_config(project_root=PROJECT_ROOT)
        policy = MetaPolicy(cfg)
        manifest = {
            "tasks": [
                {"task_id": "dev_a", "family": "linear", "kind": "regression", "n_features": 2},
                {"task_id": "heldout", "family": "secret_family", "kind": "classification", "n_features": 99, "holdout_only": True},
            ]
        }
        features = policy.features(manifest)
        self.assertEqual(features["n_tasks"], 1)
        self.assertEqual(features["n_families"], 1)
        self.assertEqual(features["max_n_features"], 2.0)
        self.assertEqual(features["classification_ratio"], 0.0)

    def test_default_protocol_reserves_two_task_families_for_challenge(self) -> None:
        cfg = load_config(project_root=PROJECT_ROOT)
        dev_ids = {str(spec["task_id"]) for spec in cfg.dev_task_specs()}
        challenge_ids = {str(spec["task_id"]) for spec in cfg.challenge_task_specs()}
        self.assertEqual(
            challenge_ids,
            {"tab_circles", "tab_imbalanced", "ood_spiral", "ood_checkerboard", "ood_sparse_parity"},
        )
        self.assertTrue(challenge_ids.isdisjoint(dev_ids))
        self.assertEqual(len(dev_ids), 13)
        self.assertEqual(len(challenge_ids), 5)
        self.assertEqual(set(cfg.quick_task_ids()) & challenge_ids, set())

    def test_resource_plan_never_claims_gpu_training(self) -> None:
        root = PROJECT_ROOT
        policy = ResourcePolicy.load(root)
        hardware = fake_hardware(
            logical_cores=32,
            ram_available_mb=65536,
            gpu_available=True,
            gpu_memory_free_mb=4096,
        )
        plan = policy.compute_plan(hardware, "auto")
        self.assertEqual(plan.device_effective, "cpu")
        self.assertFalse(plan.gpu_effective)
        self.assertEqual(plan.gpu_slots, 0)
        self.assertIn("no compatible CUDA backend", plan.reason)

        requested = policy.compute_plan(hardware, "cuda")
        self.assertEqual(requested.device_effective, "cpu")
        self.assertIn("tensor smoke test failed", requested.reason)
        self.assertFalse(requested.gpu_effective)

    def test_gpu_requires_successful_tensor_smoke_and_limits_concurrency(self) -> None:
        policy = ResourcePolicy.load(PROJECT_ROOT)
        hardware = fake_hardware(
            logical_cores=16,
            ram_available_mb=24576,
            ram_total_mb=32768,
            gpu_available=True,
            gpu_memory_free_mb=4096,
        )
        hardware.torch = {
            "import_ok": True,
            "cuda_available": True,
            "tensor_smoke": {"ok": True, "value": 512.0},
        }
        plan = policy.compute_plan(hardware, "auto")
        self.assertEqual(plan.device_effective, "cuda")
        self.assertTrue(plan.gpu_effective)
        self.assertEqual(plan.gpu_slots, 1)
        self.assertEqual(plan.cpu_workers, 1)

    def test_ram_reserve_never_drops_below_safety_floor(self) -> None:
        policy = ResourcePolicy.load(PROJECT_ROOT)
        policy.data["memory"]["reserve_mb"] = 64
        policy.data["memory"]["minimum_reserve_mb"] = 512
        self.assertEqual(policy.effective_ram_reserve_mb(32768), 512)

    def test_ram_reserve_scales_to_small_hosts(self) -> None:
        policy = ResourcePolicy.load(PROJECT_ROOT)
        reserve = policy.effective_ram_reserve_mb(6144)
        self.assertLessEqual(reserve, 1536)
        plan = policy.compute_plan(
            fake_hardware(ram_total_mb=6144, ram_available_mb=4500, logical_cores=12), "cpu"
        )
        self.assertLessEqual(plan.reserve_ram_mb, 1536)

    def test_resource_plan_respects_low_ram_and_cpu_reserve(self) -> None:
        policy = ResourcePolicy.load(PROJECT_ROOT)
        low_ram = policy.compute_plan(fake_hardware(logical_cores=8, ram_available_mb=1024), "cpu")
        self.assertEqual(low_ram.cpu_workers, 1)
        self.assertTrue(any("available RAM" in item for item in policy.warnings))

        small_cpu = ResourcePolicy.load(PROJECT_ROOT).compute_plan(
            fake_hardware(logical_cores=3, ram_available_mb=65536), "cpu"
        )
        self.assertGreaterEqual(small_cpu.cpu_workers, 1)
        self.assertLessEqual(small_cpu.cpu_workers, 1)

    def test_cli_seed_override_changes_config_hash(self) -> None:
        config = make_config(self)
        before = config.config_hash
        apply_cli_overrides(config, seed=999, device="cpu")
        self.assertNotEqual(config.config_hash, before)
        self.assertEqual(config.get("seed"), 999)
        self.assertEqual(config.get("runtime.device"), "cpu")

    def test_config_paths_are_relative_to_project_root(self) -> None:
        root, config = make_project(self)
        self.assertEqual(config.path("runs_dir"), (root / "runs").resolve())
        with self.assertRaises(ConfigError):
            config.path("missing_path")


if __name__ == "__main__":
    unittest.main()
