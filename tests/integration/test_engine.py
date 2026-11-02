"""Bounded end-to-end engine behavior in temporary projects."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from amde.baselines.registry import BASELINE_NAMES
from amde.discovery.archive import DiscoveryArchive
from amde.cli import main as cli_main
from amde.errors import CheckpointError, ConfigError
from amde.orchestrator import Engine, generate_report, run_status
from tests.support import make_project, small_config_data


def _fast_data(proposals: int = 1, confirm: int = 1, **overrides) -> dict:
    return small_config_data(proposals=proposals, confirm=confirm, **overrides)


def _run_cli(root: Path, *args: str) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = cli_main(["--root", str(root), *args])
    return code, output.getvalue()


class EngineIntegrationTests(unittest.TestCase):
    def test_cli_once_status_report_and_inspect(self) -> None:
        root, config = make_project(self, config_data=_fast_data())
        run_id = "cli_run"

        code, text = _run_cli(
            root,
            "once",
            "--run-id",
            run_id,
            "--generations",
            "1",
            "--json",
            "--no-resume",
        )
        self.assertEqual(code, 0, text)
        summary = json.loads(text)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["generations_completed"], 1)
        self.assertGreaterEqual(summary["trials_started"], 2)
        self.assertEqual(
            {item["name"] for item in summary["baselines"]},
            set(BASELINE_NAMES),
        )
        best = summary.get("best_candidate")
        self.assertIsNotNone(best)
        self.assertEqual(summary["arc"]["profile"]["status"], "BLOCKED")
        self.assertFalse(summary["resource_plan"]["gpu_effective"])
        self.assertEqual(summary["resource_plan"]["gpu_slots"], 0)
        global_archive = DiscoveryArchive(config.path("storage_dir") / "discovery_archive.sqlite3")
        archive_stats = global_archive.stats()
        self.assertGreaterEqual(archive_stats["candidate_count"], 1)
        self.assertGreaterEqual(archive_stats["sighting_count"], 1)

        code, text = _run_cli(root, "status", "--run-id", run_id, "--json")
        self.assertEqual(code, 0, text)
        status = json.loads(text)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(len(status["baselines"]), len(BASELINE_NAMES))
        self.assertEqual(len(status["generations"]), 1)
        self.assertEqual(status["generations"][0]["proposals"], 1)
        self.assertGreaterEqual(status["trial_counts"].get("ok", 0), 2)
        self.assertTrue(any(item.get("valid") for item in status["checkpoints"]))
        self.assertEqual(status["heartbeat"]["device_effective"], "cpu")
        self.assertIn("resources", status["heartbeat"])

        code, text = _run_cli(root, "report", "--run-id", run_id)
        self.assertEqual(code, 0, text)
        report_path = Path(text.strip())
        self.assertTrue(report_path.exists())
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("# AMDE Run Report", report)
        self.assertIn("## Baselines", report)
        self.assertIn("## Search Diagnostics & Stagnation", report)
        self.assertIn("## Interpretation Limits", report)
        self.assertIn("ARC profile: BLOCKED", report)

        candidate_hash = best["candidate_hash"]
        code, text = _run_cli(
            root,
            "inspect",
            "--run-id",
            run_id,
            "--candidate",
            candidate_hash,
            "--json",
        )
        self.assertEqual(code, 0, text)
        payload = json.loads(text)
        self.assertEqual(payload[0]["candidate"]["candidate_hash"], candidate_hash)
        self.assertGreaterEqual(len(payload[0]["trials"]), 1)
        self.assertIn("model", payload[0]["candidate"]["genome"])

        # The direct helpers must agree with the CLI views.
        direct = run_status(config, run_id)
        self.assertEqual(direct["status"], "completed")
        self.assertTrue(direct["checkpoints"])

    def test_stop_file_stops_then_resume_completes(self) -> None:
        root, config = make_project(self, config_data=_fast_data())
        run_id = "stop_resume"
        run_dir = config.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        stop_path = run_dir / str(config.get("paths.stop_file_name", "STOP"))
        stop_path.write_text("stop before search\n", encoding="utf-8")

        engine = Engine(config)
        first = engine.run(mode="once", run_id=run_id, generations=1, resume=False)
        self.assertEqual(first["status"], "stopped")
        self.assertEqual(first["generations_completed"], 0)
        self.assertEqual(first["trials_started"], 0)
        self.assertEqual(first["baselines"], [])
        self.assertTrue(stop_path.exists())
        stopped_status = run_status(config, run_id)
        self.assertEqual(stopped_status["status"], "stopped")
        self.assertTrue(any(item.get("valid") for item in stopped_status["checkpoints"]))

        stop_path.unlink()
        second = engine.run(mode="once", run_id=run_id, generations=1, resume=True)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["generations_completed"], 1)
        self.assertGreaterEqual(second["trials_started"], 2)
        self.assertEqual(len(second["baselines"]), len(BASELINE_NAMES))

        status = run_status(config, run_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(len(status["generations"]), 1)
        self.assertFalse(status["stop_requested"])
        events = [item["event"] for item in status["events"]]
        self.assertIn("run_resumed", events)

    def test_search_reproducibility_is_independent_of_run_id_and_guards_protocol_hash(self) -> None:
        root, config_a = make_project(self, config_data=_fast_data(proposals=3, confirm=1))
        config_b = load_config_copy(config_a)
        engine_a = Engine(config_a)
        engine_b = Engine(config_b)
        sum_a = engine_a.run(mode="once", run_id="rep_alpha", generations=1, resume=False)
        sum_b = engine_b.run(mode="once", run_id="rep_beta", generations=1, resume=False)
        self.assertEqual(
            sum_a["best_candidate"]["candidate_hash"],
            sum_b["best_candidate"]["candidate_hash"],
        )
        self.assertEqual(
            sum_a["best_candidate"]["quick_score"],
            sum_b["best_candidate"]["quick_score"],
        )

        # Changing seed or protocol must refuse to resume an existing run checkpoint.
        changed = load_config_copy(config_a)
        changed.set("seed", 9999)
        changed.refresh_hashes()
        with self.assertRaises(CheckpointError):
            Engine(changed).run(mode="once", run_id="rep_alpha", generations=2, resume=True)

    def test_explicit_challenge_is_aggregate_only_and_single_use(self) -> None:
        data = _fast_data(challenge=True)
        # This test asserts the challenge boundary/aggregation contract, not
        # training throughput. Keep its synthetic workload small so the
        # sealed worker's internal timeout cannot collide with the outer
        # integration-gate timeout on slower Windows hosts.
        data["evaluation"]["max_steps"] = 40
        data["challenge"].update(
            {
                "min_generations_between_openings": 99,
                "max_openings_per_run": 1,
                "seeds": [11, 12],
            }
        )
        root, config = make_project(self, config_data=data)
        run_id = "challenge_run"

        engine = Engine(config)
        first = engine.run(mode="once", run_id=run_id, generations=1, resume=False)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["challenge_openings"], [])
        candidate = first.get("best_confirmed") or first.get("best_candidate")
        self.assertIsNotNone(candidate)

        second = engine.run(
            mode="challenge",
            run_id=run_id,
            resume=True,
            candidate_hash=candidate["candidate_hash"],
        )
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(second["challenge_openings"]), 1)
        opening = second["challenge_openings"][0]
        self.assertEqual(opening["opening_index"], 1)
        self.assertIn("explicit sealed opening", opening["notes"])
        for key in ("candidate_aggregate", "baseline_aggregate"):
            summary = opening[key]
            self.assertIsInstance(summary, dict)
            self.assertNotIn("per_task", summary)
            self.assertNotIn("records", summary)
            self.assertNotIn("seed_aggregates", summary)
            self.assertEqual(summary["n_tasks"], 2)
            self.assertEqual(summary["n_seeds"], 2)

        status = run_status(config, run_id)
        latest = status["latest_challenge"]
        self.assertIsNotNone(latest)
        self.assertNotIn("per_task", latest["candidate_aggregate"])
        self.assertNotIn("records", latest["candidate_aggregate"])

        third = engine.run(mode="challenge", run_id=run_id, resume=True)
        self.assertEqual(third["status"], "exhausted")

    def test_storage_budget_pauses_after_durable_checkpoint(self) -> None:
        data = _fast_data(proposals=1, confirm=1)
        # Deliberately tiny quota so this test verifies the real coordinator path
        # without creating large files or depending on host disk pressure.
        data["budgets"]["max_disk_gb"] = 0.00001
        root, config = make_project(self, config_data=data)
        run_id = "storage_guard_test"
        summary = Engine(config).run(mode="once", run_id=run_id, generations=1, resume=False)
        self.assertEqual(summary["status"], "paused")
        self.assertIn("run storage budget reached", summary["error"]["message"])
        status = run_status(config, run_id)
        self.assertEqual(status["status"], "paused")
        self.assertTrue(any(item.get("valid") for item in status["checkpoints"]))

    def test_empty_project_status_and_report_fail_cleanly(self) -> None:
        root, config = make_project(self, config_data=_fast_data())
        status = run_status(config)
        self.assertEqual(status["status"], "no_runs")
        with self.assertRaises(ConfigError):
            generate_report(config)


def load_config_copy(config):
    from amde.config import Config

    return Config(config.effective_copy(), config.project_root, config.source_path)


if __name__ == "__main__":
    unittest.main()
