"""Tests for reproducible, crash-resumable research campaign planning."""
from __future__ import annotations

import unittest
import random
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from pathlib import Path

from amde.config import load_config
from scripts.research_campaign import build_round_config, _compact_summary, parse_args, run_campaign
from amde.discovery.factory import default_genome
from amde.errors import ValidationError
from amde.orchestrator import _RunSession, Engine
from tests.support import make_project


ROOT = Path(__file__).resolve().parents[2]


class ResearchCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = load_config(project_root=ROOT)

    def test_campaign_defaults_to_cpu_backend(self) -> None:
        args = parse_args(["--campaign-id", "default_backend_check"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.rounds, 5)
        self.assertEqual(args.max_campaign_gb, 10.0)

    def test_round_config_is_deterministic_for_same_campaign_round(self) -> None:
        a = build_round_config(self.base, campaign_id="test_longrun", round_index=4, base_seed=123, device="cpu")
        b = build_round_config(self.base, campaign_id="test_longrun", round_index=4, base_seed=123, device="cpu")
        self.assertEqual(a.config_hash, b.config_hash)
        self.assertEqual(a.protocol_hash, b.protocol_hash)

    def test_round_config_changes_task_search_and_evaluation_seeds(self) -> None:
        a = build_round_config(self.base, campaign_id="test_longrun", round_index=4, base_seed=123, device="cpu")
        b = build_round_config(self.base, campaign_id="test_longrun", round_index=5, base_seed=123, device="cpu")
        self.assertNotEqual(a.get("seed"), b.get("seed"))
        self.assertNotEqual(
            [x["seed"] for x in a.get("benchmarks.task_specs")],
            [x["seed"] for x in b.get("benchmarks.task_specs")],
        )
        self.assertNotEqual(a.get("evaluation.seeds"), b.get("evaluation.seeds"))
        self.assertNotEqual(a.get("challenge.seeds"), b.get("challenge.seeds"))

    def test_round_config_keeps_holdout_separation_and_disables_auto_open(self) -> None:
        config = build_round_config(self.base, campaign_id="test_longrun", round_index=1, base_seed=42, device="cuda")
        self.assertEqual(len(config.dev_task_specs()), 13)
        self.assertEqual(len(config.challenge_task_specs()), 5)
        self.assertFalse(config.get("challenge.auto_open_during_search"))
        self.assertEqual(config.get("runtime.device"), "cuda")

    def test_compact_summary_prefers_confirmed_candidate(self) -> None:
        from scripts.research_campaign import _compact_summary
        result = _compact_summary({
            "status": "time_budget",
            "generations_completed": 3,
            "trials_started": 55,
            "best_candidate": {"candidate_hash": "quick", "quick_score": 0.6},
            "best_confirmed": {"candidate_hash": "confirmed", "confirm_score": 0.55},
        })
        self.assertEqual(result["best_candidate_hash"], "confirmed")
        self.assertEqual(result["best_score"], 0.55)

    def test_round_config_imports_only_archived_genome_structures(self) -> None:
        genome = default_genome("B")
        archive = [{
            "candidate_hash": "historical-candidate",
            "genome": genome,
            "source_confirm_score": 0.99,
        }]
        config = build_round_config(
            self.base,
            campaign_id="test_warm_start",
            round_index=2,
            base_seed=123,
            device="cpu",
            candidate_archive=archive,
        )
        self.assertEqual(config.get("search.seed_genomes"), [genome])
        # Historical fitness is not part of the transferred search state.
        self.assertNotIn("source_confirm_score", config.get("search.seed_genomes")[0])

    def test_search_seed_genomes_are_validated_and_re_evaluated(self) -> None:
        raw = default_genome("B")
        config = build_round_config(
            self.base,
            campaign_id="test_warm_start",
            round_index=3,
            base_seed=123,
            device="cpu",
            candidate_archive=[{"candidate_hash": "old", "genome": raw}],
        )
        session = object.__new__(_RunSession)
        session.config = config
        session.portfolio = None
        session.population = None
        session.archives = None
        session.rng = random.Random(123)
        session.proposer = None
        session._prepare_search()
        seeded = [item for item in session.population.all() if item.operator == "campaign_seed"]
        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0].genome.to_dict(), raw)
        self.assertIsNone(seeded[0].quick_score)
        self.assertIsNone(seeded[0].confirm_score)
        self.assertEqual(seeded[0].status, "campaign_seed_pending")

    def test_search_seed_genomes_fail_closed_on_invalid_structure(self) -> None:
        data = self.base.effective_copy()
        data.setdefault("search", {})["seed_genomes"] = [{"schema_version": "bad"}]
        from amde.config import Config
        config = Config(data, self.base.project_root)
        session = object.__new__(_RunSession)
        session.config = config
        session.portfolio = None
        session.population = None
        session.archives = None
        session.rng = random.Random(123)
        session.proposer = None
        with self.assertRaises(ValidationError):
            session._prepare_search()

    def test_campaign_seed_is_evaluated_directly_in_fresh_round(self) -> None:
        root, config = make_project(self)
        raw = default_genome("B")
        config.set("search.seed_genomes", [raw])
        config.refresh_hashes()
        run_id = "campaign_seed_e2e"
        summary = Engine(config).run(mode="once", run_id=run_id, generations=1, resume=False)
        self.assertEqual(summary["status"], "completed")
        checkpoint = json.loads((config.run_dir(run_id) / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
        from amde.discovery.genome import Genome
        expected_hash = Genome(raw, max_hidden_layers=3, max_hidden_width=48).hash
        members = checkpoint["state"]["population"]["members"]
        seeded = [item for item in members if item["candidate_hash"] == expected_hash]
        self.assertEqual(len(seeded), 1)
        self.assertIn(seeded[0]["status"], ("quick_ok", "confirmed"))
        # sqlite3.Connection.__exit__ commits/rolls back a transaction but does
        # not close the connection.  Close it explicitly before the TemporaryDirectory
        # cleanup runs; Windows refuses to unlink an open SQLite database.
        with closing(sqlite3.connect(config.run_dir(run_id) / "ledger.sqlite3")) as connection:
            rows = connection.execute(
                "SELECT stage, status FROM trials WHERE candidate_hash = ?", (expected_hash,)
            ).fetchall()
        self.assertTrue(any(stage == "quick" and status == "ok" for stage, status in rows))

    def test_two_round_campaign_re_evaluates_archived_genomes(self) -> None:
        root, _ = make_project(self)
        args = SimpleNamespace(
            root=str(root), config=None, campaign_id="two_round_check", rounds=2,
            hours_per_round=0.02, max_trials_per_round=3, max_campaign_gb=1.0,
            min_free_gb=1.0, seed=777, device="cpu",
        )
        self.assertEqual(run_campaign(args), 0)
        state_path = root / "reports" / "campaigns" / "two_round_check" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["next_round"], 2)
        self.assertEqual(len(state["rounds"]), 2)
        self.assertGreaterEqual(len(state["candidate_archive"]), 1)
        second_run = state["rounds"][1]["run_id"]
        checkpoint_path = root / "runs" / second_run / "checkpoints" / "latest.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        campaign_seeds = [
            member for member in checkpoint["state"]["population"]["members"]
            if member.get("operator") == "campaign_seed"
        ]
        self.assertGreaterEqual(len(campaign_seeds), 1)
        self.assertTrue(all(member.get("status") in ("quick_ok", "confirmed") for member in campaign_seeds))


if __name__ == "__main__":
    unittest.main()
