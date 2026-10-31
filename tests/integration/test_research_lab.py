from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research_lab.py"


class ResearchLabIntegrationTests(unittest.TestCase):
    def test_research_lab_register_evaluate_ablation_and_archive_commands(self):
        with tempfile.TemporaryDirectory(prefix="amde_research_lab_") as temp:
            root = Path(temp)
            db = root / "research.sqlite3"
            protocol = {
                "hypothesis_id": "integration_sparse_effect_01",
                "claim": "A sparse-aware optimizer improves held-out family scores.",
                "mechanism": "The update suppresses noise in inactive gradient coordinates.",
                "population": "Synthetic independent sparse benchmark families.",
                "intervention": {"optimizer": "sparse_aware"},
                "comparator": {"optimizer": "adamw"},
                "outcome": "Paired family-balanced score difference.",
                "direction": "greater", "minimum_effect": 0.05, "alpha": 0.05,
                "expected_sd": 0.02, "target_power": 0.80,
                "benchmark_families": [f"family_{i}" for i in range(1, 8)],
                "seeds": [11, 23, 37],
                "exclusion_criteria": ["Exclude infrastructure failures, do not score them as zero."],
                "analysis_plan": "Family-level bootstrap with paired exact sign-flip test.",
            }
            protocol_path = root / "hypothesis.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            differences_path = root / "differences.json"
            differences_path.write_text(json.dumps({f"family_{i}": 0.08 for i in range(1, 8)}), encoding="utf-8")
            register = subprocess.run(
                [sys.executable, str(SCRIPT), "hypothesis-register", str(protocol_path), "--db", str(db)],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(register.returncode, 0, msg=register.stdout + register.stderr)
            evaluate = subprocess.run(
                [sys.executable, str(SCRIPT), "hypothesis-evaluate", str(protocol_path),
                 str(differences_path), "--db", str(db), "--bootstrap", "1000",
                 "--out", str(root / "hypothesis_result.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(evaluate.returncode, 0, msg=evaluate.stdout + evaluate.stderr)
            result = json.loads((root / "hypothesis_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "supported")

            batch_path = root / "hypothesis_batch.json"
            batch_payload = {
                "schema_version": "amde-hypothesis-batch-v1",
                "expected_hypothesis_ids": [result["hypothesis_id"]],
                "results": [result],
            }
            batch_path.write_text(json.dumps(batch_payload), encoding="utf-8")
            batch = subprocess.run(
                [sys.executable, str(SCRIPT), "hypothesis-batch-audit", str(batch_path),
                 "--out", str(root / "hypothesis_batch_audit.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(batch.returncode, 0, msg=batch.stdout + batch.stderr)
            batch_data = json.loads((root / "hypothesis_batch_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(batch_data["correction"], "Holm-Bonferroni")

            base_path, factor_path = root / "base.json", root / "factors.json"
            base_path.write_text(json.dumps({"optimizer": "adam", "lr": 0.01}), encoding="utf-8")
            factor_path.write_text(json.dumps({"optimizer": ["adam", "sgd"], "lr": [0.001, 0.01]}), encoding="utf-8")
            plan = subprocess.run(
                [sys.executable, str(SCRIPT), "ablation-plan", str(base_path), str(factor_path),
                 "--mode", "full_factorial", "--seeds", "1,2,3", "--out", str(root / "plan.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(plan.returncode, 0, msg=plan.stdout + plan.stderr)
            plan_data = json.loads((root / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan_data["variants"]), 4)

            archive = subprocess.run(
                [sys.executable, str(SCRIPT), "archive-stats", "--db", str(root / "archive.sqlite3")],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(archive.returncode, 0, msg=archive.stdout + archive.stderr)
            self.assertIn('"candidate_count": 0', archive.stdout)

            power = subprocess.run(
                [sys.executable, str(SCRIPT), "power-analysis", "--minimum-effect", "0.05",
                 "--expected-sd", "0.10", "--alpha", "0.05", "--target-power", "0.80",
                 "--out", str(root / "power.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(power.returncode, 0, msg=power.stdout + power.stderr)
            power_data = json.loads((root / "power.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(power_data["recommended_independent_families"], 6)


if __name__ == "__main__":
    unittest.main()
