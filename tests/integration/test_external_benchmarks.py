from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from amde.baselines.registry import make_baseline_genome


class ExternalBenchmarkIntegrationTests(unittest.TestCase):
    def test_external_csv_cli_writes_hashes_and_fails_closed_on_demo_scope(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            csv_path = work / "family.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["f1", "f2", "target"])
                for i in range(60):
                    x1 = float(i) / 10.0
                    x2 = float((i * 7) % 13) / 5.0
                    writer.writerow([x1, x2, int((i % 2) == 0)])
            manifest = {
                "schema_version": "amde-external-datasets-v1",
                "dataset_families": [{
                    "family_id": "synthetic_demo_only",
                    "domain": "synthetic_demo",
                    "source": "synthetic",
                    "task_kind": "classification",
                    "feature_columns": ["f1", "f2"],
                    "target_column": "target",
                    "csv_path": "family.csv",
                    "seen_during_search": True,
                }],
            }
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            genome = make_baseline_genome("linear_adam")
            assert genome is not None
            genome_path = work / "candidate.json"
            genome_path.write_text(json.dumps(genome.to_dict()), encoding="utf-8")
            output = work / "output"
            command = [
                sys.executable, str(root / "scripts" / "evaluate_external_benchmarks.py"),
                str(manifest_path), str(genome_path),
                "--baseline", "mlp_medium",
                "--seeds", "11,23,37",
                "--device", "cpu",
                "--max-steps", "120",
                "--epochs-scale", "0.1",
                "--bootstrap", "1000",
                "--preregistered", "--sealed-holdout",
                "--out", str(output),
            ]
            process = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + "\n" + process.stderr)
            result = json.loads((output / "external_evaluation.json").read_text(encoding="utf-8"))
            audit = json.loads((output / "generalization_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["protocol"]["family_csv_sha256"]["synthetic_demo_only"]), 64)
            self.assertEqual(audit["evidence_tier"], "insufficient_or_selection_contaminated")
            self.assertFalse(audit["external_generalization_gate"])
            self.assertTrue((output / "EXTERNAL_BENCHMARK_REPORT.md").is_file())

            # Simulate a completed-family checkpoint and verify exact-protocol resume.
            partial = {
                "schema_version": "amde-external-evaluation-partial-v1",
                "protocol_hash": result["benchmark_protocol_hash"],
                "completed_family_ids": ["synthetic_demo_only"],
                "candidate_results": result["candidate_results"],
                "baseline_results": result["baseline_results"],
                "started_unix": 1_700_000_000.0,
            }
            (output / "external_evaluation.partial.json").write_text(json.dumps(partial), encoding="utf-8")
            for name in ("external_evaluation.json", "generalization_audit.json", "protocol.json", "EXTERNAL_BENCHMARK_REPORT.md"):
                (output / name).unlink()
            resumed = subprocess.run(command + ["--resume"], cwd=root, text=True, capture_output=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + "\n" + resumed.stderr)
            self.assertIn("already checkpointed; skipping", resumed.stdout)
            self.assertFalse((output / "external_evaluation.partial.json").exists())

            duplicate = subprocess.run(command + ["--resume"], cwd=root, text=True, capture_output=True, timeout=60)
            self.assertEqual(duplicate.returncode, 2)
            self.assertIn("completed evaluation files already exist", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
