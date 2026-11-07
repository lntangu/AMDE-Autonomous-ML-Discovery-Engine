"""Unit Tests for OpenML-CC18 Benchmark Harness and Academic Paper Reporter.

Verifies:
1. OpenML-CC18 suite ID catalog integrity (72 datasets).
2. Dynamic fetching, caching, and loading.
3. Automated paper reporting calculations (Average Rank, Win/Tie/Loss, Wilcoxon).
4. Publication-ready LaTeX table generation.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.run_openml_cc18_suite import (
    OPENML_CC18_DATASET_IDS,
    get_openml_cc18_dataset_ids,
    append_master_result,
    read_completed_runs,
)
from scripts.generate_paper_artifacts import (
    compute_academic_metrics,
    format_latex_paper_table,
    generate_synthetic_demo_records,
)


class OpenMLCC18SuiteTests(unittest.TestCase):
    """Test suite for OpenML-CC18 harness and paper reporting."""

    def test_openml_cc18_dataset_ids_count(self) -> None:
        """Verifies that the OpenML-CC18 catalog contains exactly 72 curated datasets."""
        self.assertEqual(len(OPENML_CC18_DATASET_IDS), 72)
        dids = get_openml_cc18_dataset_ids(suite_id=99)
        self.assertGreaterEqual(len(dids), 70)
        self.assertIn(31, dids)   # credit-g
        self.assertIn(1590, dids) # adult

    def test_master_results_append_and_resume(self) -> None:
        """Verifies incremental appending and resumption filtering in master CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_master.csv"
            row1 = {
                "dataset_id": 31,
                "dataset_name": "credit-g",
                "n_samples": 1000,
                "n_features": 21,
                "n_classes": 2,
                "method": "amde",
                "seed": 42,
                "time_budget_s": 60,
                "test_accuracy": 0.775,
                "val_accuracy": 0.760,
                "wallclock_seconds": 15.2,
                "peak_memory_mb": 350.0,
                "status": "completed",
                "extra_info": "",
                "error_message": "",
            }
            append_master_result(csv_path, row1)
            self.assertTrue(csv_path.exists())

            completed = read_completed_runs(csv_path)
            self.assertIn((31, "amde", 42), completed)
            self.assertNotIn((31, "optuna", 42), completed)

    def test_academic_metrics_and_latex_table(self) -> None:
        """Verifies calculation of Average Rank, W/T/L, and LaTeX table synthesis."""
        demo_df = generate_synthetic_demo_records()
        self.assertEqual(len(demo_df), 30)  # 10 datasets * 3 methods

        metrics = compute_academic_metrics(demo_df)
        self.assertEqual(metrics["n_datasets"], 10)
        self.assertIn("amde", metrics["methods"])
        self.assertIn("flaml", metrics["methods"])
        self.assertIn("optuna", metrics["methods"])

        # Check rankings
        summaries = metrics["method_summaries"]
        self.assertIn("average_rank", summaries["amde"])
        self.assertLess(summaries["amde"]["average_rank"], summaries["optuna"]["average_rank"])

        # Check comparisons vs AMDE
        comps = metrics["comparisons_vs_amde"]
        self.assertIn("flaml", comps)
        self.assertIn("optuna", comps)
        self.assertGreaterEqual(comps["optuna"]["wins"], 0)
        self.assertIn("wilcoxon_p_value", comps["optuna"])

        # Check LaTeX formatting
        latex_str = format_latex_paper_table(metrics)
        self.assertIn(r"\begin{table*}", latex_str)
        self.assertIn(r"\end{table*}", latex_str)
        self.assertIn(r"\textbf{AMDE 4.1 (Ours)}", latex_str)
        self.assertIn(r"\toprule", latex_str)
        self.assertIn(r"\bottomrule", latex_str)


if __name__ == "__main__":
    unittest.main()
