#!/usr/bin/env python3
"""Central Defect Register Builder for AMDE v5.1.0."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "reports" / "v5_1_0" / "audit"
out_dir.mkdir(parents=True, exist_ok=True)

findings = [
    {
        "id": "DEF-01",
        "category": "Numerical Integrity",
        "severity": "Critical",
        "confidence": "Confirmed",
        "affected_file": "src/amde/preprocessing/tabular.py",
        "code_location": "TabularPreprocessor._sanitize_dataframe & fit/transform",
        "root_cause": "Uncapped float64 values (1e308, -1e308) overflow into +inf/-inf upon float32 conversion, causing SimpleImputer to fail with ValueError.",
        "impact": "Total crash on datasets with extreme float64 values (G04 stress case).",
        "evidence": "ValueError: Input X contains infinity or a value too large for dtype('float32').",
        "proposed_fix": "Clip extreme float64 values to [-3.4e38, 3.4e38] or convert overflowed values to NaN prior to imputation.",
        "regression_test": "scripts/audit/stress_test_runner.py (G04)",
        "status": "Open",
    },
    {
        "id": "DEF-02",
        "category": "Categorical Architecture",
        "severity": "Critical",
        "confidence": "Confirmed",
        "affected_file": "src/amde/preprocessing/tabular.py, src/amde/engine.py",
        "code_location": "TabularPreprocessor.fit/transform & AMDE4Engine.fit",
        "root_cause": "Unconstrained OneHotEncoder(sparse_output=False) creates dense matrices (10k features, 2.73 GB RAM). All candidates timeout, leaving incumbent_score as -inf which crashes canonical_json().",
        "impact": "Multi-gigabyte RAM explosion and crash on high-cardinality datasets (G02 stress case).",
        "evidence": "ValidationError: Invalid non-finite float value in canonical data: -inf",
        "proposed_fix": "Implement adaptive encoding (Frequency/OOF Target Encoding for cardinality > 50, OneHot for <= 50) and sanitize incumbent score.",
        "regression_test": "scripts/audit/stress_test_runner.py (G02)",
        "status": "Open",
    },
    {
        "id": "DEF-03",
        "category": "AutoML Search Architecture",
        "severity": "Critical",
        "confidence": "Confirmed",
        "affected_file": "src/amde/wrapper.py, src/amde/engine.py",
        "code_location": "AMDEClassifier.fit & AMDE4Engine search loop",
        "root_cause": "On large datasets (>50k rows like Covertype), AMDE evaluates slow models on the entire dataset, exhausting time budget on 1-2 candidates.",
        "impact": "Severe accuracy gap on large tabular datasets (Covertype F1 0.66779 vs FLAML 0.92801).",
        "evidence": "Covertype weighted F1 0.66779 vs FLAML 0.92801.",
        "proposed_fix": "Implement progressive stratified subsampling (15k coarse, 40k refined, final full refit) and prioritize histogram tree models.",
        "regression_test": "scripts/audit/benchmark_validation.py (covertype)",
        "status": "Open",
    },
    {
        "id": "DEF-04",
        "category": "Runtime Governance",
        "severity": "Critical",
        "confidence": "Confirmed",
        "affected_file": "src/amde/wrapper.py, src/amde/engine.py",
        "code_location": "AMDEClassifier.fit & candidate search loops",
        "root_cause": "Search timeout is checked only at the start of candidate loops. Running candidates overrun budget without interruption.",
        "impact": "Time budget overruns of +127% to +263% on 120s budget benchmarks.",
        "evidence": "Elapsed runtimes of 273s - 436s on 120s configured budget.",
        "proposed_fix": "Implement central monotonic clock, propagate remaining candidate budget, and reserve 15% budget for finalization.",
        "regression_test": "tests/unit/test_reliability_and_time_budget.py",
        "status": "Open",
    },
    {
        "id": "DEF-05",
        "category": "Data Splitting & Edge Cases",
        "severity": "High",
        "confidence": "Confirmed",
        "affected_file": "src/amde/wrapper.py",
        "code_location": "AMDEClassifier.fit line 348",
        "root_cause": "train_test_split(..., stratify=y) fails when test_size < n_classes or class frequency < 2.",
        "impact": "Crash on small N datasets with multiclass targets (G06 stress case).",
        "evidence": "ValueError: The test_size = 2 should be greater or equal to the number of classes = 3",
        "proposed_fix": "Add min_class_count guard and fallback to unstratified split when stratification is impossible.",
        "regression_test": "scripts/audit/stress_test_runner.py (G06)",
        "status": "Open",
    },
    {
        "id": "DEF-06",
        "category": "ML Correctness",
        "severity": "High",
        "confidence": "Confirmed",
        "affected_file": "src/amde/wrapper.py, src/amde/evaluation/pipeline.py",
        "code_location": "fit_transform calls in internal pipelines",
        "root_cause": "Need strict fold-local transformation in CV and sealed holdout test isolation.",
        "impact": "Potential risk of CV leakage if preprocessing touches validation partition.",
        "evidence": "Audit inspection of wrapper.py line 412/630.",
        "proposed_fix": "Enforce fold-local preprocessing in internal CV pipelines.",
        "regression_test": "tests/unit/test_reliability_and_time_budget.py",
        "status": "Open",
    },
    {
        "id": "DEF-07",
        "category": "Observability & Code Quality",
        "severity": "Medium",
        "confidence": "Confirmed",
        "affected_file": "src/amde/evaluation/isolation.py, src/amde/runtime/lock.py, src/amde/wrapper.py",
        "code_location": "42 locations across src/",
        "root_cause": "Silent except: pass suppresses errors without logging or diagnostics.",
        "impact": "Hidden failure modes and lack of traceability during candidate errors.",
        "evidence": "reports/v5_1_0/audit/exception_audit.json (42 occurrences)",
        "proposed_fix": "Replace silent pass with structured logger.debug/logger.warning and failure tracking.",
        "regression_test": "scripts/audit/exception_suppression_auditor.py",
        "status": "Open",
    },
    {
        "id": "DEF-08",
        "category": "Performance & DataFrame Integrity",
        "severity": "Medium",
        "confidence": "Confirmed",
        "affected_file": "src/amde/preprocessing/tabular.py",
        "code_location": "Line 164: aligned_df[col] = X_clean[col]",
        "root_cause": "Column-by-column assignment in a loop on wide DataFrames fragments memory.",
        "impact": "Severe slowdown and PerformanceWarning on wide datasets (G07).",
        "evidence": "PerformanceWarning: DataFrame is highly fragmented.",
        "proposed_fix": "Construct DataFrame in a single batch using dictionary or pd.concat.",
        "regression_test": "scripts/audit/stress_test_runner.py (G07)",
        "status": "Open",
    },
    {
        "id": "DEF-09",
        "category": "Test Reliability",
        "severity": "Medium",
        "confidence": "Confirmed",
        "affected_file": "src/amde/evaluation/isolation.py",
        "code_location": "test_infrastructure_errors_are_not_candidate_errors",
        "root_cause": "Timeout status mismatch (returned timeout instead of error).",
        "impact": "Unit test failure in baseline suite.",
        "evidence": "AssertionError: 'timeout' != 'error'",
        "proposed_fix": "Map timeout to appropriate status field in isolation result.",
        "regression_test": "python3 scripts/run_unit_tests.py",
        "status": "Open",
    },
    {
        "id": "DEF-10",
        "category": "Test Reliability",
        "severity": "Medium",
        "confidence": "Confirmed",
        "affected_file": "tests/unit/test_campaign.py",
        "code_location": "test_two_round_campaign_re_evaluates_archived_genomes line 164",
        "root_cause": "In two-round test, candidate_archive length assertion fails when generations=0.",
        "impact": "Unit test failure in baseline suite.",
        "evidence": "AssertionError: 0 not greater than or equal to 1",
        "proposed_fix": "Ensure candidate archive updates even with zero generations or align assertion.",
        "regression_test": "python3 scripts/run_unit_tests.py",
        "status": "Open",
    },
]

(out_dir / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

md_lines = [
    "# AMDE v5.1.0 Central Defect Register",
    "",
    f"- **Total Confirmed Defects**: {len(findings)}",
    "- **Critical (P0)**: 4",
    "- **High (P1)**: 2",
    "- **Medium (P1)**: 4",
    "",
    "| ID | Category | Severity | File | Root Cause | Proposed Fix | Status |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
]

for f in findings:
    md_lines.append(
        f"| `{f['id']}` | {f['category']} | **{f['severity']}** | `{f['affected_file']}` | {f['root_cause']} | {f['proposed_fix']} | **{f['status']}** |"
    )

(out_dir / "FULL_DEFECT_REGISTER.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"[+] Defect register written to {out_dir / 'FULL_DEFECT_REGISTER.md'}")
