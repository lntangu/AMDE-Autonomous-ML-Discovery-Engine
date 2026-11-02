# AMDE 1.3.0 — External tabular benchmark evaluation

## Purpose and scope

`scripts/evaluate_external_benchmarks.py` evaluates one frozen AMDE genome against one fixed baseline over explicitly supplied CSV benchmark families. Candidate and baseline receive the same family datasets, duplicate-grouped splits, seed list, evaluator, device, maximum-step cap and parameter cap. The tool preserves dataset hashes, candidate/baseline configurations, per-seed metrics, realized step counts and elapsed time, then runs the family-clustered statistical audit.

This tool supports tabular binary classification (`target` exactly 0/1) and regression. A single manifest must contain only one task kind because balanced accuracy and clipped R² are not interchangeable metrics. Run separate manifests for classification and regression.

## Manifest format

Create JSON matching `amde-external-datasets-v1`:

```json
{
  "schema_version": "amde-external-datasets-v1",
  "dataset_families": [
    {
      "family_id": "family_001",
      "domain": "materials_science",
      "source": "external",
      "source_version": "publisher-release-or-date",
      "license": "dataset-license",
      "reference": "publisher citation or DOI",
      "task_kind": "classification",
      "feature_columns": ["x1", "x2", "x3"],
      "categorical_columns": ["x3"],
      "target_column": "label",
      "csv_path": "datasets/family_001.csv",
      "seen_during_search": false
    }
  ]
}
```

- `csv_path` is resolved relative to the manifest, unless absolute.
- Targets must be finite numeric values. Binary classification labels must be exactly 0 and 1. Feature columns may contain finite numeric values, categorical text, or missing values (``, `NA`, `N/A`, `NaN`, `null`, `none`, `missing`).
- Optional `categorical_columns` explicitly declares string-valued categorical fields, including numeric-looking identifiers such as `00123` whose leading zeroes must be preserved. Other feature columns are inferred as numeric when all observed non-missing values parse as finite numbers; otherwise they are treated as categorical.
- Preprocessing is fitted on the training partition only: numeric features use training-median imputation plus a missingness indicator; categorical features use a deterministic training-only one-hot vocabulary and explicit other/missing/unknown buckets. Validation/test data cannot extend the fitted vocabulary. The applied encoder metadata is written into per-seed result records.
- Binary classification labels must be exactly 0 and 1. Regression accepts finite numeric targets.
- Each family must contain at least 30 rows and 20 unique feature vectors. Binary classification needs at least 3 unique feature vectors per class.
- Identical normalized raw feature vectors are grouped and assigned to one split. If the same feature vector has conflicting labels, ingestion fails. Exact byte-identical CSV files cannot be counted as separate benchmark families. This prevents two simple forms of duplicate inflation; it does not detect near-duplicates, entity leakage, temporal leakage, overlapping cohorts with different files or source-level dependence.
- Default resource limits are 200,000 rows, 128 features and 512 MiB per CSV file.
- `seen_during_search` must be a boolean. Missing values default conservatively to `true`, which prevents that family being counted as unseen. The value is a researcher attestation, not something AMDE can independently verify.
- `source` is `external`, `synthetic` or `internal`. External entries require source version, license and reference. AMDE records those strings but cannot verify copyright or license compliance.
- An external-generalization claim needs at least 10 independent external families across at least 3 genuinely distinct domains. Multiple files from one underlying cohort do not automatically count as independent families.

The included `examples/external_benchmarks/manifest.example.json` is **synthetic demonstration data**, deliberately marked `seen_during_search=true`. It must never be presented as external generalization evidence.

## Run an evaluation

First freeze a candidate genome JSON and record its SHA-256 identity. Select the fixed baseline and thresholds before exposing final holdout scores. Then run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_external_benchmarks.py `
  .\examples\external_benchmarks\manifest.example.json `
  .\examples\candidate_linear_adam.json `
  --baseline mlp_medium `
  --seeds 11,23,37 `
  --device cpu `
  --max-steps 1200 `
  --preregistered `
  --sealed-holdout `
  --out .\reports\external_demo
```

The example has one synthetic family, so the report will correctly fail both generalization gates. The flags are attestations; use them only when the evaluation protocol was actually preregistered and the families were actually held out from search, tuning and candidate selection. For a real confirmatory evaluation, create a separate manifest with real independent external data and keep the final test labels inaccessible to the search process until the candidate is frozen.

Output files:

- `external_evaluation.json`: manifest/data hashes, candidate and baseline genomes, per-seed and per-family results, and provenance.
- `protocol.json`: exact protocol inputs hashed into the run identifier.
- `generalization_audit.json`: family-clustered bootstrap/permutation analysis and all gate decisions.
- `EXTERNAL_BENCHMARK_REPORT.md`: readable result summary, per-family deltas and realized training-step counts.

The runner atomically checkpoints each completed family. After a controlled interruption, rerun the exact command with `--resume`; it verifies the full protocol hash (including dataset file hashes, candidate/baseline identity, seeds and limits) before skipping completed families. Use `--overwrite` only when intentionally discarding an earlier output set. The report explicitly shows candidate and baseline realized steps and elapsed time. The shared `max_steps` and parameter limits are caps, not a guarantee of equal realized compute. Before claiming compute parity, normalize the actual compute budget or report this difference transparently.

## Baseline choices

Built-in baselines: `majority_mean`, `linear_sgd`, `linear_adam`, `linear_adagrad`, `linear_nadam`, `mlp_small`, `mlp_medium`. Optional scikit-learn baselines: `extra_trees`, `random_forest`, and `hist_gradient_boosting`. Install them with `python -m pip install -e .[tabular]`. The evaluation protocol stores the scikit-learn version and estimator parameters in `baseline_identity`, so changing the optional library version changes the protocol hash. Tree baselines run on CPU; the report records that fact even if the candidate is evaluated on CUDA. Compare predictive scores separately from wall-clock speed when the execution backends differ.

## Statistical interpretation

The bootstrap and paired sign-flip test resample independent benchmark families, not rows or repeated seeds. Ablation analysis also supports paired family/seed comparisons with Holm-Bonferroni correction across multiple variants. Repeated splits/seeds are averaged into one score per family. The confirmatory gate requires a preregistered protocol, a frozen candidate, unseen families, positive lower 95% family-bootstrap confidence bound beyond the minimum effect, paired p-value <= 0.05, family win rate above the threshold, and no family regression beyond the configured maximum. The multi-domain external tier also requires all families labeled external and the minimum external-family/domain counts.

Passing these gates is **empirical support within the sampled domains and protocol**, not universal proof. Independence, dataset provenance, lawful use, true holdout separation and absence of hidden leakage remain research responsibilities. Report all families, failures, aborted runs, baselines, seeds and negative results; do not select the strongest of many candidates on the final holdout and then reuse the same holdout as confirmatory evidence.
