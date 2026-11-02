# AMDE 1.3.0 — Generalization audit protocol

## Objective

The audit compares one frozen candidate with one fixed baseline on **paired, independently defined benchmark families**. The family, not individual examples or repeated seeds, is the resampling unit. Candidate and baseline must have scores on exactly the same families.

This protocol is intended to support claims such as “the candidate improved over baseline on these preregistered, unseen external benchmark families.” It cannot prove universal, distribution-free or all-domain generalization.

## Separation of roles

- **Search/development:** may be used to generate, tune and choose candidates. Its scores are not held-out evidence.
- **Validation/selection:** may be used to select the final frozen candidate; if any evaluation family is used for this selection, set `evaluation_used_for_selection: true` and the confirmatory gate must fail.
- **Final evaluation:** must use a frozen candidate, fixed baseline, sealed or otherwise access-controlled datasets/families not seen during search, and no post-hoc threshold changes.
- **Preregistration:** record candidate-selection rule, family inclusion/exclusion, metrics, minimum effect, minimum win rate, maximum family regression, seed plan, sample-size rule and statistical method before revealing final scores. The current CLI records the supplied `preregistered` attestation; it does not independently verify a timestamped preregistration repository.

## Input schema

Create a JSON file with this structure:

```json
{
  "schema_version": "amde-generalization-input-v1",
  "candidate_id": "frozen-candidate-hash",
  "baseline_id": "fixed-baseline-name-and-version",
  "benchmark_protocol_id": "protocol-name-version",
  "benchmark_protocol_hash": "sha256-of-preregistered-protocol",
  "preregistered": true,
  "evaluation_used_for_selection": false,
  "minimum_families": 10,
  "minimum_external_families": 10,
  "minimum_domains": 3,
  "minimum_effect": 0.01,
  "minimum_win_rate": 0.65,
  "maximum_family_regression": 0.10,
  "candidate_family_scores": {"family_a": 0.81},
  "baseline_family_scores": {"family_a": 0.77},
  "family_metadata": {
    "family_a": {"source": "external", "domain": "tabular", "seen_during_search": false}
  }
}
```

The one-family JSON fragment above illustrates the schema only; it is deliberately too small to pass the default gates. Add at least 10 genuinely independent families, and at least 10 external families across 3 or more domains for the external evidence tier. Do not label a dataset “external” unless it is independent of the project and its candidate-selection process.

Scores must be comparable and paired: same train/validation/test protocol, metric definition, data version, split, sample budget and compute budget for candidate and baseline. If seeds are repeated inside one family, aggregate them to one prespecified family-level score before this audit. Do not count seeds, folds, examples or tasks from one correlated source as independent families.

## Run the audit

```powershell
.\.venv\Scripts\python.exe scripts\generalization_audit.py .\my_generalization_results.json --out .\reports\generalization_audit --bootstrap 20000 --seed 42
```

The report contains a family-cluster bootstrap percentile 95% interval, paired sign-flip test (exact up to 16 families and Monte Carlo above that), family win/loss/tie counts, lower-quartile paired delta, worst-family regression, provenance and gate decisions. The Monte Carlo p-value uses a plus-one correction.

## Evidence tiers and interpretation

- `insufficient_or_selection_contaminated`: the key separation, preregistration or sample gates are missing.
- `preliminary_unseen_family_evidence`: limited unseen-family evidence, not confirmatory.
- `unseen_family_synthetic_or_mixed_support`: preregistered, uncontaminated evidence across the configured count of unseen families, possibly synthetic or mixed-source.
- `multi_domain_external_holdout_support`: additionally all evaluated families are labelled external, the minimum external family count is met, and multiple domains are represented.

These are empirical evidence labels, not proofs. Independence of families and metadata provenance must be justified by the researcher. The CLI validates the data shape and calculations; it cannot independently authenticate a benchmark publisher, dataset license, data split or claim that the candidate was actually frozen before evaluation.

## Required research report

Preserve the raw per-family scores, immutable benchmark protocol, source/version/license, split hashes, candidate and baseline hashes/configurations, execution logs, hardware/compute budgets, all failed/aborted runs, preregistration artifact, audit JSON and Markdown, and a reproducible script. Report all prespecified metrics and negative results; do not report only the best candidate or best seed.


## Preregistered hypothesis and ablation utilities (1.3.0)

`scripts/research_lab.py` freezes hypothesis protocols in an append-only SQLite registry and stores result hashes. A hypothesis protocol specifies at least six independent benchmark-family identifiers, unique seeds, direction, minimum effect, alpha, expected standard deviation and target power. The power estimate uses a transparent paired-mean normal approximation and is marked inadequate when the preregistered family count is below its estimate; an underpowered result cannot be labelled supported or falsified.

Ablation audits require complete, identical family-seed coverage between each variant and the control, aggregate seed replicates within family, and apply Holm-Bonferroni across tested variants. Family clusters, not individual seeds, are the inferential units. The utilities do not authenticate dataset provenance, create independence between related datasets, or eliminate all forms of researcher degrees of freedom.
