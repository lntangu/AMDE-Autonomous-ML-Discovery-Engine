# AMDE 1.3.0 — Research toolkit

The research toolkit is a small, dependency-free set of utilities around the AMDE search engine. It makes preregistration, ablation planning, hypothesis evaluation and historical candidate memory explicit. It does not automatically turn arbitrary observations into validated scientific discoveries.

## 1. Cross-campaign discovery archive

The engine appends successfully evaluated candidates to `storage/discovery_archive.sqlite3`. Exact canonical genome SHA-256 is the identity key; canonical JSON is retained and checked for a theoretical hash collision. Optional behaviour signatures are diagnostic only and never suppress candidates. Archive writes are secondary to the per-run SQLite ledger: an archive-capacity or archive-I/O issue is recorded as a warning and must not invalidate an otherwise successful campaign.

Default bounds are 100,000 unique genomes and 1,000,000 sightings. Override `search.global_archive_max_candidates` and `search.global_archive_max_sightings` in the project config to change the limits. `search.global_archive_seed_count` controls how many recent archived genomes may be offered as warm starts (0–64). Historical scores are descriptive metadata and are not compared across protocol hashes or imported as fitness. Every warm-start genome is re-evaluated on the new campaign's task manifest and seeds.

Inspect the archive:

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py archive-stats --limit 20
```

## 2. Power planning

The `power-analysis` command estimates the number of independent benchmark families needed to detect a preregistered minimum effect, given an assumed standard deviation, alpha and target power:

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py power-analysis `
  --minimum-effect 0.05 --expected-sd 0.10 `
  --alpha 0.05 --target-power 0.80 --direction two-sided `
  --out reports\research\power_analysis.json
```

The method is a normal approximation for a paired family-level mean difference. It does not apply a finite-sample Student-t correction, cannot verify the independence of benchmark families, and cannot guarantee realized power. The expected standard deviation must be justified using independent pilot evidence or conservative assumptions. Treat the recommendation as a planning floor, not a universal sample-size rule.

## 3. Hypothesis preregistration

A hypothesis JSON must include:

- `hypothesis_id`, `claim`, `mechanism`, `population`;
- `intervention`, `comparator`, `outcome`, and `direction` (`greater`, `less`, or `two-sided`);
- `minimum_effect`, `alpha`, `expected_sd`, `target_power`;
- at least six unique `benchmark_families`, at least three unique integer `seeds`;
- `exclusion_criteria` and a specific `analysis_plan`.

Example protocol fragment (replace with the actual experiment and freeze it before evaluation):

```json
{
  "hypothesis_id": "nadam_sparse_family_001",
  "claim": "NAdam improves a prespecified outcome on sparse tabular families.",
  "mechanism": "Nesterov-corrected first-moment estimates may improve updates on sparse gradients.",
  "population": "Independent sparse tabular benchmark families not used in search.",
  "intervention": {"optimizer": "nadam"},
  "comparator": {"optimizer": "adam"},
  "outcome": "Family-balanced paired score difference.",
  "direction": "greater",
  "minimum_effect": 0.05,
  "alpha": 0.05,
  "expected_sd": 0.10,
  "target_power": 0.80,
  "benchmark_families": ["sparse_a", "sparse_b", "sparse_c", "sparse_d", "sparse_e", "sparse_f"],
  "seeds": [11, 23, 37, 53, 71],
  "exclusion_criteria": ["Infrastructure failures are not imputed as zero."],
  "analysis_plan": "Paired family-level bootstrap interval and exact sign-flip test; no post-hoc family exclusion."
}
```

Register the immutable protocol, then evaluate one paired mean difference per preregistered family:

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py hypothesis-register .\hypothesis.json
.\.venv\Scripts\python.exe scripts\research_lab.py hypothesis-evaluate .\hypothesis.json .\family_differences.json --bootstrap 20000
```

The protocol is stored append-only in `storage/research/hypotheses.sqlite3`; a changed protocol under the same ID is rejected. A new version requires a new ID. If the registered family count is below the normal-approximation power target, the result is forced to `inconclusive_underpowered`, even when its observed p-value looks small. This deliberately prevents underpowered pilot evidence from being promoted to support/falsification.

When a study tests multiple preregistered hypotheses, place the **complete hypothesis family** in a batch file and apply a second Holm-Bonferroni correction across hypotheses:

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py hypothesis-batch-audit .\hypothesis_batch.json --alpha 0.05
```

The file must include the frozen `expected_hypothesis_ids` list and every corresponding `amde-hypothesis-result-v1` result. Missing or extra hypotheses are rejected. The audit distinguishes a family-wise significant result from support for the preregistered minimum effect; underpowered hypotheses remain inconclusive. This only controls multiplicity over the complete prespecified set supplied, not hypotheses omitted after looking at results.

The tool validates exact family coverage but cannot authenticate provenance, family independence or whether the protocol was truly frozen before data access. The researcher must retain those records.

## 4. Ablation planning and audit

Create a bounded deterministic plan:

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py ablation-plan .\base_config.json .\factors.json --mode one_factor --seeds 11,23,37,53,71
```

`factors.json` maps top-level config keys to two or more candidate values. `one_factor` changes one factor at a time; `full_factorial` explores the Cartesian product and rejects plans above `--max-variants`.

The result file for `ablation-audit` must have `schema_version: "amde-ablation-results-v1"`, a control ID and rows with `variant_id`, `family_id`, integer `seed` and finite `score`. Each treatment must have exactly the same family-seed keys as control. At least six independent families are required. Replicate seeds are averaged within family before inference; Holm-Bonferroni adjusts across all tested variants.

```powershell
.\.venv\Scripts\python.exe scripts\research_lab.py ablation-audit .\ablation_results.json --alpha 0.05 --bootstrap 20000
```

Ablation results only support confirmatory interpretation when the design, variants, exclusions, outcome, family set and analysis plan were frozen before results were inspected. They do not identify causal mechanisms by themselves; controlled interventions and assumptions are required.

## Scope boundaries

AMDE 1.3.0 expands small tabular ML method discovery and its research process. The built-in model family remains linear/small MLP; the release does not include CNN/Transformer/GNN NAS, a general symbolic theorem prover, authenticated dataset provenance, physical QPU execution, or universal generalization guarantees. See `OPEN_ENDED_SEARCH.md`, `EXTERNAL_BENCHMARKS.md` and `GENERALIZATION_PROTOCOL.md` for the limits of each component.
