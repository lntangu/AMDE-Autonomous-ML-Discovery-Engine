# AMDE 1.3.0 — Research Platform Expansion

## Main changes

- Persistent cross-campaign discovery archive with canonical genome identity, append-only sightings, bounded storage and re-evaluated warm starts. Historical scores are never ranked across different protocol hashes.
- Expanded optimizer search with AdaGrad and corrected NAdam reference/PyTorch implementations, parity tests and baselines.
- Research toolkit for immutable hypothesis preregistration, family-level power planning, paired ablation design/audit, within-study Holm-Bonferroni and family-wise Holm-Bonferroni across a complete prespecified hypothesis batch.
- External mixed-type CSV ingestion with training-only numeric imputation, missingness flags, categorical one-hot encoding and explicit unknown/other buckets.
- Optional scikit-learn Extra Trees, Random Forest and HistGradientBoosting baselines for external tabular evaluation. Library version and model parameters are included in the protocol identity; these adapters run on CPU.
- Parallel full-suite runner with isolated unit/integration processes, independent logs, timeout controls and machine-readable release report.

## Compatibility and safety

- Existing 1.2.x run folders and campaign records must be preserved. Use a fresh project folder and new run/campaign IDs for 1.3.0.
- Historical archive candidates may be warm-started, but all fitness is re-evaluated on the current manifest and seed protocol.
- Optional CUDA/Quantum simulator remains a classical statevector simulation, not a physical QPU.

## Test evidence in the build environment

- Compile preflight: PASS.
- Unit tests: 206/206 PASS.
- Integration tests: 12/12 PASS.
- Unit and integration suites ran concurrently as separate processes.
- Release ZIP is separately checked after clean extraction with SHA-256 manifest and version check.

These tests do not qualify native Windows CUDA throughput, Quadro P2000 smoke, hybrid long-run recovery, independent scientific generalization or 24-hour operation. Those must be tested on the target PC.

## Scope not claimed

The built-in candidate model family remains linear/small-MLP plus optimizer/training recipe search. This release does not add CNN/Transformer/GNN NAS, autonomous theorem proving, authenticated publisher provenance, physical QPU execution, quantum advantage, or a proof of universal generalization. Hypothesis tools help structure and audit researcher-supplied claims; they do not autonomously invent or validate scientific truth.
