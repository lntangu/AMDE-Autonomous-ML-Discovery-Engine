# AMDE 1.3.0 — Open-ended search design

## What “open-ended” means here

The DSL is a typed, allow-listed expression grammar over a finite operator/variable alphabet and numeric constants. The set of all finite expression trees over such a grammar is countably infinite in the mathematical model: larger finite trees create new representable expressions. The runtime does **not** execute arbitrary Python and does not allow imports, I/O, networking, attributes, loops, or user-defined callables.

A single AMDE campaign is necessarily finite. It is bounded by `max_dsl_nodes`, `max_dsl_depth`, `max_genome_bytes`, trial/time budgets, memory limits and storage limits. AMDE 1.2.0 progressively raises the generated-expression frontier with generation, but never exceeds the configured hard safety caps. The default caps are 256 DSL nodes and depth 24; the validator supports configured ceilings up to 512 nodes and depth 64. Multiple campaign configurations can explore larger bounded frontiers, but no finite run enumerates an infinite space.

## DSL v2 additions

DSL v2 retains DSL v1 parsing for old candidates and adds bounded scalar operators: `log_abs`, `exp_clip`, `sin`, `cos`, `sigmoid`, `softplus`, `relu`, `reciprocal`, `erf`, `softsign`, `leaky_relu`, `hypot`, `atan2`, `lerp` and `select_pos`. Numerical evaluation checks finite inputs/outputs and clips only operations explicitly designed to be bounded. Candidates that produce invalid or oversized values are rejected by normal evaluation/error handling.

## Duplicate policy

1. Every accepted DSL v2 expression is normalized into a canonical AST.
2. The normalizer sorts operands for explicitly commutative operations, folds closed constant-only subtrees where safe, and applies a small list of exact identities.
3. Candidate identity uses SHA-256 over canonical genome JSON. The population retains a `seen` hash set across checkpoint/resume for that run, so an evicted candidate is not proposed again in the same run.
4. The novelty ledger retains canonical payloads alongside hashes and checks for a theoretical hash collision.
5. Numeric behavior fingerprints are diagnostic only. They are **not** used to discard a candidate because finite probes cannot prove semantic equivalence.

This removes obvious structural duplicates, not every semantically equivalent program. Complete equivalence checking for an expressive language is not generally decidable; the project does not claim a perfect semantic deduplicator. AMDE 1.3.0 adds a persistent SQLite discovery archive shared across campaigns. It records exact canonical identity, first/last sighting, run/protocol provenance and descriptive score observations. Scores from different protocol manifests are not comparable and are never used to rank cross-protocol warm starts. Prior genomes may be proposed into a new campaign, but their previous scores are never imported as fitness: every warm-start candidate must be re-evaluated under the new campaign's protocol and seeds. The archive has configurable hard caps (100,000 genomes and 1,000,000 sightings by default); at capacity it declines new writes without failing the active campaign. Behaviour fingerprints remain diagnostic and never automatically discard a candidate.

## Search strategies

The normal search portfolio adds `open_ended` beside random, evolutionary, crossover, novelty and local mutation. The generator retains stable recipe priors while creating new DSL expressions, and the proposer retries known hashes. If a configured search region becomes saturated, it fails closed rather than silently evaluating a duplicate.

## Reproducibility

The search and data protocol identifiers were bumped in 1.2.0 because DSL normalization and the proposal distribution changed. Use a **new run/campaign ID** for 1.2.0. DSL v1 genomes remain parseable, but old 1.1.1 checkpoints are not automatically compatible with the new protocol hash. Preserve the old folder and evidence.

## What this does not establish

A larger representation space is not automatically a better search. Search quality still depends on the inductive bias, fitness function, benchmark coverage, compute allocation, candidate validation, and statistical selection. AMDE 1.2.0 does not prove it can discover a universally superior optimizer or generalize to every possible task. Claims must be bounded by independent benchmark families, datasets, domains and a preregistered protocol.

## Optimizer-space expansion (1.3.0)

The typed recipe space now includes AdaGrad and NAdam in addition to SGD, momentum, RMSProp, Adam, AdamW, Lion and DSL update rules. Reference and PyTorch trainers implement the same defined update equations and are covered by CPU backend parity tests. The fixed baseline set includes linear AdaGrad and linear NAdam; smoke mode remains smaller to stay within its time budget. This expands optimizer discovery but does not turn the model architecture space into arbitrary neural architecture search.


## AMDE 1.3.0 cross-campaign memory

Successful candidate observations are also appended to `storage/discovery_archive.sqlite3`. Exact canonical genome hashes are authoritative; optional behaviour signatures are diagnostic and do not reject candidates. Historical scores are not comparable across protocols and are not transferred into new campaigns. A small number of recent genomes may be proposed as warm starts, but each must be re-evaluated under the receiving campaign's own manifest, seeds and protocol hash.

The optimizer schema includes AdaGrad and NAdam in addition to the 1.2.x set. Behaviour descriptors normalize the optimizer coordinate over the complete registered optimizer vocabulary.
