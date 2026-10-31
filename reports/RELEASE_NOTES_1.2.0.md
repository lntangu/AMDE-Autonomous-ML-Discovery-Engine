# AMDE 1.2.0 — Research-search and validation release

## Added

- DSL v2 with a larger allow-listed scalar expression vocabulary while retaining DSL v1 parsing for legacy candidate payloads.
- Progressively expanding open-ended proposal grammar and an `open_ended` search strategy. The theoretical grammar has countably many finite expressions; each campaign is still bounded by explicit node/depth/bytes/time/memory/storage limits.
- Canonical AST normalization for obvious structural duplicates, SHA-256 genome identity, persistent per-run `seen` hashes through checkpoints, and diagnostic behavior fingerprints that are never treated as proof of semantic equivalence.
- Family-clustered generalization audit with bootstrap confidence intervals, paired sign-flip inference, family robustness gates, provenance metadata and scoped evidence tiers.
- External CSV benchmark runner for paired frozen-candidate vs fixed-baseline evaluation, strict numeric data validation, duplicate-grouped splits, data hashes, per-family/per-seed records and automatic generalization audit.
- Synthetic demo manifest/candidate files and a documented external-dataset schema.
- Detached hybrid controller: `start`, `status`, `stop` and `logs`, so PowerShell users can safely manage a background supervisor without interrupting a `.bat` job with Ctrl+C.
- Regression coverage for cooperative `stopped` worker exit and detached start/status/stop lifecycle.

## Fixed

- Classical campaign return code now treats `stopped` as a successful cooperative terminal state.
- Hybrid supervisor treats a worker state of `stopped` as a cooperative stop instead of counting it as a transient crash and scheduling a restart.
- Random DSL generation enforces the exact node budget across every child branch.
- Generalization audit rejects invalid statistical thresholds and excessive bootstrap settings.
- External CSV ingestion rejects malformed/non-finite/missing values, conflicting duplicate-feature labels, mixed task kinds in one manifest, and files/datasets beyond configured resource limits.

## Protocol compatibility

- Engine version: `1.2.0`. Search RNG and data protocol identifiers were bumped.
- DSL v1 payloads remain readable, but 1.1.1 checkpoints must not be resumed under the new protocol as though the search distribution were unchanged. Use a new campaign ID and preserve the old project folder.

## Scientific limits

- The theoretical finite-expression grammar is countably infinite, but a concrete campaign is finite and capped. No campaign can exhaust an infinite set.
- Canonicalization eliminates only a declared subset of syntactic duplicates. Behavioral fingerprints are diagnostic and cannot prove equivalence.
- The generalization audit only assesses supplied, paired family-level results and metadata. It does not supply external benchmarks or independently authenticate provenance.
- Verification on the build environment: 177/177 unit tests passed; 11/11 integration cases passed (cases 1–4 and 5–10 in bounded slices, case 11 via a separate direct invocation); source compileall passed; synthetic external-evaluation CLI smoke passed and correctly failed the confirmatory generalization gates. The aggregate integration runner was split because the single long invocation intermittently stalled in this environment; the final combined 5–11 invocation timed out at case 11, which then passed independently in 3.06 seconds.
- No universal generalization theorem, quantum advantage, new optimizer discovery, Windows CUDA qualification or 24-hour stability claim is made by this release. Those require separate empirical evidence.
