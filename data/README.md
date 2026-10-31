# Data directory

AMDE does **not** bundle or download datasets. The active benchmark suite is generated
deterministically in memory from task specifications in `config/default.json`; generated
examples are never written to this directory.

- `data/arc/` is an optional input location for local ARC task JSON files.
- The engine never downloads data and never modifies an ARC checkout.
- Place only data you are licensed to use here.

The data and split manifests, including hashes, are written under each run's
`runs/<run_id>/manifests/` directory. The challenge split is generated only inside the
sealed evaluator and is deliberately not persisted.
