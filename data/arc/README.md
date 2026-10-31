# ARC local input (optional)

**Current status: BLOCKED.** No ARC task JSON files are present in this directory, so
`doctor` and every run report:

```text
ARC: BLOCKED — no ARC task JSON files found under ...\data\arc
```

AMDE does not download ARC data, does not access the network, and does not modify an ARC
repository. Only place local files here if you have the rights to use them.

## Accepted shape

The adapter reads `*.json` files recursively under this directory (bounded by file count
and size) and expects one task object per file:

```json
{
  "train": [
    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
  ],
  "test": [
    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
  ]
}
```

Rules enforced by the adapter:

- grids must be non-empty, rectangular, at most 30×30;
- every cell must be an integer from 0 to 9;
- each task needs at least one non-empty `train` and `test` pair;
- invalid files are reported, not silently accepted;
- tasks are split deterministically 70/30 by task hash, never by individual examples.

If valid files are found, `doctor` can report `READY_BASELINE_ONLY`. The adapter can run
real, deliberately weak identity and palette-fill exact-match baselines on the dev tasks.
Candidate search for ARC grids remains unsupported: the current candidate trainer emits
scalar predictions, not 2-D grids. ARC results must not be presented as candidate
discovery or as ARC-solving evidence.

## Licensing and safety

ARC data may have separate license/usage terms. Do not copy data from unknown sources,
do not add a downloader, and do not modify another ARC repository from AMDE. Data files
are ignored by `.gitignore` except this README.
