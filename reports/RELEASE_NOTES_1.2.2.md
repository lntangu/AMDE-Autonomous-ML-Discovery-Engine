# AMDE 1.2.2 — backend-aware CUDA smoke timeout fix

## Problem

AMDE 1.2.1 `smoke_config()` forced both quick and confirm evaluator timeouts to 10 seconds regardless of the requested backend. On the target Windows Quadro P2000 system, the CUDA smoke run timed out while evaluating the `mlp_small` baseline before candidate search began. The quantum doctor had already verified CUDA availability, so this timeout alone did not establish a CUDA backend failure.

## Change

- Explicit CPU smoke keeps its 10-second quick and confirm caps.
- `auto` and explicit `cuda` smoke use a 45-second quick timeout and 60-second confirm timeout.
- The overall smoke time budget remains capped at 120 seconds.
- No search grammar, benchmark, metric, candidate, checkpoint or protocol semantics were changed.
- Added regression tests for CPU, CUDA and auto timeout selection.

## Interpretation

A successful CUDA smoke must still report `device_effective=cuda` and `gpu_effective=true`. A timeout at the new bound remains a failure to diagnose; do not infer CUDA success from GPU detection alone. This release does not establish quantum advantage, generalization or long-run stability.
