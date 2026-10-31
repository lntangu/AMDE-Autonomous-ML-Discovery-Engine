# AMDE 1.2.2 — verification results

## Change under test

Backend-aware smoke timeout policy:
- Explicit CPU smoke: 10-second quick and confirm timeout.
- `auto`/CUDA smoke: 45-second quick timeout and 60-second confirm timeout.
- Overall smoke budget remains capped at 120 seconds.

## Build-environment verification

- Unit tests: **182/182 PASS**.
- Integration tests: **11/11 PASS**, executed in controlled groups (1–6, 8–11, then case 7 independently because the aggregate runner stalled at case 7 in the build environment).
- Wheel build: **PASS** using the locally installed build backend (`--no-build-isolation`).
- Source compilation: verified during the test/build workflow.
- SHA-256 manifest and clean-archive extraction: verified at release packaging.

## Native Windows qualification

- Prior AMDE 1.2.1: Windows unit 179/179, integration 11/11, manifest 177/177 PASS.
- Prior AMDE 1.2.1: Quadro P2000 CUDA Doctor PASS.
- Prior AMDE 1.2.1 CUDA smoke: **FAILED** because `smoke_config()` forced a 10-second baseline-evaluation timeout; this release corrects that timeout policy.
- AMDE 1.2.2 CUDA smoke on the target Windows machine: **NOT YET RUN**.

A successful CUDA smoke must finish and report `device_effective=cuda` and `gpu_effective=true`. Do not treat GPU detection or the CUDA Doctor alone as proof that the AMDE training evaluator completed on CUDA. This release does not establish quantum advantage, generalization or long-run stability.
