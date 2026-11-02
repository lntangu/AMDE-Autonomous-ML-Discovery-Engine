# AMDE 4.0 — KNOWN LIMITATIONS & RESEARCH BOUNDARIES
**Distribution Version:** 4.0.1  

---

## 1. Algorithmic Limitations

1. **Short Evaluation Budgets:** Under budgets $\le 50$ evaluations, complex meta-controllers like EGADA do not accumulate enough reward feedback to decisively outperform uniform allocation or random search.
2. **Search Space Expressivity:** The pipeline space currently supports sequential preprocessor + estimator topologies. Arbitrary acyclic directed graph (DAG) multi-branch ensembling is not natively supported.
3. **Multi-Fidelity Ranking Noise:** On small datasets ($N < 500$), 20% screening rungs exhibit high rank-inversion noise relative to full training sets. Full fidelity is recommended for high-precision evaluations.

---

## 2. Infrastructure & Hardware Limitations

1. **Workstation Tuning:** Memory caps (`max_memory_mb = 2048`) and VRAM limits (`max_gpu_memory_mb = 3072`) are calibrated for mobile workstation configurations (e.g. Intel Core i7-8850H with NVIDIA Quadro P2000). Larger distributed clusters require increasing these budget caps.
2. **Windows vs Linux Process Supervision:** Historical AMDE 2 process management tests require native Windows Job Objects; under Linux/WSL, standard POSIX process termination (`SIGTERM` / `SIGKILL`) is used.
3. **Single-Node Concurrency:** Multiprocessing concurrency uses standard library process pools. Distributed cluster backends (Slurm, Ray, Kubernetes) require custom worker executor adapters.
