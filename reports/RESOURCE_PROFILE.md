# AMDE — Historical Resource Profile (hardware observed before 1.1.1)

## Intended target profile

Previously reported user hardware: Windows 11, Intel Core i7-8850H (6 physical / 12 logical cores), 32 GB RAM, NVIDIA Quadro P2000 4 GB VRAM, PyTorch 2.14.0+cu126, driver 582.78. These earlier hardware readings came from AMDE 0.2.5/0.2.6, not a 1.1.0 target-machine qualification.

## Hybrid allocation

- Classical AMDE campaign: CPU backend; the established engine controls its evaluator worker pool and resource reserves.
- Quantum research campaign: one process; PyTorch CPU thread count is limited to one. `--quantum-device cuda` requests CUDA for statevector, VQC and quantum kernel. It does not silently fallback to CPU if explicit CUDA fails.
- The quantum branch runs small CPU MLP/RBF baselines while the classical branch is active, so CPU contention is expected. This is controlled concurrency, not complete hardware isolation.
- CUDA jobs wait if free VRAM drops below 1536 MiB or `nvidia-smi` reports temperature >=80°C. The worker pauses up to 30 minutes and then fails closed; it does not blindly allocate through pressure.
- Statevector memory is bounded by `--quantum-statevector-mb`. Temporary tensors make actual peak use larger than one bare statevector.
- Default VQC architectures use 4, 6 and 8 qubits; max config allows up to 10 (VQC module supports up to 12), subject to memory constraints. The Grover maze demo is separately capped.
- Campaign caps default to 12 GiB classical + 10 GiB quantum with 5 GiB minimum free disk in the documented hybrid command.

The target laptop does not need 100% GPU utilisation. Small VQCs may be launch/synchronisation bound. Compare completed valid experiment throughput and quality per wall-clock/energy, not just GPU percentage.
