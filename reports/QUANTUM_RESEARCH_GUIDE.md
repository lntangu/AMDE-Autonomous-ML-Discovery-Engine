# AMDE 1.1.0 — Quantum and Hybrid Research Guide

## 1. What is implemented

| Component | Implementation | Execution | Evidence boundary |
|---|---|---|---|
| Statevector | Exact complex64 statevector, H/RY/RZ/CNOT, Z expectation, ideal Grover diffusion | PyTorch CPU or CUDA | Classical simulation; state size grows as `2^n` |
| VQC | Angle-encoded variational circuit with linear/ring entanglement | PyTorch CPU or CUDA | Simulator result, not physical QPU |
| Gradient | Autograd and exact two-point parameter-shift for the Pauli rotation gates used | PyTorch | Parameter-shift parity test is a software correctness check |
| Quantum kernel | Statevector feature map and fidelity kernel-ridge evaluation | CPU/CUDA | Compared with CPU RBF kernel-ridge baseline |
| Classical baselines | MLP and RBF kernel-ridge | CPU | Same generated split/seed as quantum candidates |
| Maze illustration | BFS shortest path vs ideal Grover query model | CPU/CUDA simulator | Oracle is table-backed; precomputation is excluded from Grover query count and separately timed |
| Hybrid supervisor | Classical AMDE campaign + quantum campaign as separate subprocesses | CPU + requested CUDA/CPU | Supervisor lifecycle tested with worker stubs; target hardware still needs verification |

This release does **not** include a physical QPU provider, hardware noise-aware compilation, error mitigation, a tensor-network simulator, or a quantum-inspired optimizer as a separate search backend. Those are potential future research modules, not current capabilities.

## 2. Statevector memory

The statevector stores `2^n` complex64 amplitudes (8 bytes per amplitude, before temporary/intermediate tensors). Approximate bare-state memory:

| Qubits | Bare statevector |
|---:|---:|
| 20 | 8 MiB |
| 24 | 128 MiB |
| 25 | 256 MiB |
| 26 | 512 MiB |
| 27 | 1 GiB |
| 28 | 2 GiB |
| 29 | 4 GiB |

Batched circuits multiply this requirement by the batch size. Actual runtime memory is higher because operations may allocate additional tensors. Do not set the statevector cap equal to all VRAM. The quantum campaign defaults to small circuit families (4/6/8 qubits) and reserves at least 1536 MiB free VRAM before CUDA jobs. The 4 GB Quadro P2000 should start with the defaults; increase qubits only after measured memory/latency tests.

## 3. Maze and Grover

Grover's algorithm provides a quadratic query-complexity improvement for unstructured search when a suitable oracle can mark valid candidates. That does not automatically give a faster end-to-end maze solver:

- A maze is a graph problem. BFS/A* exploit connectivity and path structure.
- The included experiment enumerates fixed-length move strings classically to construct a table-backed marked-state oracle.
- The simulator applies the phase mark directly; the table is not a reversible gate-level oracle circuit.
- The reported Grover iteration/query count omits oracle construction and is not a wall-clock speedup.
- Statevector simulation itself performs classical work that scales with the full statevector.

The demo is an educational query-model experiment, not a performance benchmark proving quantum superiority.

## 4. Scientific design

Each quantum campaign round derives independent deterministic seeds for train, validation and test splits. Shifted-test tasks are marked in the CSV leaderboard. Candidate configurations may be archived based on validation evidence as exploration provenance, but old scores are not used as current-round evidence. Each candidate is re-evaluated on the new round.

Compare VQC and quantum-kernel results against MLP and RBF baselines under the same data splits. For any future real-world experiment, include:

- preprocessing and data-encoding costs;
- trainable parameter count, circuit depth and two-qubit gate count;
- number of circuit evaluations/shots and optimizer steps;
- wall-clock and energy/cost if hardware is used;
- hardware noise, readout error and calibration conditions;
- multiple dataset/task families, external holdouts and repeated seeds;
- tuning budgets matched between quantum and classical methods.

Do not select a result only because it carries a quantum label. Promotion requires independent, repeated, adequately powered evidence.

## 5. Quantum preflight

```powershell
scripts\quantum_doctor.bat --device cuda
```

The doctor checks PyTorch import, CUDA availability, free VRAM and a four-qubit statevector gate smoke. `PASS` only confirms the small simulator path. It does not establish the VQC campaign throughput or quantum advantage.

To run the maze illustration:

```powershell
scripts\quantum_maze_demo.bat --device cpu --max-qubits 20 --max-statevector-mb 256
```

## 6. Finite QML pilot

```powershell
scripts\quantum_research_campaign.bat --campaign-id qml_pilot_001 --rounds 1 --seed 42 --device cuda --epochs 4 --batch-size 16 --train-size 48 --val-size 24 --test-size 48 --max-configs 3 --max-qubits 8 --max-statevector-mb 256 --max-campaign-gb 2 --min-free-gb 5
```

Review `reports\quantum_campaigns\qml_pilot_001\REPORT.md`, `leaderboard.csv`, `state.json` and `quantum_ledger.sqlite3`. A partial/failed round should be investigated before restarting with a new ID or the same immutable plan.

## 7. Hybrid CPU + CUDA campaign

After the separate CPU and CUDA pilots pass, use:

```powershell
scripts\hybrid_research_campaign.bat --campaign-id amde_hybrid_001 --hours-per-round 1 --max-trials-per-round 400 --classical-max-gb 12 --quantum-max-gb 10 --min-free-gb 5 --seed 42 --quantum-device cuda --quantum-epochs 12 --quantum-batch-size 32 --quantum-train-size 96 --quantum-val-size 48 --quantum-test-size 96 --quantum-max-configs 5 --quantum-max-qubits 10 --quantum-statevector-mb 256 --max-restarts 5
```

The supervisor launches the classical campaign with CPU and the quantum branch with the selected device. Both use independent state/ledger/locks and round IDs. `--rounds 0` is internal and means each branch continues creating rounds until stopped or storage/disk limits are reached.

Stop both branches:

```powershell
scripts\stop_hybrid.bat
```

Inspect `reports\hybrid_campaigns\amde_hybrid_001\supervisor_state.json` and both worker logs. Wait until the supervisor reports `stopped`, `resource_paused` or `circuit_breaker` and both child locks are released. For an intentional restart of the exact same plan, re-run the exact command with `--clear-stop`, after checking that no previous child PID remains. `--clear-stop` removes `runs\HYBRID_STOP`, `runs\STOP` and `runs\QUANTUM_STOP`; it can stop/resume other AMDE processes sharing this project root, so do not use it while unrelated runs are active.

The supervisor uses exponential backoff and a bounded restart counter. When failures exceed `--max-restarts`, it stops both branches and records `circuit_breaker`; it does not loop forever on a crashing process. If a branch reaches a storage cap or low-disk threshold, the supervisor stops both branches and records `resource_paused`.

## 8. Long-duration limits

A campaign may run across many finite rounds, but 24/7 operation still depends on Windows updates, power, thermal management, disk/SSD health, drivers, backups and human review. Keep adequate free disk, monitor temperatures/RAM/VRAM, inspect worker logs, and archive verified ledgers/checkpoints before changing storage limits. The design aims for resumability and safe stopping; it does not guarantee perpetual operation or discovery.
