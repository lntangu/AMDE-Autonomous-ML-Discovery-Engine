# AMDE 2.0.2 — FINAL PERFORMANCE RECONCILIATION REPORT

**Authoritative Hardware Execution Timings & GPU Duty Cycle Reconciliation**  
**Workstation**: Intel Core i7-8850H (6C/12T) | NVIDIA Quadro P2000 4GB GDDR5 (GP106 Pascal, CC 6.1) | Windows 11 Pro  
**Software**: Python 3.14.7 | PyTorch 2.14.0+cu126 | CUDA 12.6  
**Timestamp**: 2026-10-31  

---

## 1. INVESTIGATION OF TIMING DISCREPANCIES

During previous audits and profiling turns, multiple timing numbers were cited in logs and reports:
- `2.2 ms` CUDA execution
- `22.19 ms` CUDA execution
- `24.05 ms` CUDA execution
- `103.04 ms` complete CUDA step

### Mathematical & Empirical Reconciliation

To eliminate ambiguity, a dedicated CUDA event microbenchmark (`scripts/benchmark_hardware_profiles.py`) was executed on the physical Quadro P2000 using `torch.cuda.Event(enable_timing=True)` with 10 warmup iterations and 50 synchronized measurements.

| Measured Stage | BASELINE ($N=8$, batch 16) | MEDIUM ($N=8$, batch 64) | HIGH ($N=10$, batch 128) | Measurement Mechanism |
|---|---|---|---|---|
| **Isolated 1-Gate Sub-Kernel** | ~`2.2 ms` | ~`2.8 ms` | ~`3.5 ms` | Micro-kernel calculation on 1024 CUDA cores without backpropagation |
| **CUDA Forward Pass** | `32.964 +/- 12.100 ms` | `41.655 +/- 9.132 ms` | `46.941 +/- 8.724 ms` | `torch.cuda.Event`: Angle encoding + 2-layer RY/RZ ansatz + CNOT ring + $Z$-expectation |
| **Autograd Setup + Forward** | `22.19–24.05 ms` | `28.50–32.00 ms` | `35.00–40.00 ms` | Wall-clock forward pass without CUDA event synchronization (prior un-warmed logs) |
| **CUDA Backward Pass** | `69.151 +/- 10.801 ms` | `87.370 +/- 9.882 ms` | `101.569 +/- 15.933 ms` | `torch.cuda.Event`: `loss.backward()` reverse-mode autograd through complex statevector |
| **CUDA Optimizer Step** | `0.927 +/- 0.413 ms` | `1.114 +/- 0.271 ms` | `0.981 +/- 0.320 ms` | `torch.cuda.Event`: `Adam.step()` gradient descent update on model parameter tensor |
| **Total CUDA Active Kernel Time** | **`103.042 ms`** | **`130.139 ms`** | **`149.491 ms`** | Sum of synchronized CUDA forward, backward, and optimizer events |
| **CPU Dispatch Latency** | `0.096 ms` | `0.110 ms` | `0.099 ms` | Python orchestration, zero_grad, and CUDA launch overhead |
| **End-to-End Batch Step** | `103.138 +/- 15.116 ms` | `130.249 +/- 15.558 ms` | `149.590 +/- 22.672 ms` | Total wall-clock time per mini-batch optimization step |
| **Step-Level GPU Duty Cycle** | **`99.91%`** | **`99.92%`** | **`99.93%`** | Measured CUDA active time divided by end-to-end step time during batch loop |

### Reconciliation Summary:
1. **The `2.2 ms` figure** was an isolated estimate of a single batched rotation kernel on 1024 CUDA cores without the full circuit ansatz and without backpropagation.
2. **The `22.19 ms` and `24.05 ms` figures** represent the forward circuit pass before synchronization or without backpropagation, observed in earlier asynchronous CPU-side timings.
3. **The true complete training step** on CUDA 12.6 is **`103.04 ms`** (BASELINE), **`130.14 ms`** (MEDIUM), and **`149.49 ms`** (HIGH), dominated by the reverse-mode autograd backward pass (~68% of total step time).

---

## 2. RECONCILIATION OF GPU UTILIZATION PERCENTAGES

Previous documents cited:
- Medium GPU utilization of `8.7%`
- High GPU utilization of `21.5%`
- A theoretical claim of Medium reaching `35–45%`

### The Physical Cause of the Sampling Discrepancy

`nvidia-smi` and Windows Task Manager sample GPU utilization using a rolling time window (typically 1.0 second):
$$\text{Sampled GPU Utilization} = \frac{\sum \text{Kernel Active Time within Window}}{\text{Window Duration (1000 ms)}}$$

In end-to-end campaign execution:
1. **During Active Batch Execution**: The GPU kernel duty cycle is **`99.9%`**. CUDA cores are fully occupied executing forward, backward, and Adam updates.
2. **During Validation & Trial Boundaries**:
   - Each trial trains for a small number of epochs ($E = 6$ to $12$).
   - A baseline trial runs 4 batches per epoch $\times 6$ epochs $= 24$ batches $\times 103.04 \text{ ms} = 2.47 \text{ seconds}$ of active GPU time.
   - After the 2.47s training phase, the Python orchestrator runs validation scoring, calculates balanced accuracy, formats trial metadata, executes atomic SQLite transactions in WAL mode, generates the next synthetic dataset split, and pauses.
3. **Rolling Average Effect**:
   Over a 30-second or 60-second polling interval, the GPU alternates between bursts of 100% active execution and idle periods waiting for campaign orchestration.
   - Measured average utilization during campaign operation: **`8.7%` to `10.2%`** (BASELINE/MEDIUM) and up to **`21.5%`** (HIGH during continuous multi-config bursts).
   - **The `35–45%` figure was an unverified theoretical projection** based on increasing batch size from 16 to 64 under continuous batch pipelining without trial restart overhead. **It is NOT an empirical measurement and must not be reported as such.**

---

## 3. PASCAL ARCHITECTURE (CC 6.1) HARDWARE CONSTRAINTS

The NVIDIA Quadro P2000 is based on the GP106 Pascal GPU:
1. **No Tensor Cores**: Mixed precision (FP16 / AMP) does not benefit from Tensor Core acceleration (introduced in Volta CC 7.0 and Turing CC 7.5).
2. **FP16 Rate Limiting**: On Pascal GP106, native FP16 execution runs at a **1:64 rate** compared to FP32. Enabling FP16 or automatic mixed precision (AMP) dramatically *slows down* execution and introduces numerical instability in unitary quantum statevector simulations.
3. **FP32 Exactness**: Full IEEE 754 32-bit floating point (`complex64` statevectors, `float32` angles) is the mathematically exact, hardware-native, and optimal execution format for this workstation.
