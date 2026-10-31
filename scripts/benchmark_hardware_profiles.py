#!/usr/bin/env python3
"""Rigorous empirical CUDA event benchmark and profile reconciliation for AMDE 2.0.2.

Precisely breaks down:
- Pure CUDA forward kernel time (ms)
- Pure CUDA backward backprop kernel time (ms)
- Pure CUDA optimizer step time (ms)
- Total CUDA active execution time (ms)
- CPU dispatch and Python orchestration latency (ms)
- Total end-to-end step time (ms)
- Measured GPU duty cycle (%)
- Peak VRAM allocation and reservation (MB)
- Throughput in valid completed trials per hour
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.quantum.profiles import PROFILE_BASELINE, PROFILE_MEDIUM, PROFILE_HIGH, AdaptiveComputeGovernor
from amde.quantum.vqc import VQCConfig, VariationalQuantumClassifier, train_vqc
from amde.quantum.statevector import load_torch, resolve_device


def get_gpu_temperature() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raw = completed.stdout.strip().splitlines()[0].strip()
            return float(raw)
    except Exception:
        pass
    return None


def run_cuda_event_profile(
    *,
    n_qubits: int,
    n_layers: int,
    batch_size: int,
    device: str = "cuda",
    warmup_steps: int = 10,
    measured_steps: int = 50,
) -> dict[str, Any]:
    torch = load_torch()
    actual_device = resolve_device(device)
    if actual_device != "cuda":
        raise RuntimeError("CUDA event profiling requires a functional CUDA device")

    config = VQCConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        batch_size=batch_size,
        epochs=10,
        gradient_method="autograd",
        optimizer="adam",
    )
    model = VariationalQuantumClassifier(config, device=actual_device, seed=42)
    optimizer = torch.optim.Adam([model.theta], lr=0.01)

    # Pre-generate synthetic features on device
    features = torch.clamp(torch.randn((batch_size, n_qubits), device=actual_device), -1.0, 1.0)
    targets = (torch.rand(batch_size, device=actual_device) >= 0.5).float()

    # Warmup runs
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Create CUDA timing events
    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end_event = torch.cuda.Event(enable_timing=True)
    bwd_end_event = torch.cuda.Event(enable_timing=True)
    opt_end_event = torch.cuda.Event(enable_timing=True)

    fwd_times: list[float] = []
    bwd_times: list[float] = []
    opt_times: list[float] = []
    e2e_times: list[float] = []

    torch.cuda.reset_peak_memory_stats()
    temp_before = get_gpu_temperature()

    for _ in range(measured_steps):
        t_cpu_start = time.perf_counter()

        start_event.record()
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        fwd_end_event.record()

        loss.backward()
        bwd_end_event.record()

        optimizer.step()
        opt_end_event.record()

        opt_end_event.synchronize()
        t_cpu_end = time.perf_counter()

        fwd_ms = start_event.elapsed_time(fwd_end_event)
        bwd_ms = fwd_end_event.elapsed_time(bwd_end_event)
        opt_ms = bwd_end_event.elapsed_time(opt_end_event)
        e2e_ms = (t_cpu_end - t_cpu_start) * 1000.0

        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        opt_times.append(opt_ms)
        e2e_times.append(e2e_ms)

    temp_after = get_gpu_temperature()
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    def mean_std(vals: list[float]) -> tuple[float, float]:
        m = sum(vals) / len(vals)
        variance = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return round(m, 3), round(math.sqrt(variance), 3)

    m_fwd, s_fwd = mean_std(fwd_times)
    m_bwd, s_bwd = mean_std(bwd_times)
    m_opt, s_opt = mean_std(opt_times)
    m_e2e, s_e2e = mean_std(e2e_times)
    total_cuda_ms = round(m_fwd + m_bwd + m_opt, 3)
    cpu_dispatch_ms = round(max(0.0, m_e2e - total_cuda_ms), 3)
    duty_cycle = round(100.0 * (total_cuda_ms / max(0.001, m_e2e)), 2)

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "forward_cuda_ms": {"mean": m_fwd, "std": s_fwd},
        "backward_cuda_ms": {"mean": m_bwd, "std": s_bwd},
        "optimizer_cuda_ms": {"mean": m_opt, "std": s_opt},
        "total_cuda_kernel_ms": total_cuda_ms,
        "cpu_dispatch_ms": cpu_dispatch_ms,
        "end_to_end_step_ms": {"mean": m_e2e, "std": s_e2e},
        "measured_gpu_duty_cycle_percent": duty_cycle,
        "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 2),
        "temperature_before_c": temp_before,
        "temperature_after_c": temp_after,
    }


def benchmark_end_to_end_trial(profile_name: str, prof: Any, device: str = "cuda") -> dict[str, Any]:
    torch = load_torch()
    actual_device = resolve_device(device)
    config = VQCConfig(
        n_qubits=prof.n_qubits,
        n_layers=prof.n_layers,
        epochs=prof.epochs,
        batch_size=prof.batch_size,
    )
    # Generate matched challenge dataset
    rng = torch.Generator(device="cpu").manual_seed(12345)
    train_x = torch.randn((prof.train_size, prof.n_qubits), generator=rng).tolist()
    train_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in train_x]
    val_x = torch.randn((prof.val_size, prof.n_qubits), generator=rng).tolist()
    val_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in val_x]
    test_x = torch.randn((prof.test_size, prof.n_qubits), generator=rng).tolist()
    test_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in test_x]

    temp_start = get_gpu_temperature()
    t0 = time.perf_counter()
    result = train_vqc(
        train_x, train_y, val_x, val_y, test_x, test_y,
        config=config, device=actual_device, seed=42
    )
    elapsed = time.perf_counter() - t0
    temp_end = get_gpu_temperature()

    trials_per_hour = round(3600.0 / elapsed, 1)

    return {
        "profile": profile_name,
        "protocol_id": prof.protocol_id,
        "n_qubits": prof.n_qubits,
        "batch_size": prof.batch_size,
        "epochs": prof.epochs,
        "train_size": prof.train_size,
        "elapsed_seconds": round(elapsed, 3),
        "val_balanced_accuracy": round(result.get("val_balanced_accuracy", 0.0), 4),
        "test_balanced_accuracy": round(result.get("test", {}).get("balanced_accuracy", 0.0), 4),
        "peak_vram_mb": round(result.get("peak_gpu_allocated_bytes", 0) / (1024**2), 2),
        "measured_trials_per_hour": trials_per_hour,
        "temperature_start_c": temp_start,
        "temperature_end_c": temp_end,
    }


def main() -> int:
    print("=" * 70)
    print("AMDE 2.0.2 EMPIRICAL HARDWARE BENCHMARK & RECONCILIATION")
    print(f"Platform: {sys.platform} | Python: {sys.version.split()[0]}")
    torch = load_torch()
    print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)} (CC {torch.cuda.get_device_capability(0)})")
    print("=" * 70)

    profiles = {
        "BASELINE": PROFILE_BASELINE,
        "MEDIUM": PROFILE_MEDIUM,
        "HIGH": PROFILE_HIGH,
    }

    step_results = {}
    print("\n--- 1. PURE CUDA EVENT STEP-LEVEL BENCHMARK ---")
    for name, prof in profiles.items():
        print(f"\nProfiling {name} (N={prof.n_qubits}, batch={prof.batch_size}, layers={prof.n_layers})...")
        res = run_cuda_event_profile(
            n_qubits=prof.n_qubits,
            n_layers=prof.n_layers,
            batch_size=prof.batch_size,
            device="cuda",
            warmup_steps=10,
            measured_steps=50,
        )
        step_results[name] = res
        print(f"  Forward CUDA:   {res['forward_cuda_ms']['mean']:.3f} +/- {res['forward_cuda_ms']['std']:.3f} ms")
        print(f"  Backward CUDA:  {res['backward_cuda_ms']['mean']:.3f} +/- {res['backward_cuda_ms']['std']:.3f} ms")
        print(f"  Optimizer CUDA: {res['optimizer_cuda_ms']['mean']:.3f} +/- {res['optimizer_cuda_ms']['std']:.3f} ms")
        print(f"  Total CUDA:     {res['total_cuda_kernel_ms']:.3f} ms")
        print(f"  CPU Dispatch:   {res['cpu_dispatch_ms']:.3f} ms")
        print(f"  End-to-End:     {res['end_to_end_step_ms']['mean']:.3f} +/- {res['end_to_end_step_ms']['std']:.3f} ms")
        print(f"  GPU Duty Cycle: {res['measured_gpu_duty_cycle_percent']:.2f}%")
        print(f"  Peak VRAM:      {res['peak_vram_allocated_mb']} MB (Alloc) / {res['peak_vram_reserved_mb']} MB (Reserved)")

    trial_results = {}
    print("\n--- 2. END-TO-END TRIAL EXECUTION BENCHMARK ---")
    for name, prof in profiles.items():
        print(f"\nExecuting trial {name}...")
        t_res = benchmark_end_to_end_trial(name, prof, device="cuda")
        trial_results[name] = t_res
        print(f"  Elapsed:        {t_res['elapsed_seconds']} s")
        print(f"  Throughput:     {t_res['measured_trials_per_hour']} trials/hour")
        print(f"  Val BAcc:       {t_res['val_balanced_accuracy']}")
        print(f"  Test BAcc:      {t_res['test_balanced_accuracy']}")
        print(f"  Peak VRAM:      {t_res['peak_vram_mb']} MB")
        print(f"  Temperature:    {t_res['temperature_start_c']}C -> {t_res['temperature_end_c']}C")

    # Output JSON summary
    report = {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "step_level_cuda_events": step_results,
        "end_to_end_trials": trial_results,
    }
    out_path = ROOT / "BENCHMARK_PROFILES_RECONCILIATION.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved empirical benchmark report to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
