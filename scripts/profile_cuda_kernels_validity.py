#!/usr/bin/env python3
"""Comprehensive CUDA Activity Profiling, Training Step Consistency & Statistical Audit.

AMDE 2.0.2 Scientific Performance Audit:
1. PyTorch Profiler (CPU & CUDA Activity) to separate:
   - CUDA stream elapsed time
   - Individual kernel execution time (GPU active self-time)
   - Kernel launch gaps & CPU dispatcher latency
   - Forward, backward, and optimizer GPU kernel execution times
   - True hardware GPU occupancy
2. Training Step Consistency & Early Stopping Analysis:
   - Requested epochs vs completed epochs
   - Expected steps vs actual steps executed
   - Mathematical reconciliation of trial duration
3. Paired-Seed Statistical Comparison on Validation Data:
   - 10 paired seeds
   - Mean, std, 95% CI, Cohen's d, paired t-test, and Holm-Bonferroni correction
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.quantum.profiles import PROFILE_BASELINE, PROFILE_MEDIUM, PROFILE_HIGH
from amde.quantum.statevector import load_torch, resolve_device
from amde.quantum.vqc import VQCConfig, VariationalQuantumClassifier, train_vqc


def run_pytorch_profiler_breakdown(
    *,
    n_qubits: int,
    n_layers: int,
    batch_size: int,
    device: str = "cuda",
    warmup_steps: int = 5,
    profile_steps: int = 10,
) -> dict[str, Any]:
    torch = load_torch()
    actual_device = resolve_device(device)
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

    features = torch.clamp(torch.randn((batch_size, n_qubits), device=actual_device), -1.0, 1.0)
    targets = (torch.rand(batch_size, device=actual_device) >= 0.5).float()

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Stream event timing
    start_event = torch.cuda.Event(enable_timing=True)
    fwd_event = torch.cuda.Event(enable_timing=True)
    bwd_event = torch.cuda.Event(enable_timing=True)
    opt_event = torch.cuda.Event(enable_timing=True)

    stream_fwd_times: list[float] = []
    stream_bwd_times: list[float] = []
    stream_opt_times: list[float] = []
    wall_step_times: list[float] = []

    for _ in range(profile_steps):
        t0 = time.perf_counter()
        start_event.record()
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        fwd_event.record()

        loss.backward()
        bwd_event.record()

        optimizer.step()
        opt_event.record()
        opt_event.synchronize()
        t1 = time.perf_counter()

        stream_fwd_times.append(start_event.elapsed_time(fwd_event))
        stream_bwd_times.append(fwd_event.elapsed_time(bwd_event))
        stream_opt_times.append(bwd_event.elapsed_time(opt_event))
        wall_step_times.append((t1 - t0) * 1000.0)

    # PyTorch Profiler detailed kernel timing
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    key_averages = prof.key_averages()
    total_cpu_time_us = sum(evt.cpu_time_total for evt in key_averages) / 5.0
    total_cuda_time_us = sum(getattr(evt, "device_time_total", getattr(evt, "cuda_time", 0.0)) for evt in key_averages) / 5.0

    # Top CUDA kernels
    cuda_events = [evt for evt in key_averages if getattr(evt, "device_time_total", getattr(evt, "cuda_time", 0.0)) > 0]
    cuda_events.sort(key=lambda x: getattr(x, "device_time_total", getattr(x, "cuda_time", 0.0)), reverse=True)
    top_kernels = [
        {
            "name": evt.key,
            "cuda_time_us": round(getattr(evt, "device_time_total", getattr(evt, "cuda_time", 0.0)) / 5.0, 1),
            "cpu_time_us": round(evt.cpu_time_total / 5.0, 1),
            "count": max(1, evt.count // 5),
        }
        for evt in cuda_events[:8]
    ]

    mean_fwd_stream_ms = sum(stream_fwd_times) / len(stream_fwd_times)
    mean_bwd_stream_ms = sum(stream_bwd_times) / len(stream_bwd_times)
    mean_opt_stream_ms = sum(stream_opt_times) / len(stream_opt_times)
    total_stream_ms = mean_fwd_stream_ms + mean_bwd_stream_ms + mean_opt_stream_ms
    mean_wall_step_ms = sum(wall_step_times) / len(wall_step_times)
    pure_kernel_gpu_ms = total_cuda_time_us / 1000.0
    kernel_launch_gaps_ms = max(0.0, total_stream_ms - pure_kernel_gpu_ms)
    cpu_dispatch_ms = max(0.0, mean_wall_step_ms - total_stream_ms)
    actual_hardware_utilization = round(100.0 * (pure_kernel_gpu_ms / max(0.001, mean_wall_step_ms)), 2)

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "stream_forward_ms": round(mean_fwd_stream_ms, 3),
        "stream_backward_ms": round(mean_bwd_stream_ms, 3),
        "stream_optimizer_ms": round(mean_opt_stream_ms, 3),
        "total_stream_elapsed_ms": round(total_stream_ms, 3),
        "pure_kernel_gpu_time_ms": round(pure_kernel_gpu_ms, 3),
        "kernel_launch_gaps_ms": round(kernel_launch_gaps_ms, 3),
        "cpu_dispatch_ms": round(cpu_dispatch_ms, 3),
        "wall_step_ms": round(mean_wall_step_ms, 3),
        "actual_hardware_utilization_percent": actual_hardware_utilization,
        "top_cuda_kernels": top_kernels,
    }


def analyze_training_step_consistency(profile_name: str, prof: Any, device: str = "cuda") -> dict[str, Any]:
    torch = load_torch()
    actual_device = resolve_device(device)
    config = VQCConfig(
        n_qubits=prof.n_qubits,
        n_layers=prof.n_layers,
        epochs=prof.epochs,
        batch_size=prof.batch_size,
    )
    rng = torch.Generator(device="cpu").manual_seed(12345)
    train_x = torch.randn((prof.train_size, prof.n_qubits), generator=rng).tolist()
    train_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in train_x]
    val_x = torch.randn((prof.val_size, prof.n_qubits), generator=rng).tolist()
    val_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in val_x]
    test_x = torch.randn((prof.test_size, prof.n_qubits), generator=rng).tolist()
    test_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in test_x]

    steps_per_epoch = math.ceil(prof.train_size / prof.batch_size)
    expected_full_steps = steps_per_epoch * prof.epochs

    t0 = time.perf_counter()
    result = train_vqc(
        train_x, train_y, val_x, val_y, test_x, test_y,
        config=config, device=actual_device, seed=42, patience=5
    )
    total_trial_sec = time.perf_counter() - t0

    epochs_completed = result["epochs_completed"]
    actual_steps_executed = steps_per_epoch * epochs_completed
    early_stopped = epochs_completed < prof.epochs

    return {
        "profile": profile_name,
        "train_size": prof.train_size,
        "batch_size": prof.batch_size,
        "epochs_requested": prof.epochs,
        "steps_per_epoch": steps_per_epoch,
        "expected_full_steps": expected_full_steps,
        "epochs_completed": epochs_completed,
        "actual_steps_executed": actual_steps_executed,
        "early_stopped": early_stopped,
        "total_trial_seconds": round(total_trial_sec, 3),
        "empirical_seconds_per_step": round(total_trial_sec / max(1, actual_steps_executed), 4),
    }


def paired_seed_statistical_comparison(device: str = "cuda", n_seeds: int = 10) -> dict[str, Any]:
    torch = load_torch()
    actual_device = resolve_device(device)
    seeds = [1000 + i * 37 for i in range(n_seeds)]

    results: dict[str, list[float]] = {"BASELINE": [], "MEDIUM": [], "HIGH": []}
    profiles = {
        "BASELINE": PROFILE_BASELINE,
        "MEDIUM": PROFILE_MEDIUM,
        "HIGH": PROFILE_HIGH,
    }

    print(f"Executing {n_seeds} paired seeds on validation split across profiles...")
    for s in seeds:
        rng = torch.Generator(device="cpu").manual_seed(s)
        train_x = torch.randn((128, 8), generator=rng).tolist()
        train_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in train_x]
        val_x = torch.randn((64, 8), generator=rng).tolist()
        val_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in val_x]
        test_x = torch.randn((64, 8), generator=rng).tolist()
        test_y = [1.0 if sum(r[:2]) > 0 else 0.0 for r in test_x]

        for name, prof in profiles.items():
            pad_qubits = prof.n_qubits
            tx = [r + [0.0] * (pad_qubits - len(r)) for r in train_x]
            vx = [r + [0.0] * (pad_qubits - len(r)) for r in val_x]
            dx = [r + [0.0] * (pad_qubits - len(r)) for r in test_x]
            cfg = VQCConfig(
                n_qubits=prof.n_qubits,
                n_layers=prof.n_layers,
                epochs=prof.epochs,
                batch_size=prof.batch_size,
            )
            res = train_vqc(tx, train_y, vx, val_y, dx, test_y, config=cfg, device=actual_device, seed=s, patience=5)
            results[name].append(float(res["val_balanced_accuracy"]))

    stats: dict[str, Any] = {}
    for name in ("BASELINE", "MEDIUM", "HIGH"):
        vals = results[name]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = math.sqrt(variance)
        sem = std / math.sqrt(len(vals))
        t_crit = 2.262  # t(9) for 95% CI
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem
        stats[name] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "raw_val_bacc": [round(x, 4) for x in vals],
        }

    # Paired comparisons
    def paired_test(a_vals: list[float], b_vals: list[float]) -> dict[str, Any]:
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        mean_diff = sum(diffs) / len(diffs)
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        std_diff = math.sqrt(var_diff)
        sem_diff = std_diff / math.sqrt(len(diffs))
        t_stat = mean_diff / max(1e-12, sem_diff)
        # Approximate p-value from t-stat (degrees of freedom = 9)
        # Standard normal approx for tail
        p_approx = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))
        # Cohen's d
        s_pooled = math.sqrt((stats["BASELINE"]["std"] ** 2 + stats["MEDIUM"]["std"] ** 2) / 2.0)
        d = mean_diff / max(1e-12, s_pooled)
        return {
            "mean_paired_difference": round(mean_diff, 4),
            "std_difference": round(std_diff, 4),
            "t_statistic": round(t_stat, 3),
            "p_value_raw": round(p_approx, 4),
            "cohens_d": round(d, 3),
        }

    pair_med_base = paired_test(results["MEDIUM"], results["BASELINE"])
    pair_high_base = paired_test(results["HIGH"], results["BASELINE"])
    pair_high_med = paired_test(results["HIGH"], results["MEDIUM"])

    # Holm-Bonferroni correction across 3 comparisons
    sorted_pairs = sorted(
        [("MEDIUM_vs_BASELINE", pair_med_base), ("HIGH_vs_BASELINE", pair_high_base), ("HIGH_vs_MEDIUM", pair_high_med)],
        key=lambda x: x[1]["p_value_raw"],
    )
    for idx, (pair_name, p_data) in enumerate(sorted_pairs):
        alpha_corrected = 0.05 / (3 - idx)
        p_data["holm_bonferroni_threshold"] = round(alpha_corrected, 4)
        p_data["significant_after_correction"] = bool(p_data["p_value_raw"] < alpha_corrected)

    return {
        "n_paired_seeds": n_seeds,
        "profiles_summary": stats,
        "paired_comparisons": {
            "MEDIUM_vs_BASELINE": pair_med_base,
            "HIGH_vs_BASELINE": pair_high_base,
            "HIGH_vs_MEDIUM": pair_high_med,
        },
    }


def main() -> int:
    print("=" * 70)
    print("AMDE 2.0.2 FINAL SCIENTIFIC AUDIT & CUDA ACTIVITY PROFILING")
    print(f"Platform: {sys.platform} | Python: {sys.version.split()[0]}")
    torch = load_torch()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    profiles = {
        "BASELINE": PROFILE_BASELINE,
        "MEDIUM": PROFILE_MEDIUM,
        "HIGH": PROFILE_HIGH,
    }

    print("\n--- 1. PYTORCH PROFILER KERNEL SELF-TIME & STREAM BREAKDOWN ---")
    cuda_breakdowns = {}
    for name, prof in profiles.items():
        print(f"\nProfiling CUDA kernels for {name}...")
        res = run_pytorch_profiler_breakdown(
            n_qubits=prof.n_qubits,
            n_layers=prof.n_layers,
            batch_size=prof.batch_size,
            device="cuda",
        )
        cuda_breakdowns[name] = res
        print(f"  Stream Elapsed:      {res['total_stream_elapsed_ms']:.3f} ms")
        print(f"  Pure Kernel GPU Time:{res['pure_kernel_gpu_time_ms']:.3f} ms")
        print(f"  Kernel Launch Gaps:  {res['kernel_launch_gaps_ms']:.3f} ms")
        print(f"  CPU Dispatch Delay:  {res['cpu_dispatch_ms']:.3f} ms")
        print(f"  Wall Step Time:      {res['wall_step_ms']:.3f} ms")
        print(f"  True Hardware GPU %: {res['actual_hardware_utilization_percent']:.2f}%")
        print(f"  Top Kernel:          {res['top_cuda_kernels'][0]['name']} ({res['top_cuda_kernels'][0]['cuda_time_us']} us)")

    print("\n--- 2. TRAINING STEP CONSISTENCY & EARLY STOPPING AUDIT ---")
    step_consistency = {}
    for name, prof in profiles.items():
        print(f"\nAnalyzing training step execution for {name}...")
        sc = analyze_training_step_consistency(name, prof, device="cuda")
        step_consistency[name] = sc
        print(f"  Epochs Requested:    {sc['epochs_requested']}")
        print(f"  Epochs Completed:    {sc['epochs_completed']} (Early Stopped: {sc['early_stopped']})")
        print(f"  Steps per Epoch:     {sc['steps_per_epoch']}")
        print(f"  Actual Steps:        {sc['actual_steps_executed']} (Full expected: {sc['expected_full_steps']})")
        print(f"  Total Trial Time:    {sc['total_trial_seconds']:.3f} s")
        print(f"  Sec per Step:        {sc['empirical_seconds_per_step']:.4f} s")

    print("\n--- 3. PAIRED-SEED STATISTICAL PROTOCOL (10 SEEDS) ---")
    stat_results = paired_seed_statistical_comparison(device="cuda", n_seeds=10)
    for name in ("BASELINE", "MEDIUM", "HIGH"):
        s = stat_results["profiles_summary"][name]
        print(f"  {name:8s}: Mean Val BAcc = {s['mean']:.4f} +/- {s['std']:.4f} | 95% CI: [{s['ci_95'][0]}, {s['ci_95'][1]}]")
    for pair_name, p in stat_results["paired_comparisons"].items():
        print(f"  {pair_name}: diff = {p['mean_paired_difference']:+.4f}, t = {p['t_statistic']:.3f}, p = {p['p_value_raw']:.4f}, d = {p['cohens_d']:.3f}, significant = {p['significant_after_correction']}")

    # Save complete JSON artifact
    output = {
        "timestamp": time.time(),
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cuda_kernel_activity_profiling": cuda_breakdowns,
        "training_step_consistency": step_consistency,
        "statistical_comparison": stat_results,
    }
    out_file = ROOT / "FINAL_BENCHMARK_VALIDITY_MATRIX.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark validity matrix to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
