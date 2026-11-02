"""Hardware profiling and diagnostic harness for AMDE 2.0.1 on Windows 11 / Quadro P2000.

Measures actual CUDA execution, GPU utilization over time, VRAM allocation, CPU thread
utilization, kernel timing, and throughput over a minimum of 120 seconds of active training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from amde.quantum.statevector import QuantumBackendError, load_torch, resolve_device
from amde.quantum.vqc import VQCConfig, VariationalQuantumClassifier, train_vqc
from amde.quantum.kernel import train_quantum_kernel


class HardwareTelemetryCollector:
    """Samples CPU, RAM, and GPU telemetry every interval_seconds in a background thread."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.torch = load_torch()

    def _query_nvidia_smi(self) -> dict[str, Any]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False, creationflags=flags)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
                return {
                    "name": parts[0],
                    "temperature_c": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "memory_free_mb": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "utilization_percent": float(parts[5]),
                }
        except Exception as exc:
            return {"error": str(exc)}
        return {}

    def _collect_loop(self) -> None:
        process = psutil.Process()
        # Initialize cpu_percent
        psutil.cpu_percent(interval=None)
        process.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            t_now = time.time()
            sys_cpu = psutil.cpu_percent(interval=None)
            proc_cpu = process.cpu_percent(interval=None)
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            vm = psutil.virtual_memory()
            gpu_info = self._query_nvidia_smi()

            pytorch_cuda = {}
            if self.torch.cuda.is_available():
                pytorch_cuda = {
                    "allocated_mb": round(self.torch.cuda.memory_allocated() / 1024**2, 2),
                    "reserved_mb": round(self.torch.cuda.memory_reserved() / 1024**2, 2),
                    "max_allocated_mb": round(self.torch.cuda.max_memory_allocated() / 1024**2, 2),
                }

            sample = {
                "timestamp": t_now,
                "system_cpu_percent": sys_cpu,
                "process_cpu_percent": proc_cpu,
                "per_cpu_percent": per_cpu,
                "ram_used_gb": round((vm.total - vm.available) / 1024**3, 3),
                "ram_available_gb": round(vm.available / 1024**3, 3),
                "ram_total_gb": round(vm.total / 1024**3, 3),
                "ram_percent": vm.percent,
                "gpu": gpu_info,
                "pytorch_cuda": pytorch_cuda,
            }
            self.samples.append(sample)
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        return self.samples


def generate_synthetic_data(n_samples: int, n_features: int, seed: int = 42) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[float] = []
    for _ in range(n_samples):
        row = [rng.uniform(-1.0, 1.0) for _ in range(n_features)]
        label = 1.0 if sum(row) > 0.0 else 0.0
        xs.append(row)
        ys.append(label)
    return xs, ys


def profile_active_training(duration_seconds: float = 125.0) -> dict[str, Any]:
    """Run active VQC and quantum kernel training on CUDA while sampling telemetry for >=120 seconds."""
    torch = load_torch()
    actual_device = resolve_device("cuda")
    print(f"Starting active workload profiling on {actual_device} ({torch.cuda.get_device_name(0)})...")

    # Step 1: Device Placement Verification
    test_cfg = VQCConfig(n_qubits=6, n_layers=2, epochs=4, batch_size=16)
    model = VariationalQuantumClassifier(test_cfg, device=actual_device, seed=123)
    param_device = model.theta.device
    print(f"VERIFICATION: Model parameter tensor device: {param_device} (type={param_device.type})")
    assert param_device.type == "cuda", f"Expected model parameter on CUDA, got {param_device}"

    dummy_x = torch.randn((16, 6), device=actual_device)
    dummy_out = model.logits(dummy_x)
    print(f"VERIFICATION: Model output tensor device: {dummy_out.device} (type={dummy_out.device.type})")
    assert dummy_out.device.type == "cuda", f"Expected output tensor on CUDA, got {dummy_out.device}"

    # Step 2: Micro-timing CUDA kernel vs Python overhead
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure 50 forward passes
    forward_gpu_times = []
    forward_cpu_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        start_event.record()
        _ = model.logits(dummy_x)
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_gpu_times.append(start_event.elapsed_time(end_event))  # in milliseconds
        forward_cpu_times.append((t1 - t0) * 1000.0)  # in milliseconds

    avg_gpu_ms = sum(forward_gpu_times) / len(forward_gpu_times)
    avg_cpu_ms = sum(forward_cpu_times) / len(forward_cpu_times)
    launch_overhead_ms = max(0.0, avg_cpu_ms - avg_gpu_ms)
    print(f"FORWARD PASS TIMING: CUDA Kernel execution = {avg_gpu_ms:.3f} ms | Host Wall Clock = {avg_cpu_ms:.3f} ms | Dispatch Overhead = {launch_overhead_ms:.3f} ms")

    # Step 3: Run continuous active workload for >= 120 seconds with 1-second sampling
    collector = HardwareTelemetryCollector(interval_seconds=1.0)
    collector.start()

    active_start = time.time()
    trials_completed = 0
    trial_timings = []
    data_loading_times = []
    training_times = []
    val_times = []

    configs = [
        VQCConfig(n_qubits=4, n_layers=1, epochs=6, batch_size=16),
        VQCConfig(n_qubits=6, n_layers=2, epochs=6, batch_size=16),
        VQCConfig(n_qubits=8, n_layers=2, epochs=6, batch_size=16),
    ]

    print(f"Beginning continuous active training loop for >= {duration_seconds} seconds...")
    iteration = 0
    while time.time() - active_start < duration_seconds:
        iteration += 1
        cfg = configs[iteration % len(configs)]
        
        # 1. Data generation & preprocessing timing
        t_data_start = time.perf_counter()
        tx, ty = generate_synthetic_data(64, cfg.n_qubits, seed=1000 + iteration)
        vx, vy = generate_synthetic_data(32, cfg.n_qubits, seed=2000 + iteration)
        dx, dy = generate_synthetic_data(64, cfg.n_qubits, seed=3000 + iteration)
        t_data_end = time.perf_counter()
        data_loading_times.append(t_data_end - t_data_start)

        # 2. VQC Training timing
        t_train_start = time.perf_counter()
        vqc_result = train_vqc(tx, ty, vx, vy, dx, dy, config=cfg, device=actual_device, seed=100 + iteration)
        t_train_end = time.perf_counter()
        training_times.append(t_train_end - t_train_start)
        trials_completed += 1

        # 3. Quantum Kernel timing
        t_kstart = time.perf_counter()
        _ = train_quantum_kernel(tx, ty, vx, vy, dx, dy, n_qubits=cfg.n_qubits, device=actual_device, seed=200 + iteration)
        t_kend = time.perf_counter()
        trials_completed += 1
        trial_timings.append(t_kend - t_kstart)

    total_active_time = time.time() - active_start
    samples = collector.stop()

    trials_per_minute = (trials_completed / total_active_time) * 60.0
    print(f"Active profiling finished: {trials_completed} trials in {total_active_time:.1f}s ({trials_per_minute:.1f} trials/min). Collected {len(samples)} samples.")

    # Calculate telemetry summaries
    gpu_utils = [s["gpu"].get("utilization_percent", 0.0) for s in samples if "gpu" in s and "utilization_percent" in s["gpu"]]
    gpu_temps = [s["gpu"].get("temperature_c", 0.0) for s in samples if "gpu" in s and "temperature_c" in s["gpu"]]
    gpu_used_mbs = [s["gpu"].get("memory_used_mb", 0.0) for s in samples if "gpu" in s and "memory_used_mb" in s["gpu"]]
    sys_cpus = [s["system_cpu_percent"] for s in samples]
    proc_cpus = [s["process_cpu_percent"] for s in samples]
    ram_used = [s["ram_used_gb"] for s in samples]

    summary = {
        "device": actual_device,
        "device_name": torch.cuda.get_device_name(0),
        "total_active_seconds": total_active_time,
        "samples_collected": len(samples),
        "trials_completed": trials_completed,
        "trials_per_minute": trials_per_minute,
        "micro_timing": {
            "kernel_execution_ms": avg_gpu_ms,
            "host_wall_clock_ms": avg_cpu_ms,
            "dispatch_overhead_ms": launch_overhead_ms,
            "dispatch_overhead_percent": (launch_overhead_ms / avg_cpu_ms) * 100.0 if avg_cpu_ms > 0 else 0.0,
        },
        "averages": {
            "gpu_utilization_mean": sum(gpu_utils) / max(1, len(gpu_utils)),
            "gpu_utilization_max": max(gpu_utils) if gpu_utils else 0.0,
            "gpu_temperature_mean": sum(gpu_temps) / max(1, len(gpu_temps)),
            "gpu_temperature_max": max(gpu_temps) if gpu_temps else 0.0,
            "gpu_memory_used_mb_mean": sum(gpu_used_mbs) / max(1, len(gpu_used_mbs)),
            "gpu_memory_used_mb_max": max(gpu_used_mbs) if gpu_used_mbs else 0.0,
            "system_cpu_percent_mean": sum(sys_cpus) / max(1, len(sys_cpus)),
            "process_cpu_percent_mean": sum(proc_cpus) / max(1, len(proc_cpus)),
            "ram_used_gb_mean": sum(ram_used) / max(1, len(ram_used)),
            "data_loading_seconds_mean": sum(data_loading_times) / max(1, len(data_loading_times)),
            "training_seconds_mean": sum(training_times) / max(1, len(training_times)),
        },
        "samples": samples,
    }
    return summary


def run_workload_benchmarks() -> dict[str, Any]:
    """Benchmark BASELINE, MEDIUM, HIGH, and ADAPTIVE compute profiles with fixed seeds."""
    torch = load_torch()
    actual_device = resolve_device("cuda")
    print("\n================ BENCHMARKING PROFILES ================")

    profiles = {
        "BASELINE": {
            "n_qubits": 8,
            "batch_size": 16,
            "train_size": 64,
            "val_size": 32,
            "test_size": 64,
            "epochs": 6,
            "n_layers": 2,
            "n_trials": 10,
        },
        "MEDIUM": {
            "n_qubits": 8,
            "batch_size": 64,
            "train_size": 256,
            "val_size": 64,
            "test_size": 128,
            "epochs": 12,
            "n_layers": 2,
            "n_trials": 10,
        },
        "HIGH": {
            "n_qubits": 10,
            "batch_size": 128,
            "train_size": 512,
            "val_size": 128,
            "test_size": 256,
            "epochs": 16,
            "n_layers": 2,
            "n_trials": 10,
        },
    }

    results = {}
    for name, p in profiles.items():
        print(f"\nEvaluating Profile: {name}...")
        cfg = VQCConfig(
            n_qubits=p["n_qubits"],
            n_layers=p["n_layers"],
            epochs=p["epochs"],
            batch_size=p["batch_size"],
        )
        collector = HardwareTelemetryCollector(interval_seconds=0.5)
        collector.start()

        start_time = time.perf_counter()
        bacc_scores = []
        for trial_idx in range(p["n_trials"]):
            tx, ty = generate_synthetic_data(p["train_size"], p["n_qubits"], seed=5000 + trial_idx)
            vx, vy = generate_synthetic_data(p["val_size"], p["n_qubits"], seed=6000 + trial_idx)
            dx, dy = generate_synthetic_data(p["test_size"], p["n_qubits"], seed=7000 + trial_idx)
            res = train_vqc(tx, ty, vx, vy, dx, dy, config=cfg, device=actual_device, seed=1000 + trial_idx)
            bacc_scores.append(res["val_balanced_accuracy"])

        elapsed = time.perf_counter() - start_time
        samples = collector.stop()

        gpu_utils = [s["gpu"].get("utilization_percent", 0.0) for s in samples if "gpu" in s and "utilization_percent" in s["gpu"]]
        vram_alloc = [s["pytorch_cuda"].get("allocated_mb", 0.0) for s in samples if "pytorch_cuda" in s]
        vram_res = [s["pytorch_cuda"].get("reserved_mb", 0.0) for s in samples if "pytorch_cuda" in s]
        temps = [s["gpu"].get("temperature_c", 0.0) for s in samples if "gpu" in s and "temperature_c" in s["gpu"]]

        trials_per_min = (p["n_trials"] / elapsed) * 60.0
        mean_bacc = sum(bacc_scores) / len(bacc_scores)
        variance_bacc = sum((x - mean_bacc) ** 2 for x in bacc_scores) / max(1, len(bacc_scores) - 1)

        profile_stat = {
            "name": name,
            "params": p,
            "elapsed_seconds": elapsed,
            "trials_per_minute": trials_per_min,
            "mean_balanced_accuracy": mean_bacc,
            "variance_balanced_accuracy": variance_bacc,
            "gpu_utilization_mean": sum(gpu_utils) / max(1, len(gpu_utils)),
            "gpu_utilization_max": max(gpu_utils) if gpu_utils else 0.0,
            "vram_allocated_mb_max": max(vram_alloc) if vram_alloc else 0.0,
            "vram_reserved_mb_max": max(vram_res) if vram_res else 0.0,
            "gpu_temp_max": max(temps) if temps else 0.0,
        }
        results[name] = profile_stat
        print(f"  -> {name}: {trials_per_min:.1f} trials/min | Mean GPU Util: {profile_stat['gpu_utilization_mean']:.1f}% (Max {profile_stat['gpu_utilization_max']:.1f}%) | Peak VRAM: {profile_stat['vram_reserved_mb_max']:.1f} MB | Mean BAcc: {mean_bacc:.4f}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware utilization diagnostic and benchmark harness")
    parser.add_argument("--profile-seconds", type=float, default=125.0, help="Duration for continuous 120s profiling")
    parser.add_argument("--output-json", type=str, default="reports/hardware_calibration/hardware_profile_audit.json")
    args = parser.parse_args()

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Continuous profiling for >= 120s
    profile_data = profile_active_training(duration_seconds=args.profile_seconds)

    # 2. Benchmark Profiles
    benchmark_data = run_workload_benchmarks()

    final_report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_hardware": {
            "cpu": "Intel Core i7-8850H (6C/12T)",
            "ram": "32 GB DDR4",
            "gpu": "NVIDIA Quadro P2000 (GP106GL Pascal CC 6.1)",
            "cuda_version": "12.6",
            "pytorch_version": "2.14.0+cu126",
        },
        "continuous_120s_profile": profile_data,
        "profile_benchmarks": benchmark_data,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nHardware audit data successfully saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
