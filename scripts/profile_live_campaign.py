"""Live 120-Second CPU/GPU Diagnostics Sampler for AMDE 2.0.1 (amde2six003).

Performs strict READ-ONLY sampling of host hardware and process performance
during the active six-hour campaign on Windows 11 / Quadro P2000.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_gpu_metrics() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu,clocks.current.sm,clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=flags)
        line = proc.stdout.strip()
        parts = [p.strip() for p in line.split(",")]
        return {
            "utilization_gpu_percent": float(parts[0]),
            "utilization_mem_percent": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "memory_free_mb": float(parts[3]),
            "memory_used_mb": float(parts[4]),
            "temperature_c": float(parts[5]),
            "sm_clock_mhz": float(parts[6]),
            "throttle_reasons_active": parts[7],
            "sw_thermal_slowdown": parts[8],
            "hw_thermal_slowdown": parts[9],
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_gpu_compute_apps() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    apps = []
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=flags)
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                apps.append({
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_memory_mb": parts[2] if len(parts) > 2 else "N/A",
                })
    except Exception:
        pass
    return apps


def main() -> int:
    duration_seconds = 120.0
    interval = 1.0
    campaign_id = "amde2six003"
    print(f"Starting 120-second live diagnostics for campaign: {campaign_id}")

    # Monitored processes
    monitored_pids = {
        "guardian": 2784,
        "manager": 11316,
        "classical": 21952,
        "quantum": 29388,
        "timer": 17036,
    }

    proc_objects: dict[str, psutil.Process] = {}
    for role, pid in monitored_pids.items():
        try:
            p = psutil.Process(pid)
            # prime cpu_percent
            p.cpu_percent(interval=None)
            proc_objects[role] = p
            print(f"Tracking {role} (PID={pid}): {p.name()}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            print(f"Warning: could not attach to {role} (PID={pid}): {exc}")

    # Initial prime for system CPU
    psutil.cpu_percent(interval=None, percpu=True)

    samples: list[dict[str, Any]] = []
    start_time = time.monotonic()
    end_time = start_time + duration_seconds

    print("Sampling telemetry every 1.0s...")
    sample_index = 0
    while time.monotonic() < end_time:
        t_sample_start = time.monotonic()
        sys_cpu_total = psutil.cpu_percent(interval=None)
        sys_cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        vmem = psutil.virtual_memory()

        # Per-process sampling
        proc_data: dict[str, Any] = {}
        for role, p in proc_objects.items():
            try:
                if p.is_running():
                    mem = p.memory_info()
                    proc_data[role] = {
                        "pid": p.pid,
                        "alive": True,
                        "cpu_percent": p.cpu_percent(interval=None),
                        "rss_mb": mem.rss / (1024 * 1024),
                        "num_threads": p.num_threads(),
                        "status": p.status(),
                    }
                else:
                    proc_data[role] = {"pid": p.pid, "alive": False}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_data[role] = {"pid": p.pid, "alive": False}

        gpu_data = get_gpu_metrics()

        samples.append({
            "timestamp": time.time(),
            "elapsed": time.monotonic() - start_time,
            "system_cpu_total": sys_cpu_total,
            "system_cpu_per_core": sys_cpu_per_core,
            "ram_used_gb": (vmem.total - vmem.available) / (1024**3),
            "ram_available_gb": vmem.available / (1024**3),
            "ram_percent": vmem.percent,
            "processes": proc_data,
            "gpu": gpu_data,
        })
        sample_index += 1
        if sample_index % 15 == 0:
            print(f"[{samples[-1]['elapsed']:.1f}s / {duration_seconds}s] "
                  f"SysCPU={sys_cpu_total:.1f}% | "
                  f"ClassicalCPU={proc_data.get('classical', {}).get('cpu_percent', 0.0):.1f}% | "
                  f"QuantumCPU={proc_data.get('quantum', {}).get('cpu_percent', 0.0):.1f}% | "
                  f"GPU={gpu_data.get('utilization_gpu_percent', 0.0):.1f}% "
                  f"({gpu_data.get('temperature_c', 0.0)}C, VRAM={gpu_data.get('memory_used_mb', 0.0):.0f}MB)")

        # Precise sleep for remainder of interval
        elapsed_loop = time.monotonic() - t_sample_start
        sleep_dur = max(0.05, interval - elapsed_loop)
        time.sleep(sleep_dur)

    total_duration = time.monotonic() - start_time
    print(f"Collected {len(samples)} samples over {total_duration:.2f} seconds.")

    # Compute summary statistics
    gpu_utils = [s["gpu"].get("utilization_gpu_percent", 0.0) for s in samples if "utilization_gpu_percent" in s["gpu"]]
    gpu_temps = [s["gpu"].get("temperature_c", 0.0) for s in samples if "temperature_c" in s["gpu"]]
    gpu_vram = [s["gpu"].get("memory_used_mb", 0.0) for s in samples if "memory_used_mb" in s["gpu"]]
    sys_cpus = [s["system_cpu_total"] for s in samples]
    classical_cpus = [s["processes"].get("classical", {}).get("cpu_percent", 0.0) for s in samples if "classical" in s["processes"]]
    quantum_cpus = [s["processes"].get("quantum", {}).get("cpu_percent", 0.0) for s in samples if "quantum" in s["processes"]]
    manager_cpus = [s["processes"].get("manager", {}).get("cpu_percent", 0.0) for s in samples if "manager" in s["processes"]]
    guardian_cpus = [s["processes"].get("guardian", {}).get("cpu_percent", 0.0) for s in samples if "guardian" in s["processes"]]
    ram_used = [s["ram_used_gb"] for s in samples]
    ram_avail = [s["ram_available_gb"] for s in samples]

    summary = {
        "campaign_id": campaign_id,
        "sample_count": len(samples),
        "total_duration_seconds": total_duration,
        "gpu": {
            "utilization_percent_mean": sum(gpu_utils) / max(1, len(gpu_utils)),
            "utilization_percent_max": max(gpu_utils) if gpu_utils else 0.0,
            "utilization_percent_min": min(gpu_utils) if gpu_utils else 0.0,
            "temperature_c_mean": sum(gpu_temps) / max(1, len(gpu_temps)),
            "temperature_c_max": max(gpu_temps) if gpu_temps else 0.0,
            "vram_used_mb_mean": sum(gpu_vram) / max(1, len(gpu_vram)),
            "vram_used_mb_max": max(gpu_vram) if gpu_vram else 0.0,
            "active_compute_apps": get_gpu_compute_apps(),
        },
        "system_cpu": {
            "utilization_percent_mean": sum(sys_cpus) / max(1, len(sys_cpus)),
            "utilization_percent_max": max(sys_cpus) if sys_cpus else 0.0,
            "per_core_means": [
                sum(s["system_cpu_per_core"][i] for s in samples) / max(1, len(samples))
                for i in range(12)
            ] if samples and len(samples[0]["system_cpu_per_core"]) == 12 else [],
        },
        "processes": {
            "classical_worker": {
                "pid": monitored_pids["classical"],
                "cpu_percent_mean": sum(classical_cpus) / max(1, len(classical_cpus)),
                "cpu_percent_max": max(classical_cpus) if classical_cpus else 0.0,
                "rss_mb_latest": samples[-1]["processes"].get("classical", {}).get("rss_mb", 0.0),
                "threads": samples[-1]["processes"].get("classical", {}).get("num_threads", 0),
            },
            "quantum_worker": {
                "pid": monitored_pids["quantum"],
                "cpu_percent_mean": sum(quantum_cpus) / max(1, len(quantum_cpus)),
                "cpu_percent_max": max(quantum_cpus) if quantum_cpus else 0.0,
                "rss_mb_latest": samples[-1]["processes"].get("quantum", {}).get("rss_mb", 0.0),
                "threads": samples[-1]["processes"].get("quantum", {}).get("num_threads", 0),
            },
            "manager": {
                "pid": monitored_pids["manager"],
                "cpu_percent_mean": sum(manager_cpus) / max(1, len(manager_cpus)),
                "cpu_percent_max": max(manager_cpus) if manager_cpus else 0.0,
                "rss_mb_latest": samples[-1]["processes"].get("manager", {}).get("rss_mb", 0.0),
            },
            "guardian": {
                "pid": monitored_pids["guardian"],
                "cpu_percent_mean": sum(guardian_cpus) / max(1, len(guardian_cpus)),
                "cpu_percent_max": max(guardian_cpus) if guardian_cpus else 0.0,
                "rss_mb_latest": samples[-1]["processes"].get("guardian", {}).get("rss_mb", 0.0),
            },
        },
        "ram": {
            "used_gb_mean": sum(ram_used) / max(1, len(ram_used)),
            "available_gb_mean": sum(ram_avail) / max(1, len(ram_avail)),
            "growth_during_sample_mb": (ram_used[-1] - ram_used[0]) * 1024,
        },
    }

    # Save to disk
    out_dir = REPO_ROOT / "reports" / "hardware_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_diagnostic_amde2six003.json"
    full_output = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "samples": samples,
    }
    out_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    print(f"Saved live diagnostic report to: {out_path}")

    # Output formatted JSON summary
    print("\n=== LIVE DIAGNOSTIC SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
