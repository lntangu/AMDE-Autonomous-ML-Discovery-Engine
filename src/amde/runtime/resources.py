"""Hardware detection and conservative resource monitoring."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ResourcePlan, ResourcePolicy
from ..util import utc_now


@dataclass
class HardwareInfo:
    os: str
    platform: str
    python_version: str
    python_executable: str
    logical_cores: int
    physical_cores: int
    ram_total_mb: float
    ram_available_mb: float
    disk_total_gb: float
    disk_free_gb: float
    project_root: str
    gpu_available: bool
    gpu_name: str = "unavailable"
    gpu_driver: str = "unavailable"
    gpu_memory_total_mb: float | None = None
    gpu_memory_free_mb: float | None = None
    gpu_temperature_c: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_compute_capability: str | None = None
    gpu_reason: str = ""
    memory_source: str = "unavailable"
    torch: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    detected_at: str = field(default_factory=utc_now)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _windows_memory() -> tuple[float, float] | None:
    if os.name != "nt":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return (status.ullTotalPhys / (1024**2), status.ullAvailPhys / (1024**2))
    except Exception:
        return None
    return None


def _linux_proc_meminfo() -> tuple[float, float] | None:
    proc_path = Path("/proc/meminfo")
    if not proc_path.exists():
        return None
    try:
        values_kb: dict[str, float] = {}
        for line in proc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts:
                values_kb[key.strip()] = float(parts[0])
        total_kb = values_kb.get("MemTotal", 0.0)
        avail_kb = values_kb.get("MemAvailable", values_kb.get("MemFree", 0.0))
        if total_kb > 0.0:
            return (total_kb / 1024.0, avail_kb / 1024.0)
    except Exception:
        return None
    return None


def _mem_info() -> tuple[float, float, str]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return (vm.total / (1024**2), vm.available / (1024**2), "psutil")
    except Exception:
        pass
    win = _windows_memory()
    if win:
        return (win[0], win[1], "windows-api")
    linux_mem = _linux_proc_meminfo()
    if linux_mem:
        return (linux_mem[0], linux_mem[1], "proc-meminfo")
    return (0.0, 0.0, "unavailable")


def _nvidia_smi() -> dict[str, Any]:
    base_fields = "name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu"
    command = ["nvidia-smi", f"--query-gpu={base_fields}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except Exception as exc:
        return {"available": False, "reason": f"nvidia-smi unavailable: {exc}"}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "reason": (completed.stderr or completed.stdout or "nvidia-smi returned no data").strip()[:500],
        }
    line = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return {"available": False, "reason": f"unexpected nvidia-smi output: {line[:300]}"}

    def number(value: str) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    result = {
        "available": True,
        "name": parts[0],
        "driver": parts[1],
        "memory_total_mb": number(parts[2]),
        "memory_free_mb": number(parts[3]),
        "temperature_c": number(parts[4]),
        "utilization_percent": number(parts[5]),
        "compute_capability": None,
        "reason": "",
    }
    # compute_cap is available on newer drivers; failure is non-fatal.
    try:
        cc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if cc.returncode == 0 and cc.stdout.strip():
            result["compute_capability"] = cc.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return result


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unavailable"


def _torch_probe() -> dict[str, Any]:
    script = r"""
import json, sys, time
out = {"import_ok": False}
try:
    import torch
    out["import_ok"] = True
    out["version"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_arch_list"] = torch.cuda.get_arch_list() if out["cuda_available"] else []
    if out["cuda_available"]:
        out["device_name"] = torch.cuda.get_device_name(0)
        out["device_capability"] = torch.cuda.get_device_capability(0)
        t0 = time.perf_counter()
        x = torch.ones((8, 8), device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        out["tensor_smoke"] = {"ok": True, "value": float(y), "seconds": time.perf_counter() - t0}
    else:
        out["tensor_smoke"] = {"ok": False, "reason": "cuda not available"}
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if completed.returncode != 0:
            return {
                "import_ok": False,
                "error": (completed.stderr or completed.stdout).strip()[-2000:],
            }
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"import_ok": False, "error": f"torch probe failed: {exc}"}


def detect_hardware(project_root: Path, *, probe_torch: bool = False) -> HardwareInfo:
    project_root = Path(project_root).resolve()
    logical = os.cpu_count() or 1
    physical = logical
    warnings: list[str] = []
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        warnings.append("psutil unavailable; physical core count is an estimate")
    total_mb, available_mb, memory_source = _mem_info()
    usage = shutil.disk_usage(project_root)
    gpu = _nvidia_smi()
    info = HardwareInfo(
        os=f"{platform.system()} {platform.release()} {platform.version()}",
        platform=platform.platform(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        logical_cores=logical,
        physical_cores=physical,
        ram_total_mb=total_mb,
        ram_available_mb=available_mb,
        disk_total_gb=usage.total / (1024**3),
        disk_free_gb=usage.free / (1024**3),
        project_root=str(project_root),
        gpu_available=bool(gpu.get("available")),
        gpu_name=str(gpu.get("name", "unavailable")),
        gpu_driver=str(gpu.get("driver", "unavailable")),
        gpu_memory_total_mb=gpu.get("memory_total_mb"),
        gpu_memory_free_mb=gpu.get("memory_free_mb"),
        gpu_temperature_c=gpu.get("temperature_c"),
        gpu_utilization_percent=gpu.get("utilization_percent"),
        gpu_compute_capability=gpu.get("compute_capability"),
        gpu_reason=str(gpu.get("reason", "")),
        memory_source=memory_source,
        package_versions={
            name: _package_version(name)
            for name in ("numpy", "psutil", "torch", "pytest", "scikit-learn", "scipy")
        },
        warnings=warnings,
    )
    if memory_source == "unavailable":
        warnings.append("RAM total/available could not be detected; resource plan is conservative")
    if probe_torch and info.gpu_available:
        info.torch = _torch_probe()
    elif probe_torch:
        info.torch = {
            "import_ok": False,
            "cuda_available": False,
            "tensor_smoke": {
                "ok": False,
                "reason": "NVIDIA GPU was not detected; skipped Torch import/probe",
            },
        }
    return info


class ResourceSampler:
    def __init__(
        self,
        policy: ResourcePolicy,
        project_root: Path,
        *,
        cache_seconds: float = 5.0,
        gpu_active: bool = False,
    ) -> None:
        self.policy = policy
        self.project_root = Path(project_root)
        self.cache_seconds = float(cache_seconds)
        self.gpu_active = bool(gpu_active)
        self._last: dict[str, Any] | None = None
        self._last_time = 0.0

    def snapshot(self, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._last is not None and now - self._last_time < self.cache_seconds:
            return dict(self._last)
        sample: dict[str, Any] = {"ts": utc_now()}
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            sample["ram_total_mb"] = vm.total / (1024**2)
            sample["ram_available_mb"] = vm.available / (1024**2)
            sample["ram_percent"] = vm.percent
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            sample["memory_source"] = "psutil"
        except Exception:
            win = _windows_memory()
            if win:
                sample["ram_total_mb"] = win[0]
                sample["ram_available_mb"] = win[1]
                sample["memory_source"] = "windows-api"
            else:
                linux_mem = _linux_proc_meminfo()
                if linux_mem:
                    sample["ram_total_mb"] = linux_mem[0]
                    sample["ram_available_mb"] = linux_mem[1]
                    sample["memory_source"] = "proc-meminfo"
                else:
                    sample["memory_source"] = "unavailable"
            sample["ram_percent"] = (
                round(100.0 * (1.0 - sample["ram_available_mb"] / sample["ram_total_mb"]), 1)
                if sample.get("ram_total_mb")
                else None
            )
            sample["cpu_percent"] = None
        try:
            sample["disk_free_gb"] = shutil.disk_usage(self.project_root).free / (1024**3)
        except Exception:
            sample["disk_free_gb"] = None
        gpu = _nvidia_smi()
        sample["gpu_available"] = bool(gpu.get("available"))
        sample["gpu_memory_free_mb"] = gpu.get("memory_free_mb")
        sample["gpu_temperature_c"] = gpu.get("temperature_c")
        sample["gpu_utilization_percent"] = gpu.get("utilization_percent")
        sample["gpu_memory_total_mb"] = gpu.get("memory_total_mb")
        self._last = sample
        self._last_time = now
        return dict(sample)

    def pressure(self, sample: dict[str, Any] | None = None) -> list[str]:
        sample = sample or self.snapshot()
        reasons = []
        ram = sample.get("ram_available_mb")
        if isinstance(ram, (int, float)):
            reserve_mb = self.policy.effective_ram_reserve_mb(sample.get("ram_total_mb"))
            if ram < reserve_mb:
                reasons.append(f"RAM available {ram:.0f} MB below effective reserve {reserve_mb} MB")
        disk = sample.get("disk_free_gb")
        if isinstance(disk, (int, float)) and disk < self.policy.data["disk"]["min_free_gb"]:
            reasons.append(f"disk free {disk:.1f} GB below reserve")
        if self.gpu_active:
            temp = sample.get("gpu_temperature_c")
            if isinstance(temp, (int, float)) and temp >= self.policy.data["gpu"]["thermal_stop_c"]:
                reasons.append(f"GPU temperature {temp:.0f}C at/above stop threshold")
            free_vram = sample.get("gpu_memory_free_mb")
            if isinstance(free_vram, (int, float)):
                gpu_policy = self.policy.data["gpu"]
                required = float(gpu_policy["reserve_vram_mb"]) + float(gpu_policy["required_free_vram_mb"])
                if free_vram < required:
                    reasons.append(f"GPU free VRAM {free_vram:.0f} MB below active reserve {required:.0f} MB")
        return reasons


def set_blas_threads(threads: int = 1) -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(max(1, int(threads)))


def plan_text(plan: ResourcePlan) -> str:
    return (
        f"workers={plan.cpu_workers} device={plan.device_effective} "
        f"gpu_slots={plan.gpu_slots} reserve_ram={plan.reserve_ram_mb}MB "
        f"reason={plan.reason}"
    )
