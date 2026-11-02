"""AMDE 4.0 Hardware Capability Detection & Deployment Profiling.

Specifically tailored for the primary workstation environment:
- Intel Core i7-8850H (6 cores / 12 threads)
- 32 GB DDR4 RAM
- NVIDIA Quadro P2000 Mobile (4 GB GDDR5, Pascal arch, Compute Capability 6.1)
- Explicit non-assumption of Tensor Cores
- Telemetry distinction between unavailable vs zero
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import psutil


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware capability profile and operational boundaries."""
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_mb: float
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_total_memory_mb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None
    has_tensor_cores: bool = False
    safe_worker_concurrency: int = 4
    safe_gpu_memory_cap_mb: float = 2560.0  # 2.5 GB cap on 4 GB Quadro P2000

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Probes the live operating environment and builds an immutable capability profile."""
        phys_cores = psutil.cpu_count(logical=False) or 6
        log_cores = psutil.cpu_count(logical=True) or 12
        ram_mb = round(psutil.virtual_memory().total / (1024.0 * 1024.0), 1)

        cuda_avail = False
        gpu_name = None
        gpu_mem_mb = None
        compute_cap = None
        tensor_cores = False

        try:
            import torch
            if torch.cuda.is_available():
                cuda_avail = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0), 1)
                major, minor = torch.cuda.get_device_capability(0)
                compute_cap = f"{major}.{minor}"
                # Pascal (6.x) and earlier do NOT have Tensor Cores (Volta 7.0+ introduced Tensor Cores)
                tensor_cores = (major >= 7)
        except ImportError:
            pass

        # Concurrency: default to min(4, physical cores) to prevent hyperthreading thrash on 6-core chip
        safe_workers = min(4, phys_cores)

        return cls(
            cpu_physical_cores=phys_cores,
            cpu_logical_cores=log_cores,
            total_ram_mb=ram_mb,
            cuda_available=cuda_avail,
            gpu_name=gpu_name,
            gpu_total_memory_mb=gpu_mem_mb,
            gpu_compute_capability=compute_cap,
            has_tensor_cores=tensor_cores,
            safe_worker_concurrency=safe_workers,
            safe_gpu_memory_cap_mb=min(2560.0, (gpu_mem_mb * 0.75) if gpu_mem_mb else 0.0),
        )

    def sample_telemetry(self) -> dict[str, Any]:
        """Samples instantaneous resource telemetry, distinguishing unavailable from zero."""
        mem = psutil.virtual_memory()
        telemetry: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": round((mem.total - mem.available) / (1024.0 * 1024.0), 1),
            "ram_available_mb": round(mem.available / (1024.0 * 1024.0), 1),
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
        }

        if self.cuda_available:
            try:
                import torch
                telemetry["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024.0 * 1024.0), 1)
                telemetry["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024.0 * 1024.0), 1)
            except Exception:
                pass

        return telemetry
