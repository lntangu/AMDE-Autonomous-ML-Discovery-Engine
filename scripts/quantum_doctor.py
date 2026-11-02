#!/usr/bin/env python3
"""Preflight the optional PyTorch exact-statevector research backend."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde import __version__  # noqa: E402
from amde.quantum.statevector import (  # noqa: E402
    QuantumBackendError,
    StatevectorBudget,
    apply_cnot,
    apply_h,
    expectation_z,
    initial_state,
    load_torch,
    resolve_device,
)


def _gpu_temperature() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            value = completed.stdout.strip().splitlines()[0].strip()
            return float(value) if value.replace(".", "", 1).isdigit() else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def run(device: str, max_statevector_mb: int) -> tuple[dict[str, Any], int]:
    report: dict[str, Any] = {
        "engine_version": __version__,
        "requested_device": device,
        "physical_qpu": False,
        "simulator": "pytorch_exact_statevector",
        "max_statevector_mb": max_statevector_mb,
        "checks": {},
        "warnings": [],
    }
    try:
        torch = load_torch()
        report["pytorch"] = str(torch.__version__)
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        actual = resolve_device(device)
        report["effective_device"] = actual
        if actual == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "free_vram_mb": round(free_bytes / (1024**2), 2),
                "total_vram_mb": round(total_bytes / (1024**2), 2),
                "temperature_c": _gpu_temperature(),
            }
            if free_bytes < 1536 * 1024**2:
                raise QuantumBackendError("less than 1536 MiB free VRAM; configured quantum reserve is not met")
        torch.set_num_threads(1)
        started = time.perf_counter()
        state = initial_state(4, device=actual, budget=StatevectorBudget(max_qubits=12, max_statevector_mb=max_statevector_mb))
        state = apply_h(state, 4, 0)
        state = apply_cnot(state, 4, 0, 1)
        expectation = float(expectation_z(state, 4, 0).detach().cpu().item())
        if actual == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        norm = float(torch.sum(state.abs().square()).detach().cpu().item())
        report["checks"] = {
            "four_qubit_statevector": "PASS",
            "norm": norm,
            "expectation_z_q0": expectation,
            "elapsed_seconds": elapsed,
            "finite": math.isfinite(norm) and math.isfinite(expectation),
        }
        report["status"] = "PASS" if abs(norm - 1.0) < 1e-5 and math.isfinite(expectation) else "FAIL"
        report["limitations"] = [
            "This is a classical simulation of ideal statevectors, not a physical QPU run.",
            "Exact statevector memory scales as 2^n complex amplitudes; configured budgets are enforced.",
            "A successful four-qubit smoke does not establish quantum advantage or large-circuit capacity.",
        ]
        return report, 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report, 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-statevector-mb", type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_statevector_mb < 1:
        parser.error("--max-statevector-mb must be >= 1")
    report, code = run(args.device, args.max_statevector_mb)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
