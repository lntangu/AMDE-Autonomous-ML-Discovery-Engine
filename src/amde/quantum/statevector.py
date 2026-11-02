"""Small, resource-bounded differentiable statevector simulator (PyTorch optional).

Qubit 0 is the least-significant basis bit. This simulator is for research,
validation and small-circuit experiments; it is not a physical QPU interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class QuantumBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class StatevectorBudget:
    max_qubits: int = 22
    max_statevector_mb: int = 512
    dtype: str = "complex64"

    def validate(self, n_qubits: int, *, batch_size: int = 1) -> int:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        if n_qubits > self.max_qubits:
            raise QuantumBackendError(
                f"{n_qubits} qubits exceeds configured simulator limit {self.max_qubits}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        bytes_per_complex = 8 if self.dtype == "complex64" else 16
        required = (1 << n_qubits) * batch_size * bytes_per_complex
        limit = int(self.max_statevector_mb) * 1024 * 1024
        if required > limit:
            raise QuantumBackendError(
                f"statevector needs {required / 1024**2:.1f} MiB, above configured cap "
                f"{self.max_statevector_mb} MiB"
            )
        return required


def load_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional installation
        raise QuantumBackendError(
            "PyTorch is required for the quantum simulator. Install a PyTorch build "
            "compatible with your CPU/CUDA environment."
        ) from exc
    return torch


def resolve_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    torch = load_torch()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise QuantumBackendError("CUDA was requested but PyTorch CUDA is unavailable")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _broadcast_angle(torch: Any, angle: Any, state: Any) -> tuple[Any, Any]:
    value = torch.as_tensor(angle, dtype=torch.float32, device=state.device)
    c = torch.cos(value * 0.5)
    s = torch.sin(value * 0.5)
    if value.ndim == 0:
        return c, s
    if value.shape != (state.shape[0],):
        raise ValueError("batched gate angle must have shape [batch]")
    return c[:, None, None], s[:, None, None]


def initial_state(n_qubits: int, *, batch_size: int = 1, device: str = "cpu",
                  budget: StatevectorBudget | None = None) -> Any:
    torch = load_torch()
    (budget or StatevectorBudget()).validate(n_qubits, batch_size=batch_size)
    actual_device = resolve_device(device)
    state = torch.zeros((batch_size, 1 << n_qubits), dtype=torch.complex64, device=actual_device)
    state[:, 0] = 1.0 + 0.0j
    return state


def apply_h(state: Any, n_qubits: int, target: int) -> Any:
    torch = load_torch()
    _check_qubit(n_qubits, target)
    batch = state.shape[0]
    low = 1 << target
    high = 1 << (n_qubits - target - 1)
    pairs = state.reshape(batch, high, 2, low)
    a, b = pairs[:, :, 0, :], pairs[:, :, 1, :]
    scale = 1.0 / math.sqrt(2.0)
    return torch.stack(((a + b) * scale, (a - b) * scale), dim=2).reshape(batch, -1)


def apply_ry(state: Any, n_qubits: int, target: int, angle: Any) -> Any:
    torch = load_torch()
    _check_qubit(n_qubits, target)
    batch = state.shape[0]
    low = 1 << target
    high = 1 << (n_qubits - target - 1)
    pairs = state.reshape(batch, high, 2, low)
    a, b = pairs[:, :, 0, :], pairs[:, :, 1, :]
    c, s = _broadcast_angle(torch, angle, state)
    out0 = c * a - s * b
    out1 = s * a + c * b
    return torch.stack((out0, out1), dim=2).reshape(batch, -1)


def apply_rz(state: Any, n_qubits: int, target: int, angle: Any) -> Any:
    torch = load_torch()
    _check_qubit(n_qubits, target)
    batch = state.shape[0]
    low = 1 << target
    high = 1 << (n_qubits - target - 1)
    pairs = state.reshape(batch, high, 2, low)
    a, b = pairs[:, :, 0, :], pairs[:, :, 1, :]
    c, s = _broadcast_angle(torch, angle, state)
    phase0 = torch.complex(c, -s)
    phase1 = torch.complex(c, s)
    return torch.stack((phase0 * a, phase1 * b), dim=2).reshape(batch, -1)


_CNOT_CACHE: dict[tuple[int, int, int, str], Any] = {}
_SIGNS_CACHE: dict[tuple[int, int, str], Any] = {}


def apply_cnot(state: Any, n_qubits: int, control: int, target: int) -> Any:
    torch = load_torch()
    _check_qubit(n_qubits, control)
    _check_qubit(n_qubits, target)
    if control == target:
        raise ValueError("CNOT control and target must differ")
    key = (n_qubits, control, target, str(state.device))
    permutation = _CNOT_CACHE.get(key)
    if permutation is None:
        indices = torch.arange(1 << n_qubits, device=state.device, dtype=torch.long)
        flip = ((indices >> control) & 1) << target
        permutation = indices ^ flip
        _CNOT_CACHE[key] = permutation
    return state.index_select(1, permutation)


def expectation_z(state: Any, n_qubits: int, target: int = 0) -> Any:
    torch = load_torch()
    _check_qubit(n_qubits, target)
    key = (n_qubits, target, str(state.device))
    signs = _SIGNS_CACHE.get(key)
    if signs is None:
        indices = torch.arange(1 << n_qubits, device=state.device, dtype=torch.long)
        signs = torch.where(((indices >> target) & 1) == 0, 1.0, -1.0).to(torch.float32)
        _SIGNS_CACHE[key] = signs
    probabilities = state.real.square() + state.imag.square()
    return (probabilities * signs.unsqueeze(0)).sum(dim=1)


def grover_search_statevector(
    n_qubits: int,
    marked_indices: list[int],
    *,
    iterations: int | None = None,
    device: str = "cpu",
    budget: StatevectorBudget | None = None,
) -> dict[str, Any]:
    """Run ideal Grover amplitude amplification with a table-backed phase oracle.

    The table-backed oracle is a simulator convenience, not a reversible gate
    decomposition and not evidence of end-to-end quantum speedup. Oracle
    construction and simulator wall-clock time are reported separately.
    """
    import time

    torch = load_torch()
    if not marked_indices:
        raise ValueError("at least one marked state is required")
    size = 1 << n_qubits
    if len(set(marked_indices)) != len(marked_indices):
        raise ValueError("marked_indices must be unique")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= size for i in marked_indices):
        raise ValueError("marked index outside search space")
    actual_device = resolve_device(device)
    (budget or StatevectorBudget()).validate(n_qubits)
    marked_count = len(marked_indices)
    theta = math.asin(math.sqrt(marked_count / size))
    optimal = max(0, int(round(math.pi / (4.0 * theta) - 0.5)))
    count = optimal if iterations is None else int(iterations)
    if count < 0 or count > 1_000_000:
        raise ValueError("iterations must be between 0 and 1,000,000")

    started = time.perf_counter()
    state = torch.full((1, size), 1.0 / math.sqrt(size), dtype=torch.complex64, device=actual_device)
    marked = torch.as_tensor(marked_indices, dtype=torch.long, device=actual_device)
    for _ in range(count):
        # Phase oracle O_f: |x> -> (-1)^f(x)|x>.
        state = state.clone()
        state[:, marked] = -state[:, marked]
        # Diffusion operator 2|s><s| - I.
        mean = state.mean(dim=1, keepdim=True)
        state = 2.0 * mean - state
    if actual_device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    probabilities = state.abs().square().squeeze(0)
    marked_prob = float(probabilities[marked].sum().item())
    top_index = int(torch.argmax(probabilities).item())
    sampled_index = int(torch.multinomial(probabilities.detach().cpu(), 1).item())
    marked_set = set(marked_indices)
    return {
        "algorithm": "grover_table_oracle_statevector_v1",
        "n_qubits": n_qubits,
        "search_space_size": size,
        "marked_count": marked_count,
        "iterations": count,
        "oracle_queries": count,
        "optimal_iterations_estimate": optimal,
        "marked_probability": marked_prob,
        "top_measurement": top_index,
        "top_measurement_is_marked": top_index in marked_set,
        "sampled_measurement": sampled_index,
        "sampled_measurement_is_marked": sampled_index in marked_set,
        "device": actual_device,
        "statevector_bytes": size * 8,
        "simulator_seconds": elapsed,
        "simulation_semantics": "ideal noiseless statevector; table-backed phase oracle",
        "oracle_cost_included_in_simulator_seconds": False,
    }


def _check_qubit(n_qubits: int, target: int) -> None:
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < n_qubits:
        raise ValueError(f"qubit index must be in [0, {n_qubits - 1}]")
