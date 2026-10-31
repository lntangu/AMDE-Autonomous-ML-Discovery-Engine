"""Quantum fidelity-kernel and classical RBF-kernel baselines."""
from __future__ import annotations

import math
import time
from typing import Any

from .statevector import apply_cnot, apply_ry, apply_rz, initial_state, load_torch, resolve_device


def _feature_states(features: Any, *, n_qubits: int, device: str, feature_map: str) -> Any:
    torch = load_torch()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if x.ndim != 2 or x.shape[1] > n_qubits:
        raise ValueError("feature matrix must be 2D and no wider than n_qubits")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("features contain non-finite values")
    if x.shape[1] < n_qubits:
        x = torch.nn.functional.pad(x, (0, n_qubits - x.shape[1]))
    x = torch.clamp(x, -1.0, 1.0) * math.pi
    state = initial_state(n_qubits, batch_size=int(x.shape[0]), device=device)
    for qubit in range(n_qubits):
        state = apply_ry(state, n_qubits, qubit, x[:, qubit])
        if feature_map == "angle_ry_rz":
            state = apply_rz(state, n_qubits, qubit, x[:, qubit] * 0.5)
    if feature_map not in {"angle_ry", "angle_ry_rz"}:
        raise ValueError("feature_map must be angle_ry or angle_ry_rz")
    for qubit in range(n_qubits - 1):
        state = apply_cnot(state, n_qubits, qubit, qubit + 1)
    if n_qubits > 2:
        state = apply_cnot(state, n_qubits, n_qubits - 1, 0)
    return state


def _fidelity_kernel(left: Any, right: Any) -> Any:
    torch = load_torch()
    overlap = left @ right.conj().transpose(0, 1)
    return overlap.real.square() + overlap.imag.square()


def _bacc(predictions: Any, labels: Any) -> float:
    torch = load_torch()
    predicted = predictions >= 0.0
    positive = labels >= 0.5
    negative = ~positive
    tpr = (predicted & positive).float().sum() / positive.float().sum().clamp_min(1.0)
    tnr = ((~predicted) & negative).float().sum() / negative.float().sum().clamp_min(1.0)
    return float((0.5 * (tpr + tnr)).item())


def train_quantum_kernel(
    train_x: list[list[float]], train_y: list[float],
    val_x: list[list[float]], val_y: list[float],
    test_x: list[list[float]], test_y: list[float],
    *, n_qubits: int = 8, device: str = "auto", feature_map: str = "angle_ry_rz",
    regularization: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0), seed: int = 1,
) -> dict[str, Any]:
    """Fidelity-kernel ridge classifier with validation-selected regularization."""
    torch = load_torch()
    actual_device = resolve_device(device)
    torch.set_num_threads(1)
    if not 2 <= n_qubits <= 12:
        raise ValueError("quantum kernel n_qubits must be between 2 and 12")
    started = time.perf_counter()
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=actual_device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=actual_device)
    x_test = torch.as_tensor(test_x, dtype=torch.float32, device=actual_device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=actual_device)
    y_val = torch.as_tensor(val_y, dtype=torch.float32, device=actual_device)
    y_test = torch.as_tensor(test_y, dtype=torch.float32, device=actual_device)
    if len(x_train) < 2 or len(x_val) < 1 or len(x_test) < 1:
        raise ValueError("kernel train/validation/test splits must be non-empty")
    if len(set(float(v) for v in train_y)) != 2:
        raise ValueError("binary training split must contain both classes")
    train_states = _feature_states(x_train, n_qubits=n_qubits, device=actual_device, feature_map=feature_map)
    val_states = _feature_states(x_val, n_qubits=n_qubits, device=actual_device, feature_map=feature_map)
    test_states = _feature_states(x_test, n_qubits=n_qubits, device=actual_device, feature_map=feature_map)
    k_train = _fidelity_kernel(train_states, train_states)
    k_val = _fidelity_kernel(val_states, train_states)
    k_test = _fidelity_kernel(test_states, train_states)
    signed_y = y_train * 2.0 - 1.0
    best_reg = None
    best_val = -math.inf
    best_alpha = None
    eye = torch.eye(len(x_train), dtype=torch.float32, device=actual_device)
    for reg in regularization:
        if not math.isfinite(float(reg)) or float(reg) <= 0:
            raise ValueError("kernel regularization values must be finite and positive")
        alpha = torch.linalg.solve(k_train + float(reg) * eye, signed_y)
        predictions = k_val @ alpha
        score = _bacc(predictions, y_val)
        if score > best_val:
            best_val, best_reg, best_alpha = score, float(reg), alpha
    assert best_alpha is not None
    test_predictions = k_test @ best_alpha
    test_score = _bacc(test_predictions, y_test)
    test_accuracy = float(((test_predictions >= 0.0) == (y_test >= 0.5)).float().mean().item())
    if actual_device == "cuda":
        torch.cuda.synchronize()
    return {
        "method": "quantum_fidelity_kernel_ridge",
        "seed": int(seed),
        "device": actual_device,
        "status": "ok",
        "config": {"n_qubits": n_qubits, "feature_map": feature_map, "regularization_grid": list(regularization), "selected_regularization": best_reg},
        "val_balanced_accuracy": float(best_val),
        "test": {"balanced_accuracy": float(test_score), "accuracy": test_accuracy, "n": len(test_y)},
        "elapsed_seconds": time.perf_counter() - started,
        "simulator": "ideal_exact_statevector_fidelity_kernel",
        "physical_qpu": False,
    }


def train_rbf_kernel_baseline(
    train_x: list[list[float]], train_y: list[float],
    val_x: list[list[float]], val_y: list[float],
    test_x: list[list[float]], test_y: list[float],
    *, gamma: float = 0.5, regularization: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
) -> dict[str, Any]:
    """CPU RBF kernel-ridge baseline used for a classical kernel control."""
    torch = load_torch()
    torch.set_num_threads(1)
    started = time.perf_counter()
    xt = torch.tensor(train_x, dtype=torch.float64)
    xv = torch.tensor(val_x, dtype=torch.float64)
    xe = torch.tensor(test_x, dtype=torch.float64)
    yt = torch.tensor(train_y, dtype=torch.float64)
    yv = torch.tensor(val_y, dtype=torch.float64)
    ye = torch.tensor(test_y, dtype=torch.float64)

    def kernel(a: Any, b: Any) -> Any:
        d2 = (a.square().sum(1, keepdim=True) + b.square().sum(1).unsqueeze(0) - 2.0 * a @ b.T).clamp_min(0.0)
        return torch.exp(-float(gamma) * d2)

    k_train = kernel(xt, xt)
    k_val = kernel(xv, xt)
    k_test = kernel(xe, xt)
    signed = yt * 2.0 - 1.0
    eye = torch.eye(len(xt), dtype=torch.float64)
    best_score = -math.inf
    best_alpha = None
    best_reg = None
    for reg in regularization:
        alpha = torch.linalg.solve(k_train + float(reg) * eye, signed)
        pred = k_val @ alpha
        score = _bacc(pred, yv)
        if score > best_score:
            best_score, best_alpha, best_reg = score, alpha, float(reg)
    assert best_alpha is not None
    test_pred = k_test @ best_alpha
    test_score = _bacc(test_pred, ye)
    accuracy = float(((test_pred >= 0.0) == (ye >= 0.5)).float().mean().item())
    return {
        "method": "rbf_kernel_ridge_cpu_baseline",
        "device": "cpu",
        "status": "ok",
        "config": {"gamma": float(gamma), "regularization_grid": list(regularization), "selected_regularization": best_reg},
        "val_balanced_accuracy": float(best_score),
        "test": {"balanced_accuracy": float(test_score), "accuracy": accuracy, "n": len(test_y)},
        "elapsed_seconds": time.perf_counter() - started,
        "physical_qpu": False,
    }
