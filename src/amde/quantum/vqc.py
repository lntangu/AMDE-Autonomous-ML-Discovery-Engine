"""Variational quantum classifier over an exact PyTorch statevector simulator."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from .statevector import (
    StatevectorBudget,
    apply_cnot,
    apply_ry,
    apply_rz,
    expectation_z,
    initial_state,
    load_torch,
    resolve_device,
)


@dataclass(frozen=True)
class VQCConfig:
    n_qubits: int = 4
    n_layers: int = 2
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 0.02
    gradient_method: str = "autograd"
    optimizer: str = "adam"
    schedule: str = "constant"
    feature_map: str = "angle_ry"
    entanglement: str = "ring"
    logit_scale: float = 2.5
    max_statevector_mb: int = 256

    def validate(self) -> None:
        if not 2 <= self.n_qubits <= 12:
            raise ValueError("VQC n_qubits must be between 2 and 12")
        if not 1 <= self.n_layers <= 8:
            raise ValueError("VQC n_layers must be between 1 and 8")
        if not 1 <= self.epochs <= 1000:
            raise ValueError("VQC epochs must be between 1 and 1000")
        if not 1 <= self.batch_size <= 4096:
            raise ValueError("VQC batch_size must be between 1 and 4096")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.gradient_method not in {"autograd", "parameter_shift"}:
            raise ValueError("gradient_method must be autograd or parameter_shift")
        if self.optimizer not in {"adam", "adamw", "sgd", "rmsprop"}:
            raise ValueError("optimizer must be adam, adamw, sgd, or rmsprop")
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError("schedule must be constant or cosine")
        if self.feature_map not in {"angle_ry", "angle_ry_rz"}:
            raise ValueError("feature_map must be angle_ry or angle_ry_rz")
        if self.entanglement not in {"linear", "ring"}:
            raise ValueError("entanglement must be linear or ring")
        if self.max_statevector_mb < 1:
            raise ValueError("max_statevector_mb must be positive")


class VariationalQuantumClassifier:
    """A small hardware-inspired angle-encoded variational circuit classifier.

    This is an ideal statevector simulator model, not an execution interface to
    physical quantum hardware. Features are angle encoded; trainable RY/RZ
    rotations are entangled by a nearest-neighbour CNOT ring.
    """

    def __init__(self, config: VQCConfig, *, device: str = "cpu", seed: int = 1) -> None:
        config.validate()
        torch = load_torch()
        self.torch = torch
        self.config = config
        self.device = resolve_device(device)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.theta = torch.nn.Parameter(
            torch.empty((config.n_layers, config.n_qubits, 2), dtype=torch.float32, device=self.device)
            .uniform_(-0.15, 0.15)
        )
        self.budget = StatevectorBudget(
            max_qubits=config.n_qubits,
            max_statevector_mb=config.max_statevector_mb,
            dtype="complex64",
        )

    def expectations(self, features: Any, theta: Any | None = None) -> Any:
        torch = self.torch
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[1] != self.config.n_qubits:
            raise ValueError(f"features must have shape [batch, {self.config.n_qubits}]")
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError("features contain non-finite values")
        x = torch.clamp(x, -1.0, 1.0) * math.pi
        params = self.theta if theta is None else theta
        if tuple(params.shape) != tuple(self.theta.shape):
            raise ValueError("theta override has the wrong shape")
        self.budget.validate(self.config.n_qubits, batch_size=int(x.shape[0]))
        state = initial_state(
            self.config.n_qubits,
            batch_size=int(x.shape[0]),
            device=self.device,
            budget=self.budget,
        )
        # Angle feature map; every feature is encoded as a rotation.
        for qubit in range(self.config.n_qubits):
            state = apply_ry(state, self.config.n_qubits, qubit, x[:, qubit])
            if self.config.feature_map == "angle_ry_rz":
                state = apply_rz(state, self.config.n_qubits, qubit, x[:, qubit] * 0.5)
        for qubit in range(self.config.n_qubits - 1):
            state = apply_cnot(state, self.config.n_qubits, qubit, qubit + 1)
        if self.config.entanglement == "ring" and self.config.n_qubits > 2:
            state = apply_cnot(state, self.config.n_qubits, self.config.n_qubits - 1, 0)
        for layer in range(self.config.n_layers):
            for qubit in range(self.config.n_qubits):
                state = apply_ry(state, self.config.n_qubits, qubit, params[layer, qubit, 0])
                state = apply_rz(state, self.config.n_qubits, qubit, params[layer, qubit, 1])
            for qubit in range(self.config.n_qubits - 1):
                state = apply_cnot(state, self.config.n_qubits, qubit, qubit + 1)
            if self.config.entanglement == "ring" and self.config.n_qubits > 2:
                state = apply_cnot(state, self.config.n_qubits, self.config.n_qubits - 1, 0)
        return expectation_z(state, self.config.n_qubits, 0)

    def logits(self, features: Any, theta: Any | None = None) -> Any:
        return self.config.logit_scale * self.expectations(features, theta)

    def parameter_shift_grad(self, features: Any, labels: Any) -> tuple[Any, float]:
        """Calculate exact parameter-shift gradients for all Pauli rotations.

        Each trainable angle appears in exactly one RY/RZ gate with generator
        eigenvalues +/-1/2, so the two-point shift rule is exact here.
        """
        torch = self.torch
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device).reshape(-1)
        params = self.theta.detach()
        logits = self.logits(x, params).detach().requires_grad_(True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        dloss_dlogits = torch.autograd.grad(loss, logits)[0].detach()
        dloss_dexpectation = dloss_dlogits * self.config.logit_scale
        gradient = torch.zeros_like(params)
        shift = math.pi / 2.0
        with torch.no_grad():
            for index in range(params.numel()):
                plus = params.clone().reshape(-1)
                minus = params.clone().reshape(-1)
                plus[index] += shift
                minus[index] -= shift
                e_plus = self.expectations(x, plus.reshape_as(params))
                e_minus = self.expectations(x, minus.reshape_as(params))
                derivative = 0.5 * (e_plus - e_minus)
                gradient.reshape(-1)[index] = torch.sum(dloss_dexpectation * derivative)
        return gradient, float(loss.item())


def train_vqc(
    train_x: list[list[float]],
    train_y: list[float],
    val_x: list[list[float]],
    val_y: list[float],
    test_x: list[list[float]],
    test_y: list[float],
    *,
    config: VQCConfig,
    device: str = "auto",
    seed: int = 1,
    patience: int = 5,
) -> dict[str, Any]:
    torch = load_torch()
    config.validate()
    actual_device = resolve_device(device)
    # The quantum worker is deliberately thread-limited so it can coexist with
    # the classical CPU campaign on a 6-core laptop.
    torch.set_num_threads(1)
    model = VariationalQuantumClassifier(config, device=actual_device, seed=seed)
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=actual_device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=actual_device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=actual_device)
    y_val = torch.as_tensor(val_y, dtype=torch.float32, device=actual_device)
    x_test = torch.as_tensor(test_x, dtype=torch.float32, device=actual_device)
    y_test = torch.as_tensor(test_y, dtype=torch.float32, device=actual_device)
    if x_train.ndim != 2 or x_train.shape[1] != config.n_qubits:
        raise ValueError("training feature shape does not match n_qubits")
    if len(x_train) < 2 or len(x_val) < 1 or len(x_test) < 1:
        raise ValueError("train/validation/test splits must be non-empty")
    if len(set(float(v) for v in train_y)) != 2:
        raise ValueError("binary training split must contain both classes")
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam([model.theta], lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW([model.theta], lr=config.learning_rate, weight_decay=1e-4)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD([model.theta], lr=config.learning_rate, momentum=0.8)
    else:
        optimizer = torch.optim.RMSprop([model.theta], lr=config.learning_rate, alpha=0.9)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x51A7)
    best_score = -math.inf
    best_params = model.theta.detach().clone()
    stale = 0
    loss_history: list[float] = []
    if actual_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for _epoch in range(config.epochs):
        if config.schedule == "cosine":
            lr_now = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * _epoch / max(1, config.epochs)))
            for group in optimizer.param_groups:
                group["lr"] = max(lr_now, config.learning_rate * 0.02)
        permutation = torch.randperm(len(x_train), generator=generator)
        if actual_device == "cuda":
            permutation = permutation.to(actual_device)
        epoch_loss_acc = torch.zeros(1, dtype=torch.float32, device=actual_device)
        n_batches = 0
        model.theta.grad = None
        for start in range(0, len(x_train), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            batch_x = x_train.index_select(0, indices)
            batch_y = y_train.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            if config.gradient_method == "parameter_shift":
                gradient, loss_value = model.parameter_shift_grad(batch_x, batch_y)
                model.theta.grad = gradient
                loss = torch.as_tensor(loss_value, device=actual_device)
            else:
                logits = model.logits(batch_x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
                loss.backward()
            if not bool(torch.isfinite(model.theta.grad).all().item()):
                raise RuntimeError("VQC gradient became non-finite")
            torch.nn.utils.clip_grad_norm_([model.theta], max_norm=5.0)
            optimizer.step()
            epoch_loss_acc += loss.detach().reshape(-1)
            n_batches += 1
        loss_history.append(float((epoch_loss_acc / max(1, n_batches)).item()))
        val_score = _balanced_accuracy(model.logits(x_val).detach(), y_val)
        if val_score > best_score + 1e-12:
            best_score = val_score
            best_params = model.theta.detach().clone()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.theta.data.copy_(best_params)
    test_logits = model.logits(x_test).detach()
    test_score = _classification_metrics(test_logits, y_test)
    if actual_device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated() if actual_device == "cuda" else 0)
    return {
        "method": "vqc_statevector",
        "config": {
            "n_qubits": config.n_qubits,
            "n_layers": config.n_layers,
            "epochs_requested": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gradient_method": config.gradient_method,
            "optimizer": config.optimizer,
            "schedule": config.schedule,
            "feature_map": config.feature_map,
            "entanglement": config.entanglement,
            "logit_scale": config.logit_scale,
        },
        "seed": int(seed),
        "device": actual_device,
        "status": "ok",
        "val_balanced_accuracy": float(best_score),
        "test": test_score,
        "epochs_completed": len(loss_history),
        "loss_history": loss_history,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_bytes": peak_allocated,
        "simulator": "ideal_exact_statevector",
        "n_trainable_parameters": int(model.theta.numel()),
        "statevector_dimension": 1 << config.n_qubits,
        "physical_qpu": False,
    }


def train_mlp_baseline(
    train_x: list[list[float]],
    train_y: list[float],
    val_x: list[list[float]],
    val_y: list[float],
    test_x: list[list[float]],
    test_y: list[float],
    *,
    epochs: int = 15,
    seed: int = 1,
) -> dict[str, Any]:
    """Small classical baseline on CPU, with a one-thread CPU budget."""
    torch = load_torch()
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(len(train_x[0]), 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 1),
    )
    x_train = torch.as_tensor(train_x, dtype=torch.float32)
    y_train = torch.as_tensor(train_y, dtype=torch.float32)
    x_val = torch.as_tensor(val_x, dtype=torch.float32)
    y_val = torch.as_tensor(val_y, dtype=torch.float32)
    x_test = torch.as_tensor(test_x, dtype=torch.float32)
    y_test = torch.as_tensor(test_y, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_score = -math.inf
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        with torch.no_grad():
            val_logits = model(x_val).squeeze(-1)
            val_score = _balanced_accuracy(val_logits, y_val)
        if val_score > best_score + 1e-12:
            best_score = val_score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_logits = model(x_test).squeeze(-1)
    return {
        "method": "mlp_cpu_baseline",
        "seed": int(seed),
        "device": "cpu",
        "status": "ok",
        "val_balanced_accuracy": float(best_score),
        "test": _classification_metrics(test_logits, y_test),
        "epochs_completed": epochs,
        "loss_history": losses,
        "elapsed_seconds": time.perf_counter() - started,
        "physical_qpu": False,
    }


def _balanced_accuracy(logits: Any, labels: Any) -> float:
    torch = load_torch()
    prediction = logits >= 0.0
    positive = labels >= 0.5
    recall_pos = ((prediction & positive).float().sum() / positive.float().sum().clamp_min(1.0))
    negative = ~positive
    recall_neg = ((~prediction) & negative).float().sum() / negative.float().sum().clamp_min(1.0)
    return float((0.5 * (recall_pos + recall_neg)).item())


def _classification_metrics(logits: Any, labels: Any) -> dict[str, float]:
    torch = load_torch()
    prediction = logits >= 0.0
    positive = labels >= 0.5
    negative = ~positive
    tp = (prediction & positive).float().sum()
    tn = ((~prediction) & negative).float().sum()
    p = positive.float().sum().clamp_min(1.0)
    n = negative.float().sum().clamp_min(1.0)
    bacc = 0.5 * (tp / p + tn / n)
    accuracy = (prediction == positive).float().mean()
    return {
        "balanced_accuracy": float(bacc.item()),
        "accuracy": float(accuracy.item()),
        "n": int(labels.numel()),
    }
