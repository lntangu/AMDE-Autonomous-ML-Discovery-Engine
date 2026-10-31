"""Dependency-free trainer for linear models and tiny MLPs.

The trainer is deliberately small, deterministic, and transparent. It supports
the Level A/B/C recipe fields and Level D's safe per-parameter update DSL.
All data and hyperparameters are fixed by the evaluator, not by the candidate.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from ..discovery import dsl
from ..discovery.genome import Genome
from ..errors import CandidateNumericError, ValidationError
from ..evaluation.metrics import classification_score, regression_score
from ..util import seed_from


@dataclass
class TrainData:
    x: list[list[float]]
    y: list[float]


@dataclass
class Layer:
    W: list[list[float]]
    b: list[float]
    activation: str | None


def _act(value: float, name: str) -> float:
    if name == "tanh":
        return math.tanh(value)
    if name == "relu":
        return value if value > 0.0 else 0.0
    if name == "sigmoid":
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
    raise ValueError(f"unknown activation {name}")


def _act_grad_from_output(output: float, name: str) -> float:
    if name == "tanh":
        return 1.0 - output * output
    if name == "relu":
        return 1.0 if output > 0.0 else 0.0
    if name == "sigmoid":
        return output * (1.0 - output)
    raise ValueError(f"unknown activation {name}")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


def _schedule_scale(schedule: str, step: int, total_steps: int) -> float:
    """Deterministic step schedule shared by the reference and tensor trainers."""
    progress = min(1.0, max(0.0, (max(1, int(step)) - 1) / max(1, int(total_steps) - 1)))
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return max(0.0, 1.0 - progress)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "warmup_cosine":
        warmup = 0.10
        if progress < warmup:
            return max(0.05, progress / warmup)
        tail = (progress - warmup) / (1.0 - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, tail))))
    raise ValidationError(f"unknown learning-rate schedule {schedule!r}", code="lr_schedule")


def _check_finite(value: float, where: str) -> float:
    if not math.isfinite(value) or abs(value) > 1e100:
        raise CandidateNumericError(f"non-finite value in {where}")
    return value


def _prepare_features(
    train_x: Sequence[Sequence[float]],
    val_x: Sequence[Sequence[float]],
    dev_x: Sequence[Sequence[float]],
    *,
    standardize: bool,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], dict[str, Any]]:
    n_features = len(train_x[0]) if train_x else 0
    if standardize:
        means = [0.0] * n_features
        for x in train_x:
            for i, value in enumerate(x):
                means[i] += value
        means = [value / max(1, len(train_x)) for value in means]
        variances = [0.0] * n_features
        for x in train_x:
            for i, value in enumerate(x):
                variances[i] += (value - means[i]) ** 2
        stds = [max(1e-8, math.sqrt(value / max(1, len(train_x)))) for value in variances]
    else:
        means = [0.0] * n_features
        stds = [1.0] * n_features

    def transform(rows: Sequence[Sequence[float]]) -> list[list[float]]:
        result = []
        for row in rows:
            transformed = []
            for i, value in enumerate(row):
                v = (value - means[i]) / stds[i] if standardize else value
                if not math.isfinite(v):
                    raise CandidateNumericError("non-finite feature after preprocessing")
                # This is a numerical guard, not a learned feature transform.
                transformed.append(max(-20.0, min(20.0, v)))
            result.append(transformed)
        return result

    return (
        transform(train_x),
        transform(val_x),
        transform(dev_x),
        {"standardize": standardize, "means": means, "stds": stds},
    )


def _init_layers(
    architecture: dict[str, Any],
    n_features: int,
    recipe: dict[str, Any],
    rng: random.Random,
) -> list[Layer]:
    widths = [int(n_features)] + [int(width) for width in architecture["hidden"]] + [1]
    layers: list[Layer] = []
    for idx in range(len(widths) - 1):
        fan_in, fan_out = widths[idx], widths[idx + 1]
        scale = float(recipe["init_scale"]) / math.sqrt(max(1, fan_in))
        W: list[list[float]] = []
        for _ in range(fan_out):
            row = []
            for _ in range(fan_in):
                if recipe["init"] == "zeros":
                    value = 0.0
                elif recipe["init"] == "uniform":
                    value = rng.uniform(-scale, scale)
                else:
                    value = rng.gauss(0.0, scale)
                row.append(value)
            W.append(row)
        b = [0.0] * fan_out
        activation = architecture["activation"] if idx < len(widths) - 2 else None
        layers.append(Layer(W=W, b=b, activation=activation))
    return layers


def _forward(layers: list[Layer], x: list[float]) -> tuple[float, dict[str, Any]]:
    current = list(x)
    inputs: list[list[float]] = []
    outputs: list[list[float]] = []
    zs: list[list[float]] = []
    for layer in layers:
        if len(current) != len(layer.W[0]):
            raise CandidateNumericError("feature/layer shape mismatch")
        inputs.append(current)
        z = []
        for j, row in enumerate(layer.W):
            value = layer.b[j]
            for weight, feature in zip(row, current):
                value += weight * feature
            z.append(_check_finite(value, "layer pre-activation"))
        zs.append(z)
        if layer.activation is None:
            out = z
        else:
            out = [_check_finite(_act(v, layer.activation), "activation") for v in z]
        outputs.append(out)
        current = out
    if len(current) != 1:
        raise CandidateNumericError("model output is not scalar")
    return current[0], {"inputs": inputs, "outputs": outputs, "zs": zs}


def _loss_and_dout(
    prediction: float,
    target: float,
    task_kind: str,
    loss: str,
) -> tuple[float, float]:
    if task_kind == "regression":
        diff = prediction - target
        return diff * diff, 2.0 * diff
    p = _sigmoid(prediction)
    if loss == "mse":
        diff = p - target
        return diff * diff, 2.0 * diff * p * (1.0 - p)
    # Numerically stable binary cross-entropy from logits.
    if prediction >= 0:
        loss_value = prediction - target * prediction + math.log1p(math.exp(-prediction))
    else:
        loss_value = -target * prediction + math.log1p(math.exp(prediction))
    return loss_value, p - target


def _backward(
    layers: list[Layer],
    cache: dict[str, Any],
    dout: float,
) -> tuple[list[list[list[float]]], list[list[float]]]:
    grads_W: list[list[list[float]]] = []
    grads_b: list[list[float]] = []
    for layer in layers:
        grads_W.append([[0.0 for _ in row] for row in layer.W])
        grads_b.append([0.0 for _ in layer.b])
    delta = [dout]
    for idx in range(len(layers) - 1, -1, -1):
        layer = layers[idx]
        inp = cache["inputs"][idx]
        for j, row in enumerate(layer.W):
            dj = delta[j]
            grads_b[idx][j] += dj
            grow = grads_W[idx][j]
            for i, feature in enumerate(inp):
                grow[i] += dj * feature
        if idx > 0:
            previous = layers[idx - 1]
            z_prev = cache["zs"][idx - 1]
            out_prev = cache["outputs"][idx - 1]
            new_delta = [0.0] * len(previous.b)
            for i in range(len(previous.b)):
                total = 0.0
                for j, row in enumerate(layer.W):
                    total += row[i] * delta[j]
                new_delta[i] = total * _act_grad_from_output(out_prev[i], previous.activation or "tanh")
            delta = new_delta
    return grads_W, grads_b


def _flat_parameters(layers: list[Layer]) -> list[float]:
    values: list[float] = []
    for layer in layers:
        for row in layer.W:
            values.extend(row)
        values.extend(layer.b)
    return values


def _check_layers_finite(layers: list[Layer]) -> None:
    for layer in layers:
        for row in layer.W:
            for value in row:
                _check_finite(value, "weight")
        for value in layer.b:
            _check_finite(value, "bias")


def _clip_gradients(grads_W: list[list[list[float]]], grads_b: list[list[float]], clip: float) -> float:
    if clip <= 0:
        total = 0.0
        for matrix in grads_W:
            for row in matrix:
                for value in row:
                    total += value * value
        for vec in grads_b:
            for value in vec:
                total += value * value
        return math.sqrt(total)
    total = 0.0
    for matrix in grads_W:
        for row in matrix:
            for value in row:
                total += value * value
    for vec in grads_b:
        for value in vec:
            total += value * value
    norm = math.sqrt(total)
    if norm > clip and norm > 0:
        factor = clip / norm
        for matrix in grads_W:
            for row in matrix:
                for i in range(len(row)):
                    row[i] *= factor
        for vec in grads_b:
            for i in range(len(vec)):
                vec[i] *= factor
        norm = clip
    return norm


class OptimizerState:
    def __init__(self, layers: list[Layer], kind: str) -> None:
        self.kind = kind
        self.m_W = [[[0.0 for _ in row] for row in layer.W] for layer in layers]
        self.v_W = [[[0.0 for _ in row] for row in layer.W] for layer in layers]
        self.m_b = [[0.0 for _ in layer.b] for layer in layers]
        self.v_b = [[0.0 for _ in layer.b] for layer in layers]
        self.step = 0


def _dsl_value(expr: dict[str, Any], variables: dict[str, float]) -> float:
    return dsl.evaluate(expr, variables)


def _apply_updates(
    layers: list[Layer],
    grads_W: list[list[list[float]]],
    grads_b: list[list[float]],
    genome_data: dict[str, Any],
    state: OptimizerState,
    grad_norm: float,
    batch_size: int,
    effective_lr: float | None = None,
) -> None:
    recipe = genome_data["recipe"]
    opt = genome_data["optimizer"]
    lr = float(effective_lr if effective_lr is not None else recipe["learning_rate"])
    l2 = float(recipe["l2"])
    l1 = float(recipe["l1"])
    kind = opt["kind"]
    beta1 = float(opt["beta1"])
    beta2 = float(opt["beta2"])
    eps = float(opt["epsilon"])
    state.step += 1
    t = state.step

    for layer_idx, layer in enumerate(layers):
        for j, row in enumerate(layer.W):
            for i in range(len(row)):
                g = grads_W[layer_idx][j][i] / batch_size
                p = row[i]
                if kind in ("adamw", "lion"):
                    g += l1 * (1.0 if p > 0 else (-1.0 if p < 0 else 0.0))
                else:
                    g += l2 * p + l1 * (1.0 if p > 0 else (-1.0 if p < 0 else 0.0))
                if not math.isfinite(g):
                    raise CandidateNumericError("non-finite gradient")
                if kind == "sgd":
                    new_p = p - lr * g
                elif kind == "momentum":
                    m = beta1 * state.m_W[layer_idx][j][i] + g
                    state.m_W[layer_idx][j][i] = m
                    new_p = p - lr * m
                elif kind == "adagrad":
                    v = state.v_W[layer_idx][j][i] + g * g
                    state.v_W[layer_idx][j][i] = v
                    new_p = p - lr * g / (math.sqrt(v) + eps)
                elif kind == "rmsprop":
                    v = beta2 * state.v_W[layer_idx][j][i] + (1.0 - beta2) * g * g
                    state.v_W[layer_idx][j][i] = v
                    new_p = p - lr * g / (math.sqrt(v) + eps)
                elif kind in ("adam", "nadam", "adamw"):
                    m = beta1 * state.m_W[layer_idx][j][i] + (1.0 - beta1) * g
                    v = beta2 * state.v_W[layer_idx][j][i] + (1.0 - beta2) * g * g
                    state.m_W[layer_idx][j][i] = m
                    state.v_W[layer_idx][j][i] = v
                    m_hat = m / (1.0 - beta1**t)
                    v_hat = v / (1.0 - beta2**t)
                    if kind == "nadam":
                        # Dozat-style NAdam: combine the bias-corrected
                        # momentum and current-gradient terms separately.
                        m_hat = (
                            beta1 * m / (1.0 - beta1 ** (t + 1))
                            + (1.0 - beta1) * g / (1.0 - beta1**t)
                        )
                    new_p = p - lr * m_hat / (math.sqrt(v_hat) + eps)
                    if kind == "adamw":
                        new_p -= lr * l2 * p
                elif kind == "lion":
                    previous_m = state.m_W[layer_idx][j][i]
                    direction = beta1 * previous_m + (1.0 - beta1) * g
                    state.m_W[layer_idx][j][i] = beta2 * previous_m + (1.0 - beta2) * g
                    sign_direction = 1.0 if direction > 0.0 else (-1.0 if direction < 0.0 else 0.0)
                    new_p = p - lr * sign_direction - lr * l2 * p
                elif kind == "dsl":
                    variables = {
                        "g": g,
                        "p": p,
                        "w": p,
                        "b": p,
                        "m": state.m_W[layer_idx][j][i],
                        "v": state.v_W[layer_idx][j][i],
                        "t": float(t),
                        "lr": lr,
                        "gn": float(grad_norm),
                    }
                    updates = genome_data["dsl"]
                    if updates["m_update"] is not None:
                        state.m_W[layer_idx][j][i] = _dsl_value(updates["m_update"], variables)
                    if updates["v_update"] is not None:
                        state.v_W[layer_idx][j][i] = _dsl_value(updates["v_update"], variables)
                    new_p = _dsl_value(updates["w_update"], variables)
                else:
                    raise ValidationError(f"unknown optimizer {kind}", code="optimizer")
                row[i] = _check_finite(new_p, "updated weight")
        for j in range(len(layer.b)):
            g = grads_b[layer_idx][j] / batch_size
            p = layer.b[j]
            if kind in ("adamw", "lion"):
                g += l1 * (1.0 if p > 0 else (-1.0 if p < 0 else 0.0))
            else:
                g += l2 * p + l1 * (1.0 if p > 0 else (-1.0 if p < 0 else 0.0))
            if not math.isfinite(g):
                raise CandidateNumericError("non-finite bias gradient")
            if kind == "sgd":
                new_p = p - lr * g
            elif kind == "momentum":
                m = beta1 * state.m_b[layer_idx][j] + g
                state.m_b[layer_idx][j] = m
                new_p = p - lr * m
            elif kind == "adagrad":
                v = state.v_b[layer_idx][j] + g * g
                state.v_b[layer_idx][j] = v
                new_p = p - lr * g / (math.sqrt(v) + eps)
            elif kind == "rmsprop":
                v = beta2 * state.v_b[layer_idx][j] + (1.0 - beta2) * g * g
                state.v_b[layer_idx][j] = v
                new_p = p - lr * g / (math.sqrt(v) + eps)
            elif kind in ("adam", "nadam", "adamw"):
                m = beta1 * state.m_b[layer_idx][j] + (1.0 - beta1) * g
                v = beta2 * state.v_b[layer_idx][j] + (1.0 - beta2) * g * g
                state.m_b[layer_idx][j] = m
                state.v_b[layer_idx][j] = v
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)
                if kind == "nadam":
                    m_hat = (
                        beta1 * m / (1.0 - beta1 ** (t + 1))
                        + (1.0 - beta1) * g / (1.0 - beta1**t)
                    )
                new_p = p - lr * m_hat / (math.sqrt(v_hat) + eps)
                if kind == "adamw":
                    new_p -= lr * l2 * p
            elif kind == "lion":
                previous_m = state.m_b[layer_idx][j]
                direction = beta1 * previous_m + (1.0 - beta1) * g
                state.m_b[layer_idx][j] = beta2 * previous_m + (1.0 - beta2) * g
                sign_direction = 1.0 if direction > 0.0 else (-1.0 if direction < 0.0 else 0.0)
                new_p = p - lr * sign_direction - lr * l2 * p
            elif kind == "dsl":
                variables = {
                    "g": g,
                    "p": p,
                    "w": p,
                    "b": p,
                    "m": state.m_b[layer_idx][j],
                    "v": state.v_b[layer_idx][j],
                    "t": float(t),
                    "lr": lr,
                    "gn": float(grad_norm),
                }
                updates = genome_data["dsl"]
                if updates["m_update"] is not None:
                    state.m_b[layer_idx][j] = _dsl_value(updates["m_update"], variables)
                if updates["v_update"] is not None:
                    state.v_b[layer_idx][j] = _dsl_value(updates["v_update"], variables)
                new_p = _dsl_value(updates["b_update"], variables)
            else:
                raise ValidationError(f"unknown optimizer {kind}", code="optimizer")
            layer.b[j] = _check_finite(new_p, "updated bias")


def _predict(layers: list[Layer], x: list[list[float]], task_kind: str) -> list[float]:
    result = []
    for row in x:
        output, _ = _forward(layers, row)
        result.append(_sigmoid(output) if task_kind == "classification" else output)
    return result


def _validation_metric(layers: list[Layer], data: TrainData, task_kind: str) -> float:
    preds = _predict(layers, data.x, task_kind)
    if task_kind == "classification":
        return float(classification_score(data.y, preds)["score"])
    # Use raw R2 (not clipped) for checkpoint selection.
    mu = sum(data.y) / max(1, len(data.y))
    sse = sum((a - b) ** 2 for a, b in zip(data.y, preds))
    sst = sum((a - mu) ** 2 for a in data.y)
    return 1.0 - sse / sst if sst > 1e-15 else (1.0 if sse < 1e-15 else 0.0)


def train_and_predict(
    genome: Genome,
    *,
    train_x: list[list[float]],
    train_y: list[float],
    val_x: list[list[float]],
    val_y: list[float],
    dev_x: list[list[float]],
    task_kind: str,
    eval_seed: int,
    task_id: str,
    epochs_scale: float = 1.0,
    max_steps: int = 1200,
    parameter_limit: int = 20000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one candidate on one task/seed and return dev predictions.

    The dependency-free reference implementation remains the CPU default. A
    CUDA request is dispatched to the optional vectorized PyTorch backend;
    the caller must have successfully probed CUDA before selecting it.
    """
    if str(device).startswith("cuda"):
        from .torch_trainer import train_and_predict_torch

        return train_and_predict_torch(
            genome,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            dev_x=dev_x,
            task_kind=task_kind,
            eval_seed=eval_seed,
            task_id=task_id,
            epochs_scale=epochs_scale,
            max_steps=max_steps,
            parameter_limit=parameter_limit,
            device=device,
        )

    data = genome.data
    recipe = data["recipe"]
    architecture = data["model"]
    n_features = len(train_x[0]) if train_x else 0
    params = genome.parameter_estimate(n_features)
    if params > parameter_limit:
        raise ValidationError(
            f"candidate has {params} parameters; limit is {parameter_limit}",
            code="parameter_limit",
        )
    if len(train_x) < 1 or len(val_x) < 1 or len(dev_x) < 1:
        raise ValidationError("empty train/val/dev data", code="empty_split")

    train_x_p, val_x_p, dev_x_p, prep = _prepare_features(
        train_x, val_x, dev_x, standardize=bool(recipe["standardize"])
    )
    train_data = TrainData(train_x_p, list(train_y))
    val_data = TrainData(val_x_p, list(val_y))

    if recipe["seed_policy"] == "constant":
        init_seed = seed_from(genome.hash, task_id, "constant-init")
    else:
        init_seed = seed_from(genome.hash, task_id, eval_seed, "init")
    init_rng = random.Random(init_seed)
    layers = _init_layers(architecture, n_features, recipe, init_rng)
    shuffle_rng = random.Random(seed_from(genome.hash, task_id, eval_seed, "shuffle"))

    effective_epochs = max(2, int(round(int(recipe["epochs"]) * max(0.05, float(epochs_scale)))))
    batch_size = min(int(recipe["batch_size"]), len(train_data.x))
    batches_per_epoch = max(1, math.ceil(len(train_data.x) / batch_size))
    planned_steps = max(1, min(int(max_steps), effective_epochs * batches_per_epoch))
    state = OptimizerState(layers, str(data["optimizer"]["kind"]))
    best_layers = copy.deepcopy(layers)
    best_metric = -float("inf")
    patience = int(recipe["early_stopping_patience"])
    stale = 0
    step = 0
    truncated = False
    total_loss = 0.0

    for epoch in range(effective_epochs):
        indices = list(range(len(train_data.x)))
        shuffle_rng.shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, len(indices), batch_size):
            if step >= max_steps:
                truncated = True
                break
            batch = indices[start : start + batch_size]
            grads_W: list[list[list[float]]] = [
                [[0.0 for _ in row] for row in layer.W] for layer in layers
            ]
            grads_b: list[list[float]] = [[0.0 for _ in layer.b] for layer in layers]
            batch_loss = 0.0
            for idx in batch:
                prediction, cache = _forward(layers, train_data.x[idx])
                loss_value, dout = _loss_and_dout(
                    prediction, float(train_data.y[idx]), task_kind, str(recipe["loss"])
                )
                if not math.isfinite(loss_value) or not math.isfinite(dout):
                    raise CandidateNumericError("non-finite loss or gradient seed")
                batch_loss += loss_value
                gW, gb = _backward(layers, cache, dout)
                for li in range(len(layers)):
                    for j in range(len(grads_W[li])):
                        for i in range(len(grads_W[li][j])):
                            grads_W[li][j][i] += gW[li][j][i]
                    for j in range(len(grads_b[li])):
                        grads_b[li][j] += gb[li][j]
            grad_norm = _clip_gradients(grads_W, grads_b, float(recipe["grad_clip"]))
            lr_scale = _schedule_scale(str(recipe["lr_schedule"]), step + 1, planned_steps)
            _apply_updates(
                layers, grads_W, grads_b, data, state, grad_norm, len(batch),
                effective_lr=float(recipe["learning_rate"]) * lr_scale,
            )
            _check_layers_finite(layers)
            step += 1
            epoch_loss += batch_loss / max(1, len(batch))
        total_loss += epoch_loss / max(1, math.ceil(len(indices) / batch_size))
        metric = _validation_metric(layers, val_data, task_kind)
        if metric > best_metric + 1e-12:
            best_metric = metric
            best_layers = copy.deepcopy(layers)
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
        if truncated:
            break
    layers = best_layers
    dev_predictions = _predict(layers, dev_x_p, task_kind)
    if not all(math.isfinite(value) for value in dev_predictions):
        raise CandidateNumericError("non-finite dev prediction")
    return {
        "predictions": dev_predictions,
        "val_metric": best_metric,
        "epochs_effective": effective_epochs,
        "steps": step,
        "truncated": truncated,
        "parameter_count": params,
        "preprocessing": prep,
        "train_loss_last": total_loss / max(1, effective_epochs),
    }


def baseline_majority_mean(
    *,
    train_x: list[list[float]],
    train_y: list[float],
    dev_x: list[list[float]],
    task_kind: str,
) -> list[float]:
    if not train_y:
        raise ValidationError("empty training data", code="empty_split")
    if task_kind == "classification":
        positives = sum(1 for y in train_y if y >= 0.5)
        value = 1.0 if positives * 2 >= len(train_y) else 0.0
        return [value] * len(dev_x)
    value = sum(train_y) / len(train_y)
    return [float(value)] * len(dev_x)
