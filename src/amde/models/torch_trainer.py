"""Optional vectorized PyTorch trainer used only after a successful CUDA probe.

The candidate still supplies JSON genome data only. No candidate Python is
executed. Optimizer/update-rule semantics mirror the dependency-free reference
trainer; only tensor arithmetic and the forward/backward pass are accelerated.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

from ..discovery import dsl
from ..discovery.genome import Genome
from ..errors import CandidateNumericError, ValidationError
from ..evaluation.metrics import classification_score
from ..util import seed_from
from . import trainer as reference


def _as_tensor(torch: Any, value: Any, *, device: Any, dtype: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _eval_expr(torch: Any, expr: dict[str, Any], variables: dict[str, Any], *, device: Any, dtype: Any) -> Any:
    """Vectorized interpreter for the exact same allow-listed DSL AST."""
    op = expr["op"]
    if op == "const":
        return torch.as_tensor(float(expr["value"]), device=device, dtype=dtype)
    if op == "var":
        if expr["name"] not in variables:
            raise ValidationError(f"missing DSL variable {expr['name']!r}", code="dsl_var")
        return _as_tensor(torch, variables[expr["name"]], device=device, dtype=dtype)
    args = [_eval_expr(torch, item, variables, device=device, dtype=dtype) for item in expr["args"]]
    if op == "neg":
        value = -args[0]
    elif op == "abs":
        value = torch.abs(args[0])
    elif op == "sign":
        value = torch.sign(args[0])
    elif op == "sqrt_abs":
        value = torch.sqrt(torch.abs(args[0]))
    elif op == "square":
        value = args[0] * args[0]
    elif op == "tanh":
        value = torch.tanh(args[0])
    elif op == "add":
        value = args[0] + args[1]
    elif op == "sub":
        value = args[0] - args[1]
    elif op == "mul":
        value = args[0] * args[1]
    elif op == "div":
        den = args[1]
        valid = torch.abs(den) >= dsl.EPS
        safe_den = torch.where(valid, den, torch.ones_like(den))
        value = torch.where(valid, args[0] / safe_den, torch.zeros_like(args[0] + den))
    elif op == "max":
        value = torch.maximum(args[0], args[1])
    elif op == "min":
        value = torch.minimum(args[0], args[1])
    elif op == "clip":
        low = torch.minimum(args[1], args[2])
        high = torch.maximum(args[1], args[2])
        value = torch.minimum(torch.maximum(args[0], low), high)
    else:  # The validated DSL should make this unreachable.
        raise ValidationError(f"unhandled DSL operation {op!r}", code="dsl_op")
    if not bool(torch.isfinite(value).all().item()) or bool((torch.abs(value) > dsl.MAX_ABS_RESULT).any().item()):
        raise CandidateNumericError(f"DSL produced non-finite or oversized tensor in {op}")
    return value


def _forward(torch: Any, layers: list[dict[str, Any]], x: Any) -> Any:
    value = x
    for layer in layers:
        value = value @ layer["W"].transpose(0, 1) + layer["b"]
        activation = layer["activation"]
        if activation == "tanh":
            value = torch.tanh(value)
        elif activation == "relu":
            value = torch.relu(value)
        elif activation == "sigmoid":
            value = torch.sigmoid(value)
        elif activation is not None:
            raise ValidationError(f"unknown activation {activation!r}", code="activation")
    return value.squeeze(-1)


def _finite_layers(torch: Any, layers: list[dict[str, Any]]) -> None:
    # One device-to-host synchronization per batch, rather than one per tensor.
    # This preserves the fail-closed numeric guard while reducing CUDA stalls.
    flags = []
    for layer in layers:
        for key in ("W", "b"):
            value = layer[key]
            flags.append((~torch.isfinite(value)).any() | (torch.abs(value) > 1e100).any())
    if flags and bool(torch.stack(flags).any().item()):
        raise CandidateNumericError("non-finite/oversized model parameter")


def _validation_score(torch: Any, layers: list[dict[str, Any]], x: Any, y: Any, task_kind: str) -> float:
    """Compute validation metric on device and transfer only one scalar."""
    with torch.no_grad():
        pred = _forward(torch, layers, x)
        if task_kind == "classification":
            labels = y >= 0.5
            guesses = pred >= 0.0  # sigmoid(logit) >= 0.5 is equivalent to logit >= 0
            positive = labels
            negative = ~labels
            n_pos = positive.sum().clamp_min(1)
            n_neg = negative.sum().clamp_min(1)
            recall_pos = ((guesses & positive).sum().to(pred.dtype) / n_pos.to(pred.dtype))
            recall_neg = ((~guesses & negative).sum().to(pred.dtype) / n_neg.to(pred.dtype))
            score = 0.5 * (recall_pos + recall_neg)
        else:
            mean_y = y.mean()
            sse = torch.sum((y - pred) ** 2)
            sst = torch.sum((y - mean_y) ** 2)
            # Match the reference trainer's *raw* R² for checkpoint selection.
            # Clipping here changes which epoch is selected whenever all
            # candidate epochs have negative R², causing backend divergence.
            score = torch.where(sst > 1e-15, 1.0 - sse / sst, (sse <= 1e-15).to(pred.dtype))
        if not bool(torch.isfinite(score).item()):
            raise CandidateNumericError("non-finite validation metric")
        return float(score.item())


def _update_params(
    torch: Any,
    layers: list[dict[str, Any]],
    grads: list[Any],
    state: list[dict[str, Any]],
    genome_data: dict[str, Any],
    grad_norm: Any,
    batch_size: int,
    device: Any,
    dtype: Any,
    effective_lr: float | None = None,
) -> list[dict[str, Any]]:
    recipe = genome_data["recipe"]
    opt = genome_data["optimizer"]
    kind = opt["kind"]
    lr, l1, l2 = float(effective_lr if effective_lr is not None else recipe["learning_rate"]), float(recipe["l1"]), float(recipe["l2"])
    beta1, beta2, eps = float(opt["beta1"]), float(opt["beta2"]), float(opt["epsilon"])
    step = int(state[0]["step"]) + 1 if state else 1
    updates = genome_data.get("dsl", {})
    new_layers: list[dict[str, Any]] = []
    grad_index = 0
    for li, layer in enumerate(layers):
        new_state = {"mW": [], "vW": [], "mb": None, "vb": None, "step": step}
        layer_out: dict[str, Any] = {"activation": layer["activation"]}
        for name in ("W", "b"):
            p = layer[name].detach()
            g = grads[grad_index].detach()
            grad_index += 1
            # CPU reference clips the unaveraged minibatch gradient, then
            # averages it and adds regularization. Preserve that ordering.
            g = g / max(1, batch_size)
            sign_p = torch.sign(p)
            if kind in ("adamw", "lion"):
                g = g + l1 * sign_p
            else:
                g = g + l2 * p + l1 * sign_p
            old_m = state[li]["mW"] if name == "W" else state[li]["mb"]
            old_v = state[li]["vW"] if name == "W" else state[li]["vb"]
            if name == "W":
                old_m = old_m
                old_v = old_v
            if kind == "sgd":
                p_new = p - lr * g
                m_new, v_new = old_m, old_v
            elif kind == "momentum":
                m_new = beta1 * old_m + g
                v_new = old_v
                p_new = p - lr * m_new
            elif kind == "adagrad":
                v_new = old_v + g * g
                m_new = old_m
                p_new = p - lr * g / (torch.sqrt(torch.abs(v_new)) + eps)
            elif kind == "rmsprop":
                v_new = beta2 * old_v + (1.0 - beta2) * g * g
                m_new = old_m
                p_new = p - lr * g / (torch.sqrt(torch.abs(v_new)) + eps)
            elif kind in ("adam", "nadam", "adamw"):
                m_new = beta1 * old_m + (1.0 - beta1) * g
                v_new = beta2 * old_v + (1.0 - beta2) * g * g
                m_hat = m_new / (1.0 - beta1**step)
                v_hat = v_new / (1.0 - beta2**step)
                if kind == "nadam":
                    m_hat = (
                        beta1 * m_new / (1.0 - beta1 ** (step + 1))
                        + (1.0 - beta1) * g / (1.0 - beta1**step)
                    )
                p_new = p - lr * m_hat / (torch.sqrt(torch.abs(v_hat)) + eps)
                if kind == "adamw":
                    p_new = p_new - lr * l2 * p
            elif kind == "lion":
                direction = beta1 * old_m + (1.0 - beta1) * g
                m_new = beta2 * old_m + (1.0 - beta2) * g
                v_new = old_v
                p_new = p - lr * torch.sign(direction) - lr * l2 * p
            elif kind == "dsl":
                variables = {
                    "g": g,
                    "p": p,
                    "w": p,
                    "b": p,
                    "m": old_m,
                    "v": old_v,
                    "t": float(step),
                    "lr": lr,
                    "gn": grad_norm,
                }
                m_new = _eval_expr(torch, updates["m_update"], variables, device=device, dtype=dtype) if updates["m_update"] is not None else old_m
                v_new = _eval_expr(torch, updates["v_update"], variables, device=device, dtype=dtype) if updates["v_update"] is not None else old_v
                expr = updates["w_update"] if name == "W" else updates["b_update"]
                p_new = _eval_expr(torch, expr, variables, device=device, dtype=dtype)
            else:
                raise ValidationError(f"unknown optimizer {kind!r}", code="optimizer")
            layer_out[name] = p_new.detach().requires_grad_(True)
            if name == "W":
                new_state["mW"], new_state["vW"] = m_new.detach(), v_new.detach()
            else:
                new_state["mb"], new_state["vb"] = m_new.detach(), v_new.detach()
        new_layers.append(layer_out)
        state[li] = new_state
    return new_layers


def train_and_predict_torch(
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
    device: str = "cuda",
) -> dict[str, Any]:
    """Train one candidate using batched PyTorch tensor operations.

    CUDA use is explicit and is only routed here after a successful runtime
    smoke test. CPU execution is supported for parity tests, not selected by
    the normal dispatcher.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch backend unavailable: {exc}") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend was selected but torch.cuda.is_available() is false")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(device)
    dtype = torch.float32 if dev.type == "cuda" else torch.float64

    data = genome.data
    recipe = data["recipe"]
    n_features = len(train_x[0]) if train_x else 0
    params = genome.parameter_estimate(n_features)
    if params > parameter_limit:
        raise ValidationError(f"candidate has {params} parameters; limit is {parameter_limit}", code="parameter_limit")
    if not train_x or not val_x or not dev_x:
        raise ValidationError("empty train/val/dev data", code="empty_split")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValidationError("feature and target counts differ", code="split_shape")
    if any(len(row) != n_features for row in (*train_x, *val_x, *dev_x)):
        raise ValidationError("inconsistent feature dimensions", code="feature_shape")

    train_p, val_p, dev_p, prep = reference._prepare_features(
        train_x, val_x, dev_x, standardize=bool(recipe["standardize"])
    )
    x_train = torch.tensor(train_p, dtype=dtype, device=dev)
    y_train = torch.tensor(train_y, dtype=dtype, device=dev)
    x_val = torch.tensor(val_p, dtype=dtype, device=dev)
    y_val = torch.tensor(val_y, dtype=dtype, device=dev)
    x_dev = torch.tensor(dev_p, dtype=dtype, device=dev)

    if recipe["seed_policy"] == "constant":
        init_seed = seed_from(genome.hash, task_id, "constant-init")
    else:
        init_seed = seed_from(genome.hash, task_id, eval_seed, "init")
    init_layers = reference._init_layers(data["model"], n_features, recipe, random.Random(init_seed))
    layers = []
    state = []
    for layer in init_layers:
        w = torch.tensor(layer.W, dtype=dtype, device=dev).requires_grad_(True)
        b = torch.tensor(layer.b, dtype=dtype, device=dev).requires_grad_(True)
        layers.append({"W": w, "b": b, "activation": layer.activation})
        state.append({"mW": torch.zeros_like(w), "vW": torch.zeros_like(w),
                      "mb": torch.zeros_like(b), "vb": torch.zeros_like(b), "step": 0})

    shuffle_rng = random.Random(seed_from(genome.hash, task_id, eval_seed, "shuffle"))
    epochs = max(2, int(round(int(recipe["epochs"]) * max(0.05, float(epochs_scale)))))
    batch_size = min(int(recipe["batch_size"]), len(train_x))
    batches_per_epoch = max(1, math.ceil(len(train_x) / batch_size))
    planned_steps = max(1, min(int(max_steps), epochs * batches_per_epoch))
    best_layers = [{**layer, "W": layer["W"].detach().clone(), "b": layer["b"].detach().clone()} for layer in layers]
    best_metric = -float("inf")
    patience, stale, step = int(recipe["early_stopping_patience"]), 0, 0
    truncated = False
    total_loss = 0.0

    for _epoch in range(epochs):
        indices = list(range(len(train_x)))
        shuffle_rng.shuffle(indices)
        epoch_loss_tensor = torch.zeros((), dtype=dtype, device=dev)
        batches = 0
        for start in range(0, len(indices), batch_size):
            if step >= max_steps:
                truncated = True
                break
            batch = indices[start:start + batch_size]
            idx = torch.tensor(batch, dtype=torch.long, device=dev)
            xb, yb = x_train.index_select(0, idx), y_train.index_select(0, idx)
            logits = _forward(torch, layers, xb)
            if task_kind == "regression":
                losses = (logits - yb) ** 2
            else:
                if recipe["loss"] == "mse":
                    losses = (torch.sigmoid(logits) - yb) ** 2
                else:
                    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = losses.sum()
            params_now = [tensor for layer in layers for tensor in (layer["W"], layer["b"])]
            grads = list(torch.autograd.grad(loss, params_now, allow_unused=False))
            norm_sq = torch.zeros((), dtype=dtype, device=dev)
            for grad in grads:
                norm_sq = norm_sq + torch.sum(grad.detach() ** 2)
            norm = torch.sqrt(norm_sq)
            clip = float(recipe["grad_clip"])
            if clip > 0:
                clip_tensor = torch.as_tensor(clip, dtype=dtype, device=dev)
                factor = torch.clamp(clip_tensor / (norm + 1e-12), max=1.0)
                grads = [grad * factor for grad in grads]
                grad_norm = torch.minimum(norm, clip_tensor)
            else:
                grad_norm = norm
            lr_scale = reference._schedule_scale(str(recipe["lr_schedule"]), step + 1, planned_steps)
            layers = _update_params(
                torch, layers, grads, state, data, grad_norm, len(batch), dev, dtype,
                effective_lr=float(recipe["learning_rate"]) * lr_scale,
            )
            _finite_layers(torch, layers)
            step += 1
            batches += 1
            epoch_loss_tensor = epoch_loss_tensor + loss.detach() / max(1, len(batch))
        total_loss += float(epoch_loss_tensor.item()) / max(1, batches)
        metric = _validation_score(torch, layers, x_val, y_val, task_kind)
        if metric > best_metric + 1e-12:
            best_metric = metric
            best_layers = [{**layer, "W": layer["W"].detach().clone(), "b": layer["b"].detach().clone()} for layer in layers]
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
        if truncated:
            break
    layers = best_layers
    with torch.no_grad():
        pred_tensor = _forward(torch, layers, x_dev)
        if task_kind == "classification":
            pred_tensor = torch.sigmoid(pred_tensor)
        if not bool(torch.isfinite(pred_tensor).all().item()):
            raise CandidateNumericError("non-finite dev prediction")
        predictions = [float(v) for v in pred_tensor.detach().cpu().tolist()]
    return {
        "predictions": predictions,
        "val_metric": best_metric,
        "epochs_effective": epochs,
        "steps": step,
        "truncated": truncated,
        "parameter_count": params,
        "preprocessing": prep,
        "train_loss_last": total_loss / max(1, epochs),
        "training_backend": f"pytorch-{torch.__version__}-{dev.type}",
    }
