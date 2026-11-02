"""A tiny, safe scalar expression language used for learned update rules.

The language is intentionally not Python. A genome stores a JSON AST. The
interpreter has no access to the filesystem, environment, network, imports,
loops, comprehensions, attributes, or arbitrary function calls. Every
operation is allow-listed, every node is depth/size bounded, and non-finite
results are candidate failures rather than silently clipped values.

This is a deliberately narrow form of program search. It can express SGD,
momentum, RMSProp/Adam-like updates, sign updates, clipping, and modest
variants. It is not intended to be a general-purpose programming language.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Iterable

from ..errors import CandidateNumericError, DSLValidationError
from ..util import canonical_json

DSL_VERSION = "dsl-v2"
SUPPORTED_DSL_VERSIONS = frozenset({"dsl-v1", "dsl-v2"})
MAX_ABS_CONSTANT = 1_000_000.0
MAX_ABS_RESULT = 1_000_000_000_000.0
EPS = 1e-12

ALLOWED_VARS = frozenset({"g", "p", "w", "b", "m", "v", "t", "lr", "gn"})
V1_UNARY_OPS = frozenset({"neg", "abs", "sign", "sqrt_abs", "square", "tanh"})
V1_BINARY_OPS = frozenset({"add", "sub", "mul", "div", "max", "min"})
V1_TERNARY_OPS = frozenset({"clip"})

# DSL v2 broadens the typed expression language without introducing Python
# evaluation, loops, imports, attributes, or unbounded execution.
V2_UNARY_OPS = V1_UNARY_OPS | frozenset({
    "log_abs", "exp_clip", "sin", "cos", "sigmoid", "softplus",
    "relu", "reciprocal", "erf", "softsign", "leaky_relu",
})
V2_BINARY_OPS = V1_BINARY_OPS | frozenset({"hypot", "atan2"})
V2_TERNARY_OPS = V1_TERNARY_OPS | frozenset({"lerp", "select_pos"})
# Backwards-compatible public aliases describe the current language.
UNARY_OPS = V2_UNARY_OPS
BINARY_OPS = V2_BINARY_OPS
TERNARY_OPS = V2_TERNARY_OPS
COMMUTATIVE_OPS = frozenset({"add", "mul", "max", "min", "hypot"})


def node_count(expr: Any) -> int:
    if not isinstance(expr, dict):
        raise DSLValidationError("DSL node must be an object", code="dsl_type")
    op = expr.get("op")
    if op == "const" or op == "var":
        return 1
    args = expr.get("args")
    if not isinstance(args, list):
        raise DSLValidationError("operation node needs args", code="dsl_shape")
    return 1 + sum(node_count(arg) for arg in args)


def node_depth(expr: Any) -> int:
    if not isinstance(expr, dict):
        raise DSLValidationError("DSL node must be an object", code="dsl_type")
    op = expr.get("op")
    if op in ("const", "var"):
        return 1
    args = expr.get("args")
    if not isinstance(args, list):
        raise DSLValidationError("operation node needs args", code="dsl_shape")
    return 1 + max((node_depth(arg) for arg in args), default=0)


def validate_dsl(
    expr: Any,
    *,
    allowed_vars: Iterable[str] = ALLOWED_VARS,
    max_nodes: int = 96,
    max_depth: int = 12,
    version: str = DSL_VERSION,
) -> dict[str, Any]:
    """Validate, canonicalize and return a safe expression tree.

    ``dsl-v1`` is kept byte-for-byte structurally compatible so existing
    checkpoints remain readable. New candidates use ``dsl-v2`` and receive
    conservative algebraic canonicalization to collapse obvious syntactic
    duplicates. This is not a decision procedure for program equivalence.
    """

    if version not in SUPPORTED_DSL_VERSIONS:
        raise DSLValidationError(f"unsupported DSL version: {version!r}", code="dsl_version")
    allowed = set(allowed_vars)
    normalized = _validate_node(expr, allowed, max_nodes, max_depth, depth=1, version=version)
    if version == "dsl-v2":
        normalized = canonicalize_expr(normalized)
    count = node_count(normalized)
    if count > max_nodes:
        raise DSLValidationError(f"DSL has {count} nodes; limit is {max_nodes}", code="dsl_too_large")
    if node_depth(normalized) > max_depth:
        raise DSLValidationError("DSL is too deep", code="dsl_too_deep")
    return normalized


def _validate_node(expr: Any, allowed: set[str], max_nodes: int, max_depth: int, depth: int, version: str) -> dict[str, Any]:
    if depth > max_depth:
        raise DSLValidationError("DSL is too deep", code="dsl_too_deep")
    if not isinstance(expr, dict):
        raise DSLValidationError("DSL node must be an object", code="dsl_type")
    if set(expr.keys()) != {"op", "args"} and expr.get("op") not in ("const", "var"):
        raise DSLValidationError(f"unknown fields in DSL node: {sorted(expr.keys())}", code="dsl_fields")
    op = expr.get("op")
    if op == "const":
        if set(expr.keys()) != {"op", "value"}:
            raise DSLValidationError("const node needs exactly op/value", code="dsl_fields")
        value = expr.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DSLValidationError("const value must be numeric", code="dsl_type")
        value = float(value)
        if not math.isfinite(value) or abs(value) > MAX_ABS_CONSTANT:
            raise DSLValidationError("const value is non-finite or too large", code="dsl_const")
        return {"op": "const", "value": value}
    if op == "var":
        if set(expr.keys()) != {"op", "name"}:
            raise DSLValidationError("var node needs exactly op/name", code="dsl_fields")
        name = expr.get("name")
        if name not in allowed:
            raise DSLValidationError(f"unknown DSL variable: {name!r}", code="dsl_var")
        return {"op": "var", "name": name}
    unary_ops = V1_UNARY_OPS if version == "dsl-v1" else V2_UNARY_OPS
    binary_ops = V1_BINARY_OPS if version == "dsl-v1" else V2_BINARY_OPS
    ternary_ops = V1_TERNARY_OPS if version == "dsl-v1" else V2_TERNARY_OPS
    if op in unary_ops:
        arity = 1
    elif op in binary_ops:
        arity = 2
    elif op in ternary_ops:
        arity = 3
    else:
        raise DSLValidationError(f"operation not allow-listed: {op!r}", code="dsl_op")
    if set(expr.keys()) != {"op", "args"}:
        raise DSLValidationError("operation node needs exactly op/args", code="dsl_fields")
    args = expr.get("args")
    if not isinstance(args, list) or len(args) != arity:
        raise DSLValidationError(f"{op} expects {arity} arguments", code="dsl_arity")
    result = {
        "op": op,
        "args": [
            _validate_node(arg, allowed, max_nodes, max_depth, depth + 1, version)
            for arg in args
        ],
    }
    if node_count(result) > max_nodes:
        raise DSLValidationError(f"DSL exceeds {max_nodes} nodes", code="dsl_too_large")
    return result



def _const_node(value: float) -> dict[str, Any]:
    return {"op": "const", "value": float(value)}


def canonicalize_expr(expr: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a validated v2 AST without executing user code.

    The normalizer folds constant-only subtrees, sorts commutative operands,
    and applies a small set of exact neutral-element identities. It deliberately
    does not claim complete semantic equivalence (which is undecidable for a
    sufficiently expressive language).
    """
    op = expr["op"]
    if op in ("const", "var"):
        return dict(expr)
    args = [canonicalize_expr(arg) for arg in expr["args"]]

    # Constant-fold only closed subtrees. The interpreter is bounded and
    # deterministic; invalid numeric folds are retained for normal evaluation
    # to reject as candidate failures rather than being hidden.
    if all(arg["op"] == "const" for arg in args):
        try:
            value = evaluate({"op": op, "args": args}, {})
            if abs(value) <= MAX_ABS_CONSTANT:
                return _const_node(value)
        except Exception:
            pass

    if op in COMMUTATIVE_OPS:
        args = sorted(args, key=lambda item: canonical_json(item))
    if op == "neg" and args[0].get("op") == "neg":
        return args[0]["args"][0]
    if op == "abs" and args[0].get("op") == "abs":
        return args[0]
    if op in {"add", "sub"} and args[1] == _const_node(0.0):
        return args[0]
    if op == "add" and args[0] == _const_node(0.0):
        return args[1]
    if op == "mul":
        if _const_node(0.0) in args:
            return _const_node(0.0)
        if args[0] == _const_node(1.0):
            return args[1]
        if args[1] == _const_node(1.0):
            return args[0]
    if op == "div" and args[1] == _const_node(1.0):
        return args[0]
    if op in {"min", "max"} and args[0] == args[1]:
        return args[0]
    return {"op": op, "args": args}


def evaluate(expr: dict[str, Any], variables: dict[str, float]) -> float:
    """Evaluate a validated expression with safe numeric semantics."""

    op = expr["op"]
    if op == "const":
        return float(expr["value"])
    if op == "var":
        try:
            value = float(variables[expr["name"]])
        except KeyError as exc:
            raise DSLValidationError(f"missing variable {expr['name']!r}", code="dsl_var") from exc
        if not math.isfinite(value):
            raise CandidateNumericError(f"variable {expr['name']} is non-finite")
        return value
    args = expr["args"]
    values = [evaluate(arg, variables) for arg in args]
    try:
        if op == "neg":
            result = -values[0]
        elif op == "abs":
            result = abs(values[0])
        elif op == "sign":
            result = 1.0 if values[0] > 0 else (-1.0 if values[0] < 0 else 0.0)
        elif op == "sqrt_abs":
            result = math.sqrt(abs(values[0]))
        elif op == "square":
            result = values[0] * values[0]
        elif op == "tanh":
            result = math.tanh(values[0])
        elif op == "add":
            result = values[0] + values[1]
        elif op == "sub":
            result = values[0] - values[1]
        elif op == "mul":
            result = values[0] * values[1]
        elif op == "div":
            denominator = values[1]
            result = values[0] / denominator if abs(denominator) >= EPS else 0.0
        elif op == "max":
            result = max(values[0], values[1])
        elif op == "min":
            result = min(values[0], values[1])
        elif op == "clip":
            low, high = values[1], values[2]
            if low > high:
                low, high = high, low
            result = max(low, min(high, values[0]))
        elif op == "log_abs":
            result = math.log1p(abs(values[0]))
        elif op == "exp_clip":
            result = math.exp(max(-20.0, min(20.0, values[0])))
        elif op == "sin":
            result = math.sin(values[0])
        elif op == "cos":
            result = math.cos(values[0])
        elif op == "sigmoid":
            x = max(-60.0, min(60.0, values[0]))
            result = 1.0 / (1.0 + math.exp(-x))
        elif op == "softplus":
            x = values[0]
            result = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        elif op == "relu":
            result = max(0.0, values[0])
        elif op == "reciprocal":
            result = 1.0 / values[0] if abs(values[0]) >= EPS else 0.0
        elif op == "erf":
            result = math.erf(values[0])
        elif op == "softsign":
            result = values[0] / (1.0 + abs(values[0]))
        elif op == "leaky_relu":
            result = values[0] if values[0] >= 0.0 else 0.01 * values[0]
        elif op == "hypot":
            result = math.hypot(values[0], values[1])
        elif op == "atan2":
            result = math.atan2(values[0], values[1])
        elif op == "lerp":
            result = values[0] + max(0.0, min(1.0, values[2])) * (values[1] - values[0])
        elif op == "select_pos":
            result = values[1] if values[0] >= 0.0 else values[2]
        else:  # pragma: no cover - validation prevents this
            raise DSLValidationError(f"operation not allow-listed: {op}", code="dsl_op")
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        raise CandidateNumericError(f"DSL numeric failure in {op}: {exc}") from exc
    if not math.isfinite(result) or abs(result) > MAX_ABS_RESULT:
        raise CandidateNumericError(f"DSL produced a non-finite or oversized value in {op}")
    return float(result)


# ----------------------------------------------------------------------
# Templates and random generation
# ----------------------------------------------------------------------

def _const(value: float) -> dict[str, Any]:
    return {"op": "const", "value": float(value)}


def _var(name: str) -> dict[str, Any]:
    return {"op": "var", "name": name}


def _op(name: str, *args: dict[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": list(args)}


def dsl_template(kind: str, beta1: float = 0.9, beta2: float = 0.999) -> dict[str, dict[str, Any]]:
    """Return a known-safe set of update expressions.

    `p` is the current parameter, `g` the gradient, and m/v are persistent
    per-parameter states. These templates are starting points, not claims of
    novelty.
    """

    if kind == "sgd":
        update = _op("sub", _var("p"), _op("mul", _var("lr"), _var("g")))
        return {"w_update": update, "b_update": copy.deepcopy(update)}
    if kind == "momentum":
        b = _const(beta1)
        m_next = _op("add", _op("mul", b, _var("m")), _op("mul", _op("sub", _const(1.0), b), _var("g")))
        update = _op("sub", _var("p"), _op("mul", _var("lr"), m_next))
        return {
            "m_update": m_next,
            "w_update": update,
            "b_update": copy.deepcopy(update),
        }
    if kind in ("adam", "rmsprop", "adamw"):
        b1, b2 = _const(beta1), _const(beta2)
        m_next = _op("add", _op("mul", b1, _var("m")), _op("mul", _op("sub", _const(1.0), b1), _var("g")))
        v_next = _op("add", _op("mul", b2, _var("v")), _op("mul", _op("sub", _const(1.0), b2), _op("square", _var("g"))))
        if kind == "rmsprop":
            update = _op("div", _var("g"), _op("add", _op("sqrt_abs", v_next), _const(1e-8)))
        else:
            m_hat = _op("div", m_next, _op("sub", _const(1.0), _op("square", b1)))
            v_hat = _op("div", v_next, _op("sub", _const(1.0), _op("square", b2)))
            update = _op("div", m_hat, _op("add", _op("sqrt_abs", v_hat), _const(1e-8)))
        parameter_update = _op("sub", _var("p"), _op("mul", _var("lr"), update))
        return {
            "m_update": m_next,
            "v_update": v_next,
            "w_update": parameter_update,
            "b_update": copy.deepcopy(parameter_update),
        }
    if kind == "sign":
        update = _op("sub", _var("p"), _op("mul", _var("lr"), _op("sign", _var("g"))))
        return {
            "m_update": _op("add", _op("mul", _const(0.9), _var("m")), _op("mul", _const(0.1), _var("g"))),
            "w_update": update,
            "b_update": copy.deepcopy(update),
        }
    if kind == "clipped":
        clipped = _op("clip", _var("g"), _const(-1.0), _const(1.0))
        update = _op("sub", _var("p"), _op("mul", _var("lr"), clipped))
        return {"w_update": update, "b_update": copy.deepcopy(update)}
    raise ValueError(f"unknown DSL template: {kind}")


def random_expr(
    rng: random.Random,
    *,
    depth: int = 3,
    max_nodes: int = 96,
    variables: Iterable[str] = ALLOWED_VARS,
) -> dict[str, Any]:
    """Generate a bounded expression; the node budget is enforced exactly."""

    variables = tuple(sorted(set(variables)))
    if not variables:
        raise ValueError("random expression requires at least one variable")
    node_limit = max(1, int(max_nodes))

    def leaf() -> dict[str, Any]:
        if rng.random() < 0.62:
            return _var(rng.choice(variables))
        scale = 10.0 ** rng.randint(-6, 4)
        return _const(rng.uniform(-2.0, 2.0) * scale)

    def build(level: int, allowance: int) -> dict[str, Any]:
        if allowance <= 1 or level <= 1:
            return leaf()
        choice = rng.random()
        if choice < 0.20:
            op = rng.choice(tuple(sorted(V2_UNARY_OPS)))
            arity = 1
        elif choice < 0.33:
            op, arity = "clip", 3
        elif choice < 0.40:
            op, arity = "select_pos", 3
        else:
            op = rng.choice(tuple(sorted(V2_BINARY_OPS)))
            arity = 2

        # Reserve at least one node for every child before distributing the
        # remaining budget. A large first subtree can never starve later ones.
        if allowance < 1 + arity:
            return leaf()
        remaining = allowance - 1
        child_budgets = [1] * arity
        for _ in range(remaining - arity):
            child_budgets[rng.randrange(arity)] += 1
        children = [build(level - 1, child_budgets[index]) for index in range(arity)]
        return _op(op, *children)

    return build(max(1, int(depth)), node_limit)


def mutate_expr(
    expr: dict[str, Any],
    rng: random.Random,
    *,
    max_nodes: int = 96,
    variables: Iterable[str] = ALLOWED_VARS,
) -> dict[str, Any]:
    """Return a structurally mutated expression (not yet validated)."""

    # Clone through canonical structure; we never mutate the caller's dict.
    import copy

    result = copy.deepcopy(expr)

    def candidates(node: dict[str, Any]) -> list[dict[str, Any]]:
        out = [node]
        for arg in node.get("args", []):
            out.extend(candidates(arg))
        return out

    locations = candidates(result)
    target = rng.choice(locations)
    if target.get("op") == "const" and rng.random() < 0.5:
        changed = float(target["value"]) * rng.choice([0.5, 0.9, 1.1, 2.0])
        target["value"] = max(-MAX_ABS_CONSTANT, min(MAX_ABS_CONSTANT, changed))
        return result
    if target.get("op") == "var" and rng.random() < 0.4:
        target["name"] = rng.choice(tuple(sorted(set(variables))))
        return result
    # Replace the selected subtree with a fresh bounded expression.
    fresh = random_expr(rng, depth=rng.randint(1, 3), max_nodes=max_nodes, variables=variables)
    if target is result:
        return fresh
    # Find parent and replace.
    def replace(node: dict[str, Any]) -> bool:
        args = node.get("args")
        if not isinstance(args, list):
            return False
        for idx, arg in enumerate(args):
            if arg is target:
                args[idx] = fresh
                return True
            if replace(arg):
                return True
        return False

    replace(result)
    return result


def dsl_summary(updates: dict[str, Any]) -> dict[str, Any]:
    counts = {key: node_count(value) for key, value in updates.items() if isinstance(value, dict)}
    return {
        "version": DSL_VERSION,
        "node_counts": counts,
        "total_nodes": sum(counts.values()),
        "max_depth": max((node_depth(value) for value in updates.values() if isinstance(value, dict)), default=0),
    }
