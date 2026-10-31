"""Deterministic synthetic/classical/compositional data generators.

These are small system and method-selection benchmarks, not evidence of
general intelligence. Models see only the train split for updates and the
validation split for checkpoint selection. Search sees aggregate dev scores.
Sealed challenge splits are generated only by the sealed evaluator.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .base import Example


def _gauss(rng: random.Random) -> float:
    # Explicit Box-Muller keeps the implementation transparent and stable.
    u1 = max(1e-12, rng.random())
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def _noise(rng: random.Random, scale: float) -> float:
    return _gauss(rng) * scale


def _flip_label(rng: random.Random, y: int, probability: float) -> int:
    if rng.random() < probability:
        return 1 - y
    return y


def _linear_weights(spec: dict[str, Any], salt: str = "weights") -> list[float]:
    rng = random.Random(int(spec["seed"]) ^ (0xA5A5 if salt == "weights" else 0x5A5A))
    n = int(spec["n_features"])
    return [_gauss(rng) for _ in range(n)]


def linear_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    weights = _linear_weights(spec)
    x = [_gauss(rng) for _ in range(n)]
    y = 0.7 + sum(w * v for w, v in zip(weights, x)) + _noise(rng, 0.25)
    return Example(tuple(x), float(y))


def quadratic_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    x = [_uniform(rng, -1.5, 1.5) for _ in range(n)]
    x0, x1, x2, x3 = (x + [0.0, 0.0, 0.0, 0.0])[:4]
    y = 1.2 * x0 * x0 + 0.8 * x1 * x2 - 0.6 * x3 + 0.4 * x1 + _noise(rng, 0.25)
    return Example(tuple(x), float(y))


def xor_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    x0 = _uniform(rng, -1.0, 1.0)
    x1 = _uniform(rng, -1.0, 1.0)
    y = int((x0 > 0.0) != (x1 > 0.0))
    y = _flip_label(rng, y, 0.05)
    return Example((x0, x1), float(y))


def blobs_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    label = 1 if rng.random() < 0.5 else 0
    center = 1.0 if label == 1 else -1.0
    x = [_gauss(rng) * 0.65 + center * (1.0 if i == 0 else 0.25) for i in range(n)]
    return Example(tuple(x), float(_flip_label(rng, label, 0.03)))


def noisy_linear_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    weights = _linear_weights(spec, salt="classifier")
    x = [_gauss(rng) for _ in range(n)]
    score = sum(w * v for w, v in zip(weights, x)) - 0.2
    y = int(score > 0.0)
    y = _flip_label(rng, y, 0.08)
    return Example(tuple(x), float(y))


def circles_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    radius = abs(_gauss(rng)) * 0.45 + (0.1 if rng.random() < 0.5 else 1.2)
    angle = _uniform(rng, 0.0, 2.0 * math.pi)
    x0 = radius * math.cos(angle) + _noise(rng, 0.06)
    x1 = radius * math.sin(angle) + _noise(rng, 0.06)
    y = int(radius > 0.75)
    y = _flip_label(rng, y, 0.04)
    return Example((x0, x1), float(y))


def imbalanced_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    y = 1 if rng.random() < 0.22 else 0
    center = 0.8 if y == 1 else -0.2
    x = [_gauss(rng) * (0.8 if y == 0 else 0.6) + center * (1.0 if i == 0 else 0.1) for i in range(n)]
    return Example(tuple(x), float(_flip_label(rng, y, 0.03)))


def regime_switch_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    x0, x1, x2 = (x + [0.0, 0.0, 0.0])[:3]
    if x0 > 0.0:
        y = 1.8 * x1 - 1.1 * x2 + 0.3
    else:
        y = -1.4 * x1 + 0.7 * x2 - 0.2
    return Example(tuple(x), float(y + _noise(rng, 0.2)))


def shifted_xor_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    # Third feature is a nuisance variable, independent of the target.
    x0 = _uniform(rng, -1.0, 1.0)
    x1 = _uniform(rng, -1.0, 1.0)
    nuisance = _gauss(rng) * 0.5
    y = int((x0 > 0.0) != (x1 > 0.0))
    return Example((x0, x1, nuisance), float(_flip_label(rng, y, 0.03)))


def shifted_xor_challenge(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Covariate/concept shift for the compositional challenge cohort."""

    x0 = _uniform(rng, -1.0, 1.0)
    x1 = _uniform(rng, -1.0, 1.0)
    nuisance = _gauss(rng) * 0.5
    # Same XOR family, but the decision boundary is translated. Training data
    # never shows this boundary, so this is intentionally harder.
    y = int((x0 > 0.65) != (x1 > -0.55))
    return Example((x0, x1, nuisance), float(_flip_label(rng, y, 0.03)))


def regime_switch_challenge(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Concept-shift cohort: the regime mapping is swapped."""

    n = int(spec["n_features"])
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    x0, x1, x2 = (x + [0.0, 0.0, 0.0])[:3]
    if x0 > 0.0:
        y = -1.4 * x1 + 0.7 * x2 - 0.2
    else:
        y = 1.8 * x1 - 1.1 * x2 + 0.3
    return Example(tuple(x), float(y + _noise(rng, 0.2)))


def sine_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Smooth nonlinear regression with an interaction and nuisance features."""
    n = int(spec["n_features"])
    x = [_uniform(rng, -2.0, 2.0) if i == 0 else _uniform(rng, -1.0, 1.0) if i == 1 else _gauss(rng) for i in range(n)]
    x0, x1 = x[0], x[1]
    y = math.sin(2.5 * x0) + 0.35 * x1 * x1 + 0.25 * x0 * x1 + _noise(rng, 0.08)
    return Example(tuple(x), float(y))


def friedman_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Friedman-style nonlinear regression with five relevant dimensions."""
    n = int(spec["n_features"])
    if n < 5:
        raise ValueError("friedman_regression requires n_features >= 5")
    x = [_uniform(rng, 0.0, 1.0) for _ in range(n)]
    y = (10.0 * math.sin(math.pi * x[0] * x[1])
         + 20.0 * (x[2] - 0.5) ** 2
         + 10.0 * x[3]
         + 5.0 * x[4]
         + _noise(rng, 0.35))
    return Example(tuple(x), float(y))


def sparse_linear_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """High-dimensional regression where only the first few inputs matter."""
    n = int(spec["n_features"])
    x = [_gauss(rng) for _ in range(n)]
    weights = [1.8, -1.2, 0.7]
    y = 0.4 + sum(weights[i] * x[i] for i in range(min(3, n))) + _noise(rng, 0.18)
    return Example(tuple(x), float(y))


def heteroscedastic_regression(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Regression with input-dependent noise scale."""
    n = int(spec["n_features"])
    x = [_gauss(rng) for _ in range(n)]
    x0 = x[0]
    x1 = x[1] if n > 1 else 0.0
    x2 = x[2] if n > 2 else 0.0
    noise_scale = 0.05 + 0.35 * abs(x0)
    y = 1.2 * x0 - 0.7 * x1 + 0.25 * x2 * x2 + _noise(rng, noise_scale)
    return Example(tuple(x), float(y))


def moons_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Two interleaving half-moons with mild observation noise."""
    n = int(spec["n_features"])
    label = 1 if rng.random() < 0.5 else 0
    theta = _uniform(rng, 0.0, math.pi)
    if label == 0:
        x0, x1 = math.cos(theta), math.sin(theta)
    else:
        x0, x1 = 1.0 - math.cos(theta), 0.5 - math.sin(theta)
    values = [x0 + _noise(rng, 0.07), x1 + _noise(rng, 0.07)]
    values.extend(_gauss(rng) * 0.5 for _ in range(max(0, n - 2)))
    return Example(tuple(values), float(_flip_label(rng, label, 0.025)))


def parity_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Parity over four signs, with optional nuisance dimensions."""
    n = int(spec["n_features"])
    if n < 4:
        raise ValueError("parity_classification requires n_features >= 4")
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    y = 0
    for value in x[:4]:
        y ^= int(value >= 0.0)
    return Example(tuple(x), float(_flip_label(rng, y, 0.035)))


def spiral_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Interleaving spirals; intentionally harder than the default tabular panel."""
    n = int(spec["n_features"])
    label = 1 if rng.random() < 0.5 else 0
    radius = _uniform(rng, 0.08, 1.0)
    angle = radius * 4.0 * math.pi + (math.pi if label else 0.0)
    x0 = radius * math.cos(angle) + _noise(rng, 0.045)
    x1 = radius * math.sin(angle) + _noise(rng, 0.045)
    values = [x0, x1]
    values.extend(_gauss(rng) * 0.4 for _ in range(max(0, n - 2)))
    return Example(tuple(values), float(_flip_label(rng, label, 0.02)))


def checkerboard_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """Piecewise-constant checkerboard decision surface."""
    n = int(spec["n_features"])
    if n < 2:
        raise ValueError("checkerboard_classification requires n_features >= 2")
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    cell_x = int(math.floor((x[0] + 1.0) * 3.0))
    cell_y = int(math.floor((x[1] + 1.0) * 3.0))
    label = (cell_x + cell_y) & 1
    return Example(tuple(x), float(_flip_label(rng, label, 0.025)))


def sparse_parity_classification(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    """High-dimensional parity with only a sparse subset of relevant inputs."""
    n = int(spec["n_features"])
    if n < 8:
        raise ValueError("sparse_parity_classification requires n_features >= 8")
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    label = int(x[1] >= 0.0) ^ int(x[4] >= 0.0) ^ int(x[7] >= 0.0)
    return Example(tuple(x), float(_flip_label(rng, label, 0.03)))


def _shifted_moons(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    base = moons_classification(rng, spec, count)
    x0, x1, *rest = base.x
    # Held-out covariate shift; target relation is retained.
    return Example((x0 * 0.85 + 0.25, x1 * 1.1 - 0.15, *rest), base.y)


def _shifted_checkerboard(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    n = int(spec["n_features"])
    x = [_uniform(rng, -1.0, 1.0) for _ in range(n)]
    # Shift cell boundaries relative to the open distribution.
    cx = int(math.floor((x[0] + 1.0 - 0.18) * 3.0))
    cy = int(math.floor((x[1] + 1.0 + 0.12) * 3.0))
    label = (cx + cy) & 1
    return Example(tuple(x), float(_flip_label(rng, label, 0.03)))


def _shifted_spiral(rng: random.Random, spec: dict[str, Any], count: int) -> Example:
    base = spiral_classification(rng, spec, count)
    x0, x1, *rest = base.x
    angle = 0.28
    ca, sa = math.cos(angle), math.sin(angle)
    return Example((ca * x0 - sa * x1 + 0.1, sa * x0 + ca * x1 - 0.08, *rest), base.y)


GENERATORS = {
    "linear_regression": linear_regression,
    "quadratic_regression": quadratic_regression,
    "xor_classification": xor_classification,
    "blobs_classification": blobs_classification,
    "noisy_linear_classification": noisy_linear_classification,
    "circles_classification": circles_classification,
    "imbalanced_classification": imbalanced_classification,
    "regime_switch_regression": regime_switch_regression,
    "shifted_xor_classification": shifted_xor_classification,
    "sine_regression": sine_regression,
    "friedman_regression": friedman_regression,
    "sparse_linear_regression": sparse_linear_regression,
    "heteroscedastic_regression": heteroscedastic_regression,
    "moons_classification": moons_classification,
    "parity_classification": parity_classification,
    "spiral_classification": spiral_classification,
    "checkerboard_classification": checkerboard_classification,
    "sparse_parity_classification": sparse_parity_classification,
}

CHALLENGE_GENERATORS = {
    "shifted_xor_classification": shifted_xor_challenge,
    "regime_switch_regression": regime_switch_challenge,
    "moons_classification": _shifted_moons,
    "spiral_classification": _shifted_spiral,
    "checkerboard_classification": _shifted_checkerboard,
}
