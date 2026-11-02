"""AMDE 4.0 Identity, Serialization, and Provenance Hashing.

Enforces:
1. Invariant canonical serialization independent of dictionary key order or whitespace.
2. Collision-resistant cryptographic SHA-256 Candidate Identity.
3. Explicit separation of Candidate Identity from Evaluation Cache Identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unicodedata
from typing import Any, Mapping
import numpy as np

from amde.core.contracts import EvaluationCacheKey
from amde.core.errors import ValidationError


def canonical_json(data: Any) -> str:
    """Recursively formats structures into an invariant canonical JSON string.

    Key guarantees:
    - Dictionary keys are sorted alphabetically and normalized to Unicode NFC.
    - Floats are rounded and formatted to 7 decimal places; -0.0 is normalized to 0.0.
    - NaN and Inf float values raise ValidationError (not masked as null).
    - Booleans evaluate to lowercase 'true'/'false'.
    - NumPy integer, float, and bool types are handled natively.
    - No non-deterministic whitespace is introduced.
    """
    if isinstance(data, dict):
        items = []
        for orig_k in sorted(data.keys(), key=lambda x: unicodedata.normalize("NFC", str(x))):
            norm_k = unicodedata.normalize("NFC", str(orig_k))
            items.append(f'"{norm_k}":{canonical_json(data[orig_k])}')
        return "{" + ",".join(items) + "}"
    elif isinstance(data, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in data) + "]"
    elif isinstance(data, (float, np.floating)):
        f_val = float(data)
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValidationError(f"Invalid non-finite float value in canonical data: {data}")
        if f_val == 0.0:
            f_val = 0.0
        return f"{f_val:.7f}"
    elif isinstance(data, (bool, np.bool_)):
        return "true" if data else "false"
    elif data is None:
        return "null"
    elif isinstance(data, (int, np.integer)):
        return json.dumps(int(data))
    elif isinstance(data, str):
        return json.dumps(unicodedata.normalize("NFC", data))
    else:
        return json.dumps(unicodedata.normalize("NFC", str(data)))


def compute_candidate_hash(genome: Mapping[str, Any]) -> str:
    """Computes a deterministic cryptographic SHA-256 hash of a candidate pipeline genome.

    Strips any transient execution artifacts (timestamps, fitness, status, trial IDs).
    """
    clean_copy = copy.deepcopy(dict(genome))
    for ephemeral_key in ("fitness", "score", "timestamp", "eval_time", "trial_id", "status", "rung_id"):
        clean_copy.pop(ephemeral_key, None)
    serialized = canonical_json(clean_copy)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_data_split_hash(X_tr: np.ndarray, y_tr: np.ndarray, X_v: np.ndarray, y_v: np.ndarray) -> str:
    """Computes a cryptographic fingerprint of the complete training and validation data partitions."""
    h = hashlib.sha256()
    h.update(str(X_tr.shape).encode("utf-8"))
    h.update(str(X_tr.dtype).encode("utf-8"))
    h.update(str(X_v.shape).encode("utf-8"))
    h.update(str(X_v.dtype).encode("utf-8"))
    h.update(str(y_tr.shape).encode("utf-8"))
    h.update(str(y_tr.dtype).encode("utf-8"))
    h.update(str(y_v.shape).encode("utf-8"))
    h.update(str(y_v.dtype).encode("utf-8"))
    # Hash complete contiguous byte buffers
    h.update(np.ascontiguousarray(X_tr).tobytes())
    h.update(np.ascontiguousarray(y_tr).tobytes())
    h.update(np.ascontiguousarray(X_v).tobytes())
    h.update(np.ascontiguousarray(y_v).tobytes())
    return h.hexdigest()


def make_evaluation_cache_key(
    candidate_hash: str,
    dataset_id: str,
    split_hash: str,
    seed: int,
    fidelity_rung: int,
    metric_id: str = "accuracy",
    evaluator_version: str = "4.0.1",
) -> EvaluationCacheKey:
    """Constructs a typed, collision-resistant EvaluationCacheKey."""
    return EvaluationCacheKey(
        candidate_hash=candidate_hash,
        dataset_id=dataset_id,
        split_hash=split_hash,
        seed=seed,
        fidelity_rung=fidelity_rung,
        metric_id=metric_id,
        evaluator_version=evaluator_version,
    )
