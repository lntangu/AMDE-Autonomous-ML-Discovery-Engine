"""Append-only preregistration and falsifiable hypothesis evaluation."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from ..util import canonical_json, ensure_dir, utc_now
from .statistics import add_rejections, bootstrap_mean_ci, holm_bonferroni, paired_sign_flip, power_analysis

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,95}$")
_REQUIRED = {
    "hypothesis_id", "claim", "mechanism", "population", "intervention",
    "comparator", "outcome", "direction", "minimum_effect", "alpha",
    "benchmark_families", "seeds", "expected_sd", "target_power",
    "exclusion_criteria", "analysis_plan",
}


def validate_hypothesis(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("hypothesis protocol must be an object")
    missing, extra = _REQUIRED - set(raw), set(raw) - _REQUIRED
    if missing or extra:
        raise ValueError(f"hypothesis keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    value = dict(raw)
    hid = str(value["hypothesis_id"]).strip()
    if not _ID.fullmatch(hid):
        raise ValueError("hypothesis_id must be 3-96 safe characters")
    for field in ("claim", "mechanism", "population", "outcome", "analysis_plan"):
        if not isinstance(value[field], str) or len(value[field].strip()) < 12:
            raise ValueError(f"{field} must contain at least 12 non-whitespace characters")
        value[field] = value[field].strip()
    for field in ("intervention", "comparator"):
        if not isinstance(value[field], dict) or not value[field]:
            raise ValueError(f"{field} must be a non-empty object")
    direction = value["direction"]
    if direction not in {"greater", "less", "two-sided"}:
        raise ValueError("direction must be greater, less, or two-sided")
    effect, alpha = float(value["minimum_effect"]), float(value["alpha"])
    if not math.isfinite(effect) or effect < 0:
        raise ValueError("minimum_effect must be finite and non-negative")
    if not math.isfinite(alpha) or not 0.001 <= alpha <= 0.2:
        raise ValueError("alpha must be between 0.001 and 0.2")
    families = value["benchmark_families"]
    if not isinstance(families, list) or len(families) < 6 or any(not str(x).strip() for x in families):
        raise ValueError("benchmark_families must contain at least six independent family identifiers")
    if len(set(map(str, families))) != len(families):
        raise ValueError("benchmark_families must be unique")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3 or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError("seeds must contain at least three integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    exclusions = value["exclusion_criteria"]
    if not isinstance(exclusions, list) or any(not isinstance(x, str) for x in exclusions):
        raise ValueError("exclusion_criteria must be a list of strings")
    expected_sd = float(value["expected_sd"])
    target_power = float(value["target_power"])
    design = power_analysis(minimum_effect=effect, expected_sd=expected_sd, alpha=alpha,
                             target_power=target_power, direction=direction)
    value["hypothesis_id"] = hid
    value["minimum_effect"] = effect
    value["alpha"] = alpha
    value["expected_sd"] = expected_sd
    value["target_power"] = target_power
    value["benchmark_families"] = sorted(map(str, families))
    value["seeds"] = sorted(seeds)
    value["exclusion_criteria"] = [x.strip() for x in exclusions]
    value["power_design"] = design
    value["power_design_adequate"] = len(families) >= int(design["recommended_independent_families"])
    value["schema_version"] = "amde-hypothesis-v2"
    return value


def hypothesis_hash(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(protocol)).encode("utf-8")).hexdigest()


def evaluate_hypothesis(protocol: Mapping[str, Any], family_differences: Mapping[str, float],
                         *, bootstrap_repetitions: int = 20000, seed: int = 42) -> dict[str, Any]:
    spec = validate_hypothesis(protocol)
    expected = set(spec["benchmark_families"])
    observed = set(map(str, family_differences))
    if expected != observed:
        raise ValueError(f"family coverage mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    diffs = [float(family_differences[key]) for key in spec["benchmark_families"]]
    if any(not math.isfinite(value) for value in diffs):
        raise ValueError("family differences must be finite")
    ci = bootstrap_mean_ci(diffs, repetitions=bootstrap_repetitions, seed=seed, confidence=0.95)
    test = paired_sign_flip(diffs, alternative=spec["direction"], seed=seed)
    effect = spec["minimum_effect"]
    if spec["direction"] == "greater":
        supported = ci[0] > effect and test["p_value"] <= spec["alpha"]
        falsified = ci[1] <= effect
    elif spec["direction"] == "less":
        supported = ci[1] < -effect and test["p_value"] <= spec["alpha"]
        falsified = ci[0] >= -effect
    else:
        supported = (ci[0] > effect or ci[1] < -effect) and test["p_value"] <= spec["alpha"]
        falsified = False
    adequately_powered = bool(spec["power_design_adequate"])
    status = ("inconclusive_underpowered" if not adequately_powered else
              "supported" if supported else "falsified" if falsified else "inconclusive")
    return {
        "schema_version": "amde-hypothesis-result-v1",
        "hypothesis_id": spec["hypothesis_id"],
        "hypothesis_sha256": hypothesis_hash(spec),
        "status": status,
        "n_families": len(diffs),
        "mean_effect": sum(diffs) / len(diffs),
        "bootstrap_ci_95": [ci[0], ci[1]],
        "test": test,
        "minimum_effect": effect,
        "alpha": spec["alpha"],
        "power_design": spec["power_design"],
        "power_design_adequate": adequately_powered,
        "interpretation": (
            "Observed evidence is suggestive, but the preregistered family count is below the normal-approximation power target; do not claim support or falsification."
            if not adequately_powered else
            "Evidence supports the preregistered directional prediction within the tested families."
            if supported else
            "The confidence interval excludes the preregistered minimum effect in the predicted direction."
            if falsified else
            "Evidence is inconclusive; absence of significance is not evidence of no effect."
        ),
    }


def audit_hypothesis_batch(payload: Mapping[str, Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Apply family-wise Holm correction to a prespecified complete hypothesis set."""
    if payload.get("schema_version") != "amde-hypothesis-batch-v1":
        raise ValueError("schema_version must be amde-hypothesis-batch-v1")
    expected_ids = payload.get("expected_hypothesis_ids")
    results = payload.get("results")
    if not isinstance(expected_ids, list) or not expected_ids or any(not isinstance(x, str) or not x for x in expected_ids):
        raise ValueError("expected_hypothesis_ids must be a non-empty list of IDs frozen before analysis")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected_hypothesis_ids must be unique")
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a non-empty list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get("schema_version") != "amde-hypothesis-result-v1":
            raise ValueError("each result must use amde-hypothesis-result-v1")
        hid = result.get("hypothesis_id")
        if not isinstance(hid, str) or not hid or hid in by_id:
            raise ValueError("hypothesis results need unique non-empty hypothesis_id values")
        test = result.get("test")
        if not isinstance(test, Mapping):
            raise ValueError(f"hypothesis {hid}: missing test result")
        p_value = float(test.get("p_value"))
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"hypothesis {hid}: invalid p-value")
        by_id[hid] = result
    if set(by_id) != set(expected_ids):
        raise ValueError(
            f"hypothesis batch coverage mismatch; missing={sorted(set(expected_ids)-set(by_id))}, "
            f"extra={sorted(set(by_id)-set(expected_ids))}"
        )
    raw = {hid: float(by_id[hid]["test"]["p_value"]) for hid in expected_ids}
    adjusted = add_rejections(holm_bonferroni(raw), alpha=alpha)
    decisions: dict[str, Any] = {}
    for hid in expected_ids:
        result = by_id[hid]
        unadjusted_status = str(result.get("status", "unknown"))
        familywise_reject = bool(adjusted[hid]["reject"])
        if unadjusted_status == "inconclusive_underpowered":
            decision = "inconclusive_underpowered"
        elif familywise_reject and unadjusted_status == "supported":
            decision = "supported_after_holm"
        elif familywise_reject and unadjusted_status == "falsified":
            decision = "minimum_effect_falsified_after_holm"
        else:
            decision = "not_rejected_after_holm"
        decisions[hid] = {
            **adjusted[hid],
            "unadjusted_status": unadjusted_status,
            "familywise_decision": decision,
            "hypothesis_sha256": result.get("hypothesis_sha256"),
        }
    return {
        "schema_version": "amde-hypothesis-batch-audit-v1",
        "alpha": float(alpha),
        "correction": "Holm-Bonferroni",
        "n_hypotheses": len(expected_ids),
        "hypotheses": decisions,
        "expected_hypothesis_ids": list(expected_ids),
        "interpretation": "Family-wise error control applies only to the complete prespecified result set supplied here; omitted or post-hoc hypotheses invalidate this correction.",
    }


class HypothesisRegistry:
    """Append-only SQLite store. Registration records cannot be silently edited."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        with self._connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS hypotheses(
                    hypothesis_id TEXT PRIMARY KEY, protocol_json TEXT NOT NULL,
                    protocol_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hypothesis_results(
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL, protocol_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL, result_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id)
                );
            """)

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.path, timeout=15)
        try:
            con.execute("PRAGMA busy_timeout=15000")
            yield con
        finally:
            con.close()

    def register(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        spec = validate_hypothesis(raw)
        encoded = canonical_json(spec)
        digest = hypothesis_hash(spec)
        with self._connection() as con:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT protocol_sha256 FROM hypotheses WHERE hypothesis_id=?",
                                   (spec["hypothesis_id"],)).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("hypothesis_id already registered with a different frozen protocol")
                return {"hypothesis_id": spec["hypothesis_id"], "protocol_sha256": digest, "inserted": False}
            con.execute("INSERT INTO hypotheses VALUES(?,?,?,?)",
                        (spec["hypothesis_id"], encoded, digest, utc_now()))
            con.commit()
        return {"hypothesis_id": spec["hypothesis_id"], "protocol_sha256": digest, "inserted": True}

    def get(self, hypothesis_id: str) -> dict[str, Any] | None:
        with self._connection() as con:
            row = con.execute("SELECT protocol_json, protocol_sha256, registered_at FROM hypotheses WHERE hypothesis_id=?",
                              (hypothesis_id,)).fetchone()
        if row is None:
            return None
        return {"protocol": json.loads(row[0]), "protocol_sha256": row[1], "registered_at": row[2]}

    def record_result(self, hypothesis_id: str, result: Mapping[str, Any]) -> str:
        registered = self.get(hypothesis_id)
        if registered is None:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        if result.get("hypothesis_sha256") != registered["protocol_sha256"]:
            raise ValueError("result protocol hash does not match the preregistered hypothesis")
        encoded = canonical_json(dict(result))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._connection() as con:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("INSERT INTO hypothesis_results(hypothesis_id,protocol_sha256,result_json,result_sha256,recorded_at) VALUES(?,?,?,?,?)",
                        (hypothesis_id, registered["protocol_sha256"], encoded, digest, utc_now()))
            con.commit()
        return digest
