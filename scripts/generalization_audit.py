#!/usr/bin/env python3
"""Run a paired, family-clustered generalization audit from preregistered JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amde.evaluation.generalization import audit_generalization  # noqa: E402


def _markdown(report: dict) -> str:
    lines = [
        "# AMDE Generalization Audit",
        "",
        f"- Evidence tier: **{report['evidence_tier']}**",
        f"- Families: {report['n_families']} (unseen={report['n_unseen_families']}, external={report['n_external_families']})",
        f"- Domains: {report['n_domains']}",
        f"- Mean candidate score: {report['mean_candidate_score']:.6f}",
        f"- Mean baseline score: {report['mean_baseline_score']:.6f}",
        f"- Paired delta: {report['mean_paired_delta']:.6f}",
        f"- Family-cluster bootstrap 95% CI: [{report['bootstrap_ci_95'][0]:.6f}, {report['bootstrap_ci_95'][1]:.6f}]",
        f"- Paired permutation p-value: {report['permutation_test']['p_value']:.6g}",
        f"- Family win/loss/tie: {report['family_wins']}/{report['family_losses']}/{report['family_ties']}",
        f"- Internal gate: `{report['internal_generalization_gate']}`",
        f"- External gate: `{report['external_generalization_gate']}`",
        "",
        "## Gates",
        "",
    ]
    for key, passed in report["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {key}")
    lines += ["", "## Family-level paired deltas", ""]
    for family, delta in sorted(report["family_deltas"].items(), key=lambda item: item[1]):
        lines.append(f"- `{family}`: {delta:+.6f}")
    lines += ["", "> " + report["interpretation"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON matching docs/GENERALIZATION_PROTOCOL.md")
    parser.add_argument("--out", default="reports/generalization_audit")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "amde-generalization-input-v1":
            raise ValueError("input schema_version must be amde-generalization-input-v1")
        report = audit_generalization(
            payload["candidate_family_scores"],
            payload["baseline_family_scores"],
            family_metadata=payload["family_metadata"],
            preregistered=bool(payload.get("preregistered", False)),
            evaluation_used_for_selection=bool(payload.get("evaluation_used_for_selection", True)),
            minimum_families=int(payload.get("minimum_families", 10)),
            minimum_external_families=int(payload.get("minimum_external_families", 10)),
            minimum_domains=int(payload.get("minimum_domains", 3)),
            minimum_effect=float(payload.get("minimum_effect", 0.0)),
            minimum_win_rate=float(payload.get("minimum_win_rate", 0.65)),
            maximum_family_regression=float(payload.get("maximum_family_regression", 0.10)),
            bootstrap_repetitions=args.bootstrap,
            seed=args.seed,
        )
        report["candidate_id"] = str(payload.get("candidate_id", "unspecified"))
        report["baseline_id"] = str(payload.get("baseline_id", "unspecified"))
        report["benchmark_protocol_id"] = str(payload.get("benchmark_protocol_id", "unspecified"))
        report["benchmark_protocol_hash"] = str(payload.get("benchmark_protocol_hash", "unspecified"))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "generalization_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "GENERALIZATION_AUDIT.md").write_text(_markdown(report), encoding="utf-8")
        print(f"Evidence tier: {report['evidence_tier']}")
        print(f"Internal gate: {report['internal_generalization_gate']}")
        print(f"External gate: {report['external_generalization_gate']}")
        print(f"Report: {out.resolve()}")
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"generalization audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
