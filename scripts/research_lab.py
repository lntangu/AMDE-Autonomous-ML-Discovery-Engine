#!/usr/bin/env python3
"""AMDE preregistration, ablation planning and discovery-archive CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amde.discovery.archive import DiscoveryArchive  # noqa: E402
from amde.discovery.genome import Genome  # noqa: E402
from amde.research.ablation import audit_ablation_results, build_ablation_plan  # noqa: E402
from amde.research.hypotheses import HypothesisRegistry, audit_hypothesis_batch, evaluate_hypothesis  # noqa: E402
from amde.research.statistics import power_analysis  # noqa: E402
from amde.util import atomic_write_json  # noqa: E402


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("hypothesis-register", help="Freeze an append-only preregistration")
    register.add_argument("protocol")
    register.add_argument("--db", default="storage/research/hypotheses.sqlite3")

    evaluate = sub.add_parser("hypothesis-evaluate", help="Evaluate preregistered family-level effects")
    evaluate.add_argument("protocol")
    evaluate.add_argument("family_differences")
    evaluate.add_argument("--db", default="storage/research/hypotheses.sqlite3")
    evaluate.add_argument("--out", default="reports/research/hypothesis_result.json")
    evaluate.add_argument("--bootstrap", type=int, default=20000)
    evaluate.add_argument("--seed", type=int, default=42)

    batch = sub.add_parser("hypothesis-batch-audit", help="Apply Holm correction to a prespecified hypothesis family")
    batch.add_argument("results", help="JSON with expected_hypothesis_ids and result objects")
    batch.add_argument("--alpha", type=float, default=0.05)
    batch.add_argument("--out", default="reports/research/hypothesis_batch_audit.json")

    plan = sub.add_parser("ablation-plan", help="Create a deterministic paired ablation plan")
    plan.add_argument("base_config")
    plan.add_argument("factors")
    plan.add_argument("--mode", choices=("one_factor", "full_factorial"), default="one_factor")
    plan.add_argument("--seeds", default="11,23,37,53,71")
    plan.add_argument("--max-variants", type=int, default=128)
    plan.add_argument("--out", default="reports/research/ablation_plan.json")

    audit = sub.add_parser("ablation-audit", help="Analyse paired ablation results with Holm correction")
    audit.add_argument("results")
    audit.add_argument("--alpha", type=float, default=0.05)
    audit.add_argument("--bootstrap", type=int, default=20000)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--out", default="reports/research/ablation_audit.json")

    archive = sub.add_parser("archive-stats", help="Inspect the cross-campaign discovery archive")
    archive.add_argument("--db", default="storage/discovery_archive.sqlite3")
    archive.add_argument("--limit", type=int, default=20)

    power = sub.add_parser("power-analysis", help="Estimate independent benchmark families needed")
    power.add_argument("--minimum-effect", type=float, required=True)
    power.add_argument("--expected-sd", type=float, required=True)
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--target-power", type=float, default=0.80)
    power.add_argument("--direction", choices=("greater", "less", "two-sided"), default="two-sided")
    power.add_argument("--out", default="reports/research/power_analysis.json")

    args = parser.parse_args(argv)
    try:
        if args.command == "hypothesis-register":
            registry = HypothesisRegistry(args.db)
            result = registry.register(_read_json(args.protocol))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "hypothesis-evaluate":
            registry = HypothesisRegistry(args.db)
            protocol = _read_json(args.protocol)
            differences = _read_json(args.family_differences)
            result = evaluate_hypothesis(protocol, differences,
                                         bootstrap_repetitions=args.bootstrap, seed=args.seed)
            result_hash = registry.record_result(result["hypothesis_id"], result)
            result["result_sha256"] = result_hash
            atomic_write_json(Path(args.out), result)
            print(f"Hypothesis status: {result['status']}")
            print(f"Result: {Path(args.out).resolve()}")
            return 0
        if args.command == "hypothesis-batch-audit":
            result = audit_hypothesis_batch(_read_json(args.results), alpha=args.alpha)
            atomic_write_json(Path(args.out), result)
            print(f"Hypotheses corrected: {result['n_hypotheses']}")
            print(f"Audit: {Path(args.out).resolve()}")
            return 0
        if args.command == "ablation-plan":
            seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
            result = build_ablation_plan(_read_json(args.base_config), _read_json(args.factors),
                                         mode=args.mode, seeds=seeds, max_variants=args.max_variants)
            atomic_write_json(Path(args.out), result)
            print(f"Variants: {len(result['variants'])}; seeds: {len(result['seeds'])}")
            print(f"Protocol SHA-256: {result['protocol_sha256']}")
            print(f"Plan: {Path(args.out).resolve()}")
            return 0
        if args.command == "ablation-audit":
            result = audit_ablation_results(_read_json(args.results), alpha=args.alpha,
                                            bootstrap_repetitions=args.bootstrap, seed=args.seed)
            atomic_write_json(Path(args.out), result)
            print(f"Variants analysed: {len(result['variants'])}")
            print(f"Audit: {Path(args.out).resolve()}")
            return 0
        if args.command == "archive-stats":
            db = DiscoveryArchive(args.db)
            print(json.dumps({"stats": db.stats(), "recent": db.list(limit=args.limit)}, indent=2))
            return 0
        if args.command == "power-analysis":
            result = power_analysis(minimum_effect=args.minimum_effect, expected_sd=args.expected_sd,
                                    alpha=args.alpha, target_power=args.target_power,
                                    direction=args.direction)
            atomic_write_json(Path(args.out), result)
            print(json.dumps(result, indent=2, sort_keys=True))
            print(f"Power analysis: {Path(args.out).resolve()}")
            return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"research lab failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
