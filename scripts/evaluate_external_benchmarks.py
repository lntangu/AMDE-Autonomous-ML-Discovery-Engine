#!/usr/bin/env python3
"""Paired candidate-vs-baseline evaluation on explicitly supplied CSV families."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amde.baselines.registry import BASELINE_NAMES, make_baseline_genome  # noqa: E402
from amde.benchmarks.external import (  # noqa: E402
    build_generalization_input,
    candidate_genome_from_json,
    evaluate_external_baseline,
    evaluate_external_candidate,
    load_external_manifest,
    EXTERNAL_BASELINE_NAMES,
    external_baseline_identity,
)
from amde.evaluation.generalization import audit_generalization  # noqa: E402
from amde.util import hash_json  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_markdown(path: Path, payload: dict[str, Any], audit: dict[str, Any]) -> None:
    lines = [
        "# AMDE External Benchmark Evaluation",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Baseline: `{payload['baseline_id']}`",
        f"- Protocol hash: `{payload['benchmark_protocol_hash']}`",
        f"- Task: `{payload['task_kind']}` / `{payload['metric_name']}`",
        f"- Device: `{payload['device']}`",
        f"- Families: {len(payload['families'])}",
        f"- Seeds per family: {len(payload['seeds'])}",
        f"- Evidence tier: **{audit['evidence_tier']}**",
        f"- Internal generalization gate: `{audit['internal_generalization_gate']}`",
        f"- Multi-domain external gate: `{audit['external_generalization_gate']}`",
        "",
        "## Paired family-level results",
        "",
        "| Family | Domain | Source | Candidate | Baseline | Delta | Candidate steps | Baseline steps | Candidate sec | Baseline sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    candidate = {item["family_id"]: item for item in payload["candidate_results"]}
    baseline = {item["family_id"]: item for item in payload["baseline_results"]}
    for meta in payload["families"]:
        family_id = meta["family_id"]
        c = candidate[family_id]
        b = baseline[family_id]
        lines.append(
            f"| `{family_id}` | {meta['domain']} | {meta['source']} | "
            f"{c['family_mean_score']:.6f} | {b['family_mean_score']:.6f} | "
            f"{c['family_mean_score'] - b['family_mean_score']:+.6f} | "
            f"{sum(row['steps'] for row in c['seed_records'])} | "
            f"{sum(row['steps'] for row in b['seed_records'])} | "
            f"{sum(row['elapsed_seconds'] for row in c['seed_records']):.2f} | "
            f"{sum(row['elapsed_seconds'] for row in b['seed_records']):.2f} |"
        )
    lines += [
        "",
        "## Statistical audit",
        "",
        f"- Mean paired delta: {audit['mean_paired_delta']:+.6f}",
        f"- Family-cluster bootstrap 95% CI: [{audit['bootstrap_ci_95'][0]:+.6f}, {audit['bootstrap_ci_95'][1]:+.6f}]",
        f"- Paired sign-flip p-value: {audit['permutation_test']['p_value']:.6g}",
        f"- Family wins/losses/ties: {audit['family_wins']}/{audit['family_losses']}/{audit['family_ties']}",
        f"- Worst family regression: {audit['maximum_family_regression']:.6f} (`{audit['worst_family']}`)",
        "",
        "## Interpretation and limitations",
        "",
        audit["interpretation"],
        "",
        "Dataset provenance, independence, license compliance, preregistration and the sealed-holdout declaration are researcher-supplied attestations. AMDE validates data shape and records hashes; it cannot authenticate publisher claims or prove that evaluation data were never used for model selection.",
        "",
        "Candidate and baseline use the same family data, deterministic group splits, seeds, evaluator, device, maximum-step cap and parameter cap. Their recipes can realize different step counts and wall-clock costs; inspect the recorded per-seed steps and elapsed time before claiming compute parity.",
        "",
        "This report is evidence about the listed datasets, domains and protocol only. It is not a proof of universal generalization.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON manifest using amde-external-datasets-v1")
    parser.add_argument("candidate", help="Frozen candidate genome JSON (or object with a genome member)")
    parser.add_argument("--baseline", default="mlp_medium",
                        help=f"Fixed baseline; choices: {', '.join(EXTERNAL_BASELINE_NAMES)}")
    parser.add_argument("--seeds", default="11,23,37", help="Comma-separated deterministic split/training seeds (at least 3)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-rows", type=int, default=200000, help="Maximum CSV rows per family loaded into memory")
    parser.add_argument("--parameter-limit", type=int, default=20000)
    parser.add_argument("--epochs-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument("--minimum-effect", type=float, default=0.01)
    parser.add_argument("--minimum-win-rate", type=float, default=0.65)
    parser.add_argument("--maximum-family-regression", type=float, default=0.10)
    parser.add_argument("--preregistered", action="store_true", help="Attest that the protocol and decision rule were registered before final results were revealed")
    parser.add_argument("--sealed-holdout", action="store_true", help="Attest that these evaluation families were not used for search, tuning, or candidate selection")
    parser.add_argument("--out", default="reports/external_benchmark_evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume completed family evaluations from the output folder after verifying the protocol hash")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace a prior partial or completed evaluation in the output folder")
    args = parser.parse_args(argv)

    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError("provide at least three distinct seeds")
        if args.max_steps < 1 or args.max_rows < 30 or args.parameter_limit < 1 or not (0.05 <= args.epochs_scale <= 10.0):
            raise ValueError("invalid training budget")
        if not (0 <= args.minimum_win_rate <= 1) or args.maximum_family_regression < 0:
            raise ValueError("invalid statistical thresholds")
        if args.bootstrap < 1000 or args.bootstrap > 1_000_000:
            raise ValueError("--bootstrap must be between 1000 and 1,000,000")

        families, manifest_sha = load_external_manifest(args.manifest, max_rows=args.max_rows)
        candidate = candidate_genome_from_json(args.candidate, config_path_root=ROOT)
        baseline_genome = make_baseline_genome(args.baseline) if args.baseline in BASELINE_NAMES else None
        baseline_identity = external_baseline_identity(args.baseline)
        baseline_hash = baseline_identity["identity_sha256"]
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        family_hashes = {item.family_id: item.file_sha256 for item in families}
        protocol = {
            "protocol_id": "amde-external-tabular-paired-v3",
            "manifest_sha256": manifest_sha,
            "family_csv_sha256": family_hashes,
            "candidate_genome_sha256": candidate.hash,
            "baseline_id": args.baseline,
            "baseline_implementation": baseline_identity,
            "baseline_genome_sha256": baseline_hash,
            "task_kind": families[0].task_kind,
            "seeds": seeds,
            "split": "grouped_duplicate_feature_vectors_70_15_15_train_fitted_median_missing_indicator_onehot_v2",
            "metric_version": "amde-evaluation-metrics-v1",
            "device": args.device,
            "max_steps": args.max_steps,
            "parameter_limit": args.parameter_limit,
            "epochs_scale": args.epochs_scale,
            "minimum_families": 10,
            "minimum_external_families": 10,
            "minimum_domains": 3,
            "minimum_effect": args.minimum_effect,
            "minimum_win_rate": args.minimum_win_rate,
            "maximum_family_regression": args.maximum_family_regression,
            "bootstrap_repetitions": args.bootstrap,
            "audit_seed": args.audit_seed,
            "preregistered_attestation": bool(args.preregistered),
            "sealed_holdout_attestation": bool(args.sealed_holdout),
        }
        protocol_hash = hash_json(protocol)
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        partial_path = out / "external_evaluation.partial.json"
        final_paths = [out / "external_evaluation.json", out / "generalization_audit.json", out / "protocol.json", out / "EXTERNAL_BENCHMARK_REPORT.md"]
        if args.resume and args.overwrite:
            raise ValueError("choose --resume or --overwrite, not both")
        if args.overwrite:
            partial_path.unlink(missing_ok=True)
            for final_path in final_paths:
                final_path.unlink(missing_ok=True)
        elif any(path.exists() for path in final_paths):
            raise FileExistsError("completed evaluation files already exist; choose a new --out folder or pass --overwrite")
        print(f"AMDE external evaluation: {len(families)} families; candidate={candidate.short_id}; baseline={args.baseline}")
        print(f"protocol_sha256={protocol_hash}")

        candidate_results: list[dict[str, Any]] = []
        baseline_results: list[dict[str, Any]] = []
        if partial_path.exists():
            if not args.resume:
                raise FileExistsError(f"partial evaluation exists at {partial_path}; pass --resume or move it aside")
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if partial.get("schema_version") != "amde-external-evaluation-partial-v1":
                raise ValueError("unrecognized partial evaluation schema")
            if partial.get("protocol_hash") != protocol_hash:
                raise ValueError("partial evaluation protocol hash differs; refusing resume")
            candidate_results = list(partial.get("candidate_results", []))
            baseline_results = list(partial.get("baseline_results", []))
            completed_ids = [item.get("family_id") for item in candidate_results]
            baseline_ids = [item.get("family_id") for item in baseline_results]
            if completed_ids != baseline_ids or completed_ids != partial.get("completed_family_ids") or len(set(completed_ids)) != len(completed_ids):
                raise ValueError("partial evaluation candidate/baseline family sets are inconsistent")
            if not set(completed_ids).issubset({item.family_id for item in families}):
                raise ValueError("partial evaluation contains families absent from the current manifest")
            started_unix = float(partial.get("started_unix", time.time()))
        else:
            if args.resume:
                print("No partial state found; starting a fresh evaluation.")
            completed_ids = []
            started_unix = time.time()
        for index, family in enumerate(families, start=1):
            if family.family_id in completed_ids:
                print(f"[{index}/{len(families)}] {family.family_id} already checkpointed; skipping", flush=True)
                continue
            print(f"[{index}/{len(families)}] {family.family_id} ({family.domain}, {family.task_kind})", flush=True)
            c = evaluate_external_candidate(
                candidate, family, seeds=seeds, device=args.device,
                max_steps=args.max_steps, parameter_limit=args.parameter_limit,
                epochs_scale=args.epochs_scale,
            )
            b = evaluate_external_baseline(
                args.baseline, family, seeds=seeds, device=args.device,
                max_steps=args.max_steps, parameter_limit=args.parameter_limit,
                epochs_scale=args.epochs_scale,
            )
            candidate_results.append(c)
            baseline_results.append(b)
            completed_ids.append(family.family_id)
            _write_json(partial_path, {
                "schema_version": "amde-external-evaluation-partial-v1",
                "protocol_hash": protocol_hash,
                "completed_family_ids": completed_ids,
                "started_unix": started_unix,
                "candidate_results": candidate_results,
                "baseline_results": baseline_results,
                "updated_at_unix": time.time(),
            })
            print(f"    candidate={c['family_mean_score']:.5f} baseline={b['family_mean_score']:.5f} delta={c['family_mean_score'] - b['family_mean_score']:+.5f}", flush=True)

        if set(completed_ids) != {item.family_id for item in families}:
            raise RuntimeError("external evaluation ended without a complete family set")
        # Restore manifest order after an interrupted/resumed execution.
        candidate_by_id = {item["family_id"]: item for item in candidate_results}
        baseline_by_id = {item["family_id"]: item for item in baseline_results}
        candidate_results = [candidate_by_id[item.family_id] for item in families]
        baseline_results = [baseline_by_id[item.family_id] for item in families]

        payload = build_generalization_input(
            candidate.hash, args.baseline, candidate_results, baseline_results, families,
            protocol_hash=protocol_hash,
            preregistered=args.preregistered,
            evaluation_used_for_selection=not args.sealed_holdout,
        )
        payload.update({
            "task_kind": families[0].task_kind,
            "metric_name": "balanced_accuracy" if families[0].task_kind == "classification" else "clipped_r2",
            "device": args.device,
            "seeds": seeds,
            "protocol": protocol,
            "manifest_sha256": manifest_sha,
            "candidate_genome": candidate.to_dict(),
            "baseline_genome": baseline_genome.to_dict() if baseline_genome is not None else None,
            "baseline_identity": baseline_identity,
            "families": [
                {
                    "family_id": family.family_id,
                    "domain": family.domain,
                    "source": family.source,
                    "source_version": family.source_version,
                    "license": family.license,
                    "reference": family.reference,
                    "csv_sha256": family.file_sha256,
                    "seen_during_search": family.seen_during_search,
                    "n_rows": len(family.rows),
                    "n_features": len(family.feature_columns),
                }
                for family in families
            ],
            "candidate_results": candidate_results,
            "baseline_results": baseline_results,
            "elapsed_seconds": time.time() - started_unix,
        })
        audit = audit_generalization(
            payload["candidate_family_scores"], payload["baseline_family_scores"],
            family_metadata=payload["family_metadata"],
            preregistered=payload["preregistered"],
            evaluation_used_for_selection=payload["evaluation_used_for_selection"],
            minimum_families=payload["minimum_families"],
            minimum_external_families=payload["minimum_external_families"],
            minimum_domains=payload["minimum_domains"],
            minimum_effect=args.minimum_effect,
            minimum_win_rate=args.minimum_win_rate,
            maximum_family_regression=args.maximum_family_regression,
            bootstrap_repetitions=args.bootstrap,
            seed=args.audit_seed,
        )
        audit.update({"candidate_id": candidate.hash, "baseline_id": args.baseline, "benchmark_protocol_hash": protocol_hash})

        _write_json(out / "external_evaluation.json", payload)
        _write_json(out / "generalization_audit.json", audit)
        _write_json(out / "protocol.json", protocol)
        _write_markdown(out / "EXTERNAL_BENCHMARK_REPORT.md", payload, audit)
        partial_path.unlink(missing_ok=True)
        print(f"Evidence tier: {audit['evidence_tier']}")
        print(f"Internal gate: {audit['internal_generalization_gate']}")
        print(f"External gate: {audit['external_generalization_gate']}")
        print(f"Report: {out / 'EXTERNAL_BENCHMARK_REPORT.md'}")
        return 0
    except Exception as exc:
        print(f"external benchmark evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
