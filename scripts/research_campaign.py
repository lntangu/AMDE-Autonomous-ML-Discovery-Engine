#!/usr/bin/env python3
"""Crash-resumable, multi-round AMDE 1.1 research campaign runner.

Each round receives fresh deterministic benchmark/evaluation seeds and its own
SQLite ledger/checkpoint. The campaign state and leaderboard are atomically
updated. No prior run is overwritten and no failed run is silently restarted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.config import Config, load_config  # noqa: E402
from amde.errors import (  # noqa: E402
    AMDEBaseError,
    CheckpointError,
    ConfigError,
    DataIntegrityError,
    FAILURE_CATEGORY_CONFIG,
    FAILURE_CATEGORY_DATA_INTEGRITY,
    FAILURE_CATEGORY_RESOURCE,
    FAILURE_CATEGORY_RUNTIME,
    LockError,
    ResourcePressure,
    EXIT_CONFIG_ERROR,
    EXIT_DATA_INTEGRITY,
    EXIT_RESOURCE_EXHAUSTION,
    EXIT_RUNTIME_ERROR,
)
from amde.orchestrator import Engine, run_status  # noqa: E402
from amde.runtime.lock import SingleInstanceLock  # noqa: E402
from amde.storage.checkpoint import CheckpointManager  # noqa: E402
from amde.util import atomic_write_json, atomic_write_text, hash_json, read_json, sanitize_id, seed_from, utc_now  # noqa: E402

TERMINAL_ROUND_STATUSES = {"completed", "time_budget", "trial_budget", "exhausted", "no_candidate", "challenge_disabled"}
STOP_ROUND_STATUSES = {"stopped", "paused", "failed", "challenge_failed"}


def _round_seed(base_seed: int, campaign_id: str, round_index: int, label: str, item: str = "") -> int:
    return seed_from("amde-research-campaign-v1", int(base_seed), campaign_id, int(round_index), label, item) % (2**31 - 1)


def build_round_config(
    base: Config,
    *,
    campaign_id: str,
    round_index: int,
    base_seed: int,
    device: str,
    candidate_archive: list[dict[str, Any]] | None = None,
    remaining_campaign_gb: float | None = None,
) -> Config:
    """Create an independently seeded round config with optional genome warm starts."""
    data = base.effective_copy()
    data["seed"] = _round_seed(base_seed, campaign_id, round_index, "experiment")
    for spec in data["benchmarks"]["task_specs"]:
        spec["seed"] = _round_seed(base_seed, campaign_id, round_index, "task", str(spec["task_id"]))
    eval_count = len(data["evaluation"]["seeds"])
    data["evaluation"]["seeds"] = [
        _round_seed(base_seed, campaign_id, round_index, "evaluation", str(i))
        for i in range(eval_count)
    ]
    challenge_count = len(data["challenge"]["seeds"])
    data["challenge"]["seeds"] = [
        _round_seed(base_seed, campaign_id, round_index, "challenge", str(i))
        for i in range(challenge_count)
    ]
    if remaining_campaign_gb is not None:
        remaining = float(remaining_campaign_gb)
        if remaining <= 0:
            raise ConfigError("no campaign storage budget remains for another round")
        data["budgets"]["max_disk_gb"] = min(
            float(data["budgets"].get("max_disk_gb", remaining)), remaining
        )
    data["runtime"]["device"] = device
    # Cross-round candidates are transferred as genomes only. Their old scores
    # are deliberately discarded and the new round must re-evaluate descendants
    # on its own freshly generated cohort.
    data.setdefault("search", {})["seed_genomes"] = [
        item["genome"] for item in (candidate_archive or [])
        if isinstance(item, dict) and isinstance(item.get("genome"), dict)
    ]
    # The engine determines the effective backend after the hardware probe.
    data["runtime"].pop("effective_device", None)
    data["challenge"]["auto_open_during_search"] = False
    return Config(data, base.project_root)


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for current, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in files:
            item = Path(current) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def campaign_bytes(config: Config, campaign_id: str) -> int:
    prefix = f"{campaign_id}_r"
    total = 0
    for root_key in ("runs_dir", "artifacts_dir"):
        root = config.path(root_key)
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.name.startswith(prefix) and child.is_dir():
                total += _tree_bytes(child)
    campaign_root = config.reports_dir() / "campaigns" / campaign_id
    total += _tree_bytes(campaign_root)
    # Per-run Markdown reports live directly under reports/, outside campaign_root.
    for report in config.reports_dir().glob(f"{campaign_id}_r*_report.md"):
        try:
            if report.is_file() and not report.is_symlink():
                total += report.stat().st_size
        except OSError:
            continue
    return total


def _write_leaderboard(path: Path, rounds: list[dict[str, Any]]) -> None:
    fields = ["round", "run_id", "seed", "status", "generations", "trials", "best_candidate", "best_score", "protocol_hash", "started_at", "ended_at"]
    rows = []
    for item in rounds:
        rows.append({
            "round": item.get("round_index"),
            "run_id": item.get("run_id"),
            "seed": item.get("seed"),
            "status": item.get("status"),
            "generations": item.get("generations_completed"),
            "trials": item.get("trials_started"),
            "best_candidate": item.get("best_candidate_hash"),
            "best_score": item.get("best_score"),
            "protocol_hash": item.get("protocol_hash"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
        })
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_confirmed") or summary.get("best_candidate") or {}
    return {
        "status": summary.get("status"),
        "generations_completed": summary.get("generations_completed"),
        "trials_started": summary.get("trials_started"),
        "best_candidate_hash": best.get("candidate_hash"),
        "best_score": best.get("confirm_score", best.get("quick_score")),
        "protocol_hash": summary.get("protocol_hash"),
        "error": summary.get("error"),
        "run_dir": summary.get("run_dir"),
    }


def _campaign_report(path: Path, state: dict[str, Any]) -> None:
    rounds = state.get("rounds", [])
    completed = [r for r in rounds if r.get("status") in TERMINAL_ROUND_STATUSES]
    scores = [float(r["best_score"]) for r in completed if isinstance(r.get("best_score"), (int, float))]
    best = max((r for r in completed if isinstance(r.get("best_score"), (int, float))), key=lambda r: float(r["best_score"]), default=None)
    lines = [
        f"# AMDE Research Campaign — {state['campaign_id']}",
        "",
        f"- State: `{state.get('status')}`",
        f"- Rounds recorded: {len(rounds)}",
        f"- Rounds with a terminal budget result: {len(completed)}",
        f"- Base seed: `{state['base_seed']}`",
        f"- Device request: `{state['device']}`",
        f"- Per-round time budget: `{state['hours_per_round']}` hours",
        f"- Per-round trial budget: `{state['max_trials_per_round']}`",
        f"- Cross-round genome archive: `{len(state.get('candidate_archive', []))}` candidates",
        "",
        "## Best recorded development candidate",
        "",
    ]
    if best:
        lines.extend([
            f"- Run: `{best.get('run_id')}`",
            f"- Candidate: `{best.get('best_candidate_hash')}`",
            f"- Score: `{best.get('best_score')}`",
            "",
            "This is a development-search result, not independent held-out evidence.",
        ])
    else:
        lines.append("No confirmed candidate score has been recorded yet.")
    if scores:
        lines.extend(["", f"Observed best-score range: `{min(scores):.6f}` to `{max(scores):.6f}` (not a statistical confidence interval)."])
    lines.extend([
        "",
        "## Scientific guardrails",
        "",
        "- Each round uses fresh deterministic seeds for task generation, evaluation, search, and sealed challenge setup.",
        "- Round-specific data manifests and protocol hashes are stored in each run directory.",
        "- The sealed challenge is never opened automatically during search.",
        "- Development scores must not be described as generalization or proof of a new learning algorithm.",
        "- No automatic deletion or pruning of run evidence is performed. The campaign pauses at its storage cap.",
        "- Long-duration operation requires machine maintenance, backups, thermal/power monitoring, and periodic human review.",
        "",
        "## Rounds",
        "",
        "| Round | Run | Status | Generations | Trials | Best score |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for item in rounds:
        score = item.get("best_score")
        lines.append(
            f"| {item.get('round_index')} | `{item.get('run_id')}` | {item.get('status')} | "
            f"{item.get('generations_completed', '')} | {item.get('trials_started', '')} | "
            f"{'' if score is None else f'{float(score):.6f}'} |"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


def _confirmed_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Read confirmed genomes from a checksum-verified final checkpoint.

    The public summary carries the best confirmed candidate. The checkpoint
    additionally preserves other confirmed candidates still in the population.
    A corrupt checkpoint is never trusted for cross-round warm starts.
    """
    candidates: dict[str, dict[str, Any]] = {}
    best = summary.get("best_confirmed")
    if isinstance(best, dict) and isinstance(best.get("genome"), dict) and best.get("candidate_hash"):
        candidates[str(best["candidate_hash"])] = best
    run_dir = summary.get("run_dir")
    if run_dir:
        try:
            state = CheckpointManager(Path(str(run_dir))).load()
            if isinstance(state, dict):
                members = state.get("population", {}).get("members", [])
                if isinstance(members, list):
                    for member in members:
                        if (
                            isinstance(member, dict)
                            and member.get("confirm_score") is not None
                            and isinstance(member.get("genome"), dict)
                            and member.get("candidate_hash")
                        ):
                            candidates[str(member["candidate_hash"])] = member
        except Exception as exc:
            # Do not fail a completed experiment because of optional archival.
            # The candidate from the run summary remains available if present.
            print(f"warning: verified checkpoint archive unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
    return sorted(
        candidates.values(),
        key=lambda item: float(item.get("confirm_score") or -1.0),
        reverse=True,
    )


def _update_candidate_archive(state: dict[str, Any], summary: dict[str, Any] | None, *, max_items: int = 12, max_new_per_round: int = 3) -> None:
    """Retain a small diverse-in-time set of confirmed genome structures.

    Old scores are provenance only and are not compared across cohorts. Every
    imported genome is re-evaluated on the next round's fresh task/eval seeds.
    """
    if not isinstance(summary, dict):
        return
    archive = list(state.get("candidate_archive", []))
    for candidate in _confirmed_candidates(summary)[:max_new_per_round]:
        candidate_hash = str(candidate.get("candidate_hash") or "")
        genome = candidate.get("genome")
        if not candidate_hash or not isinstance(genome, dict):
            continue
        archive = [item for item in archive if item.get("candidate_hash") != candidate_hash]
        archive.append({
            "candidate_hash": candidate_hash,
            "genome": genome,
            "source_run_id": summary.get("run_id"),
            "source_generation": candidate.get("generation"),
            "source_confirm_score": candidate.get("confirm_score"),
            "archived_at": utc_now(),
        })
    # FIFO by cohort/round; never rank historical scores from different cohorts.
    state["candidate_archive"] = archive[-max_items:]


def run_campaign(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    base = load_config(project_root=root, config_path=args.config)
    campaign_id = sanitize_id(args.campaign_id, fallback="research_campaign")
    if len(campaign_id) > 48:
        raise ConfigError("campaign id must be 48 characters or fewer")
    if args.rounds < 0 or args.hours_per_round <= 0 or args.max_trials_per_round <= 0:
        raise ConfigError("rounds must be >= 0 and per-round hours/trials must be positive")
    if args.max_campaign_gb <= 0 or args.min_free_gb < 1:
        raise ConfigError("storage limits must be positive (min-free-gb >= 1)")
    if args.device not in ("auto", "cpu", "cuda"):
        raise ConfigError("device must be auto, cpu, or cuda")

    base_seed = int(args.seed)
    plan = {
        "schema_version": "amde-campaign-v1",
        "campaign_id": campaign_id,
        "base_seed": base_seed,
        "rounds": int(args.rounds),
        "hours_per_round": float(args.hours_per_round),
        "max_trials_per_round": int(args.max_trials_per_round),
        "max_campaign_gb": float(args.max_campaign_gb),
        "min_free_gb": float(args.min_free_gb),
        "device": args.device,
        "base_config_hash": base.config_hash,
        "base_protocol_hash": base.protocol_hash,
    }
    plan_hash = hash_json(plan)
    campaign_root = base.reports_dir() / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    state_path = campaign_root / "state.json"
    leaderboard_path = campaign_root / "leaderboard.csv"
    report_path = campaign_root / "REPORT.md"
    campaign_stop_arg = getattr(args, "campaign_stop_file", None)
    isolated_stop = bool(getattr(args, "isolated_stop", False))
    campaign_stop_path = (Path(campaign_stop_arg).resolve() if campaign_stop_arg else campaign_root / "STOP")

    def campaign_stop_requested() -> bool:
        if campaign_stop_path.exists():
            return True
        return (not isolated_stop) and (base.path("runs_dir") / "STOP").exists()

    lock = SingleInstanceLock(campaign_root / "campaign.lock", run_id=campaign_id, stale_seconds=300)
    lock.acquire()
    try:
        if state_path.exists():
            state = read_json(state_path)
            if not isinstance(state, dict) or state.get("plan_hash") != plan_hash:
                raise ConfigError("existing campaign state has a different immutable plan; use a new campaign id or the exact original arguments")
        else:
            state = {
                **plan,
                "plan_hash": plan_hash,
                "status": "initialized",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "next_round": 0,
                "rounds": [],
                "candidate_archive": [],
            }
            atomic_write_json(state_path, state)

        max_bytes = int(args.max_campaign_gb * (1024**3))
        min_free_bytes = int(args.min_free_gb * (1024**3))
        print(f"AMDE campaign {campaign_id}; rounds={'infinite' if args.rounds == 0 else args.rounds}; device={args.device}")
        print(f"State: {state_path}")

        while args.rounds == 0 or int(state["next_round"]) < args.rounds:
            if campaign_stop_requested():
                state["status"] = "stopped"
                state["stop_requested_at"] = utc_now()
                break
            used = campaign_bytes(base, campaign_id)
            disk_free = shutil.disk_usage(base.project_root).free
            if used >= max_bytes:
                state["status"] = "storage_cap_reached"
                print(f"Campaign storage cap reached: {used / 1024**3:.2f} / {args.max_campaign_gb:.2f} GiB")
                break
            if disk_free < min_free_bytes:
                state["status"] = "low_disk_space"
                print(f"Free disk space below safety floor: {disk_free / 1024**3:.2f} GiB")
                break

            round_index = int(state["next_round"])
            run_id = sanitize_id(f"{campaign_id}_r{round_index:05d}")
            remaining_gb = max(0.01, args.max_campaign_gb - used / (1024**3))
            round_cfg = build_round_config(
                base,
                campaign_id=campaign_id,
                round_index=round_index,
                base_seed=base_seed,
                device=args.device,
                candidate_archive=state.get("candidate_archive", []),
                remaining_campaign_gb=remaining_gb,
            )
            record = next((r for r in state["rounds"] if int(r.get("round_index", -1)) == round_index), None)
            if record and record.get("status") in TERMINAL_ROUND_STATUSES:
                state["next_round"] = round_index + 1
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                continue

            # If the process died after the engine finished but before campaign
            # state was committed, read the existing ledger before resuming.
            if record and record.get("status") == "running":
                try:
                    prior = run_status(round_cfg, run_id)
                    prior_status = prior.get("status")
                    if prior_status in TERMINAL_ROUND_STATUSES:
                        prior_summary = prior.get("summary") or {}
                        compact = _compact_summary(prior_summary)
                        _update_candidate_archive(state, prior_summary)
                        record.update(compact)
                        record["status"] = prior_status
                        record["ended_at"] = utc_now()
                        state["next_round"] = round_index + 1
                        state["updated_at"] = utc_now()
                        atomic_write_json(state_path, state)
                        _write_leaderboard(leaderboard_path, state["rounds"])
                        _campaign_report(report_path, state)
                        continue
                except ConfigError:
                    # Let Engine enforce fail-closed handling for an existing
                    # run folder with a missing ledger/checkpoint.
                    pass

            if record is None:
                record = {
                    "round_index": round_index,
                    "run_id": run_id,
                    "seed": round_cfg.get("seed"),
                    "status": "running",
                    "started_at": utc_now(),
                    "protocol_hash": round_cfg.protocol_hash,
                }
                state["rounds"].append(record)
            else:
                record["status"] = "running"
                record["resumed_at"] = utc_now()
            state["status"] = "running"
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            print(f"\n[{round_index}] START {run_id} seed={round_cfg.get('seed')} storage={used / 1024**3:.2f} GiB")
            try:
                summary = Engine(round_cfg).run(
                    mode="continuous",
                    run_id=run_id,
                    max_seconds=float(args.hours_per_round) * 3600.0,
                    max_trials=int(args.max_trials_per_round),
                    resume=True,
                    stop_callback=campaign_stop_requested,
                    isolate_stop=isolated_stop,
                )
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                record["ended_at"] = utc_now()
                state["status"] = "interrupted"
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                raise
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["ended_at"] = utc_now()
                state["status"] = "failed"
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                _write_leaderboard(leaderboard_path, state["rounds"])
                _campaign_report(report_path, state)
                raise

            compact = _compact_summary(summary)
            _update_candidate_archive(state, summary)
            record.update(compact)
            record["status"] = str(summary.get("status", "failed"))
            record["ended_at"] = utc_now()
            state["updated_at"] = utc_now()
            if record["status"] in TERMINAL_ROUND_STATUSES:
                state["next_round"] = round_index + 1
                state["status"] = "running"
            elif record["status"] == "stopped":
                state["status"] = "stopped"
            else:
                state["status"] = record["status"]
            atomic_write_json(state_path, state)
            _write_leaderboard(leaderboard_path, state["rounds"])
            _campaign_report(report_path, state)
            print(f"[{round_index}] {record['status']} generations={record.get('generations_completed')} trials={record.get('trials_started')} best={record.get('best_score')}")
            if record["status"] in STOP_ROUND_STATUSES or record["status"] not in TERMINAL_ROUND_STATUSES:
                break

        if state.get("status") == "running":
            state["status"] = "round_limit_reached" if args.rounds else "running"
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)
        _write_leaderboard(leaderboard_path, state["rounds"])
        _campaign_report(report_path, state)
        print(f"Campaign state: {state['status']}; next_round={state['next_round']}; recorded={len(state['rounds'])}")
        print(f"Report: {report_path}")
        return 0 if state["status"] in {"running", "round_limit_reached", "stop_requested", "stopped", "paused"} else 1
    finally:
        lock.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-resumable AMDE 1.0 multi-round research campaign runner")
    parser.add_argument("--root", default=None, help="Project root (defaults to current directory)")
    parser.add_argument("--config", default=None, help="Alternative config JSON")
    parser.add_argument("--campaign-id", required=True, help="Stable campaign identifier; do not reuse for a different plan")
    parser.add_argument("--rounds", type=int, default=5, help="Number of fresh-seed rounds; 0 runs until stopped")
    parser.add_argument("--hours-per-round", type=float, default=1.0, help="Wall-clock budget per round")
    parser.add_argument("--max-trials-per-round", type=int, default=400, help="Trial budget per round")
    parser.add_argument("--max-campaign-gb", type=float, default=10.0, help="Pause when campaign-owned run/artifact data reaches this size")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free-disk safety floor")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; each round derives independent task/search/evaluation seeds")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu",
        help="Training backend (default: cpu; opt into auto/cuda after throughput comparison)",
    )
    parser.add_argument("--campaign-stop-file", default=None, help="Campaign-scoped cooperative stop marker")
    parser.add_argument("--isolated-stop", action="store_true", help="Ignore project-wide legacy STOP marker")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_campaign(args)
    except KeyboardInterrupt:
        print("campaign interrupted; committed state is available for restart", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"campaign configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (DataIntegrityError, CheckpointError) as exc:
        print(f"campaign data integrity failure: {exc}", file=sys.stderr)
        return EXIT_DATA_INTEGRITY
    except ResourcePressure as exc:
        print(f"campaign resource exhaustion: {exc}", file=sys.stderr)
        return EXIT_RESOURCE_EXHAUSTION
    except (AMDEBaseError, OSError, ValueError) as exc:
        print(f"campaign runtime error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
