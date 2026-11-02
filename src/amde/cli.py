"""Command-line interface for AMDE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, apply_cli_overrides, load_config
from .errors import AMDEBaseError, ConfigError
from .orchestrator import (
    Engine,
    engine_doctor,
    generate_report,
    latest_run_id,
    run_status,
    smoke_config,
    stop_run,
)
from .storage.ledger import Ledger
from .util import human_duration, sanitize_id

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_root() -> Path:
    env = os.environ.get("AMDE_ROOT")
    if env:
        return Path(env)
    cwd = Path.cwd()
    if (cwd / "config" / "default.json").exists():
        return cwd
    return _PROJECT_ROOT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amde",
        description=(
            "Autonomous ML Discovery Engine: bounded local search over safe ML "
            "recipes with a persistent ledger and sealed challenge boundary."
        ),
    )
    parser.add_argument("--version", action="version", version=f"amde {__version__}")
    parser.add_argument(
        "--root",
        default=None,
        help="Project root containing config/ (default: current project or AMDE_ROOT)",
    )
    parser.add_argument("--config", default=None, help="Alternative config JSON path")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic engine seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect environment, resources, config, and ARC status")
    doctor.add_argument("--probe-torch", action="store_true", help="Run a bounded CUDA tensor smoke test")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda"), default=argparse.SUPPRESS)

    smoke = sub.add_parser("smoke", help="Run one bounded smoke generation through all A-D levels")
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--max-seconds", type=float, default=None)
    smoke.add_argument("--json", action="store_true")
    smoke.add_argument("--no-resume", action="store_true")

    once = sub.add_parser("once", help="Run a bounded number of generations")
    once.add_argument("--run-id", default=None)
    once.add_argument("--generations", type=int, default=None)
    once.add_argument("--max-trials", type=int, default=None)
    once.add_argument("--max-seconds", type=float, default=None)
    once.add_argument("--json", action="store_true")
    once.add_argument("--no-resume", action="store_true")

    continuous = sub.add_parser(
        "continuous",
        help="Run until stopped or a time/generation/trial budget is reached",
    )
    continuous.add_argument("--run-id", default=None)
    continuous.add_argument("--max-generations", type=int, default=0)
    continuous.add_argument("--max-hours", type=float, default=0.0)
    continuous.add_argument("--max-seconds", type=float, default=0.0)
    continuous.add_argument("--max-trials", type=int, default=None)
    continuous.add_argument("--json", action="store_true")
    continuous.add_argument("--no-resume", action="store_true")

    status = sub.add_parser("status", help="Show run status from the ledger")
    status.add_argument("--run-id", default=None)
    status.add_argument("--all", action="store_true", help="List all runs")
    status.add_argument("--json", action="store_true")

    stop = sub.add_parser("stop", help="Request a graceful stop through a stop file")
    stop.add_argument("--run-id", default=None)
    stop.add_argument("--global", dest="global_stop", action="store_true", help="Also write runs/STOP")

    challenge = sub.add_parser("challenge", help="Open the sealed challenge for a completed run")
    challenge.add_argument("--run-id", default=None)
    challenge.add_argument("--candidate", default=None, help="Candidate hash or prefix")
    challenge.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="Generate a Markdown report for a run")
    report.add_argument("--run-id", default=None)

    inspect = sub.add_parser("inspect", help="Inspect candidate genomes and trials")
    inspect.add_argument("--run-id", default=None)
    inspect.add_argument("--candidate", default=None, help="Candidate hash or prefix; defaults to best")
    inspect.add_argument("--limit", type=int, default=10)
    inspect.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _load(args: argparse.Namespace, *, smoke: bool = False) -> Config:
    root = Path(args.root).resolve() if args.root else default_root().resolve()
    config = load_config(project_root=root, config_path=args.config)
    device = getattr(args, "device", None)
    config = apply_cli_overrides(config, seed=args.seed, device=device)
    if smoke:
        config = smoke_config(config)
    return config


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Run:        {summary.get('run_id')}",
        f"Status:     {summary.get('status')}",
        f"Elapsed:    {summary.get('elapsed_human')}",
        f"Generations:{summary.get('generations_completed')}",
        f"Trials:     {summary.get('trials_started')}",
        f"Run dir:    {summary.get('run_dir')}",
    ]
    best = summary.get("best_confirmed") or summary.get("best_candidate")
    if best:
        lines.append(
            f"Best:       {best['candidate_hash'][:12]} level={best['level']} "
            f"quick={best.get('quick_score')} confirm={best.get('confirm_score')}"
        )
    baselines = summary.get("baselines") or []
    if baselines:
        parts = [
            f"{item['name']}={item['mean']:.4f}"
            for item in baselines
            if item.get("mean") is not None
        ]
        lines.append("Baselines:  " + ", ".join(parts))
    if summary.get("promoted"):
        lines.append("Promotion:  guard passed; incumbent updated")
    arc = (summary.get("arc") or {}).get("profile") or {}
    if arc:
        lines.append(f"ARC:        {arc.get('status')} ({arc.get('reason')})")
    if summary.get("error"):
        lines.append(f"Error:      {summary['error']['type']}: {summary['error']['message']}")
    return "\n".join(lines)


def _doctor_text(info: dict[str, Any]) -> str:
    hardware = info["hardware"]
    plan = info["resource_plan"]
    arc = info["arc"]
    lines = [
        f"AMDE {info['engine_version']}",
        f"Project:     {info['project_root']}",
        f"Python:      {hardware['python_version']} ({hardware['python_executable']})",
        f"OS:          {hardware['os']}",
        f"CPU:         {hardware['logical_cores']} logical / {hardware['physical_cores']} physical",
        f"RAM:         {hardware['ram_total_mb']:.0f} MB total, {hardware['ram_available_mb']:.0f} MB available",
        f"Disk:        {hardware['disk_free_gb']:.1f} GB free",
        f"GPU:         {hardware['gpu_name']}" if hardware["gpu_available"] else "GPU:         not available",
    ]
    if hardware["gpu_available"]:
        lines.append(
            f"             driver={hardware['gpu_driver']} free={hardware['gpu_memory_free_mb']} MB "
            f"temp={hardware['gpu_temperature_c']}C compute={hardware['gpu_compute_capability']}"
        )
    lines.extend(
        [
            f"Plan:        workers={plan['cpu_workers']} device={plan['device_effective']} "
            f"gpu_slots={plan['gpu_slots']}",
            f"             {plan['reason']}",
            f"ARC:         {arc['status']} — {arc['reason']}",
            f"Config hash: {info['config_hash'][:16]}",
            f"Protocol:    {info['protocol_hash'][:16]}",
        ]
    )
    if info.get("policy_warnings"):
        lines.append("Warnings:    " + "; ".join(info["policy_warnings"]))
    return "\n".join(lines)


def _status_text(status: dict[str, Any]) -> str:
    if status.get("status") == "no_runs":
        return f"No runs found under {status.get('runs_dir')}"
    lines = [
        f"Run:         {status['run_id']}",
        f"Status:      {status['status']}",
        f"Run dir:     {status['run_dir']}",
        f"Lock alive:  {status['lock']['alive']}",
        f"Stop file:   {status['stop_requested']}",
    ]
    heartbeat = status.get("heartbeat")
    if heartbeat:
        lines.append(
            f"Heartbeat:   {heartbeat.get('status')} gen={heartbeat.get('generation')} "
            f"at {heartbeat.get('updated_at')}"
        )
    checkpoints = status.get("checkpoints") or []
    if checkpoints:
        latest = checkpoints[0]
        lines.append(f"Checkpoint:  gen={latest.get('generation')} valid={latest.get('valid')}")
    lines.append(f"Trials:      {status.get('trial_counts')}")
    lines.append("Baselines:")
    for item in status.get("baselines", []):
        mean = item.get("mean")
        lines.append(
            f"  {item['name']:<14} {item['status']:<8} "
            f"{'—' if mean is None else f'{mean:.4f}'} worst={item.get('worst_family') or '—'}"
        )
    generations = status.get("generations") or []
    if generations:
        lines.append("Generations:")
        for row in generations[-10:]:
            lines.append(
                f"  {row['generation']:>3} {row['status']:<16} proposals={row.get('proposals', 0)} "
                f"valid={row.get('valid', 0)} confirms={row.get('confirmations', 0)} "
                f"promotions={row.get('promotions', 0)}"
            )
    best = status.get("best_dev")
    if best:
        aggregate = best.get("aggregate") or {}
        lines.append(
            f"Best dev:    {best['candidate_hash'][:12]} mean={aggregate.get('mean')} "
            f"stage={best.get('stage')}"
        )
    latest = status.get("latest_challenge")
    if latest:
        lines.append(
            f"Challenge:   opening={latest.get('opening_index')} cohort={latest.get('cohort')}"
        )
    return "\n".join(lines)


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = _load(args)
    # In auto/CUDA mode, the doctor must run the actual optional tensor smoke
    # test; otherwise it may report the GPU but cannot confirm training support.
    probe_torch = bool(args.probe_torch or config.get("runtime.device", "auto") != "cpu")
    info = engine_doctor(config, probe_torch=probe_torch)
    if args.json:
        _print_json(info)
    else:
        print(_doctor_text(info))
    return 0


def _cmd_run(args: argparse.Namespace, mode: str) -> int:
    config = _load(args, smoke=(mode == "smoke"))
    engine = Engine(config)
    max_seconds = getattr(args, "max_seconds", None)
    max_trials = getattr(args, "max_trials", None)
    generations = getattr(args, "generations", None)
    if mode == "continuous":
        generations = args.max_generations or None
        hours_seconds = float(args.max_hours or 0.0) * 3600.0
        seconds = max(float(args.max_seconds or 0.0), hours_seconds)
        max_seconds = seconds or None
        if not generations and not max_seconds and not config.get("budgets.max_trials", 0):
            print(
                "warning: continuous mode has no time/generation/trial bound; use STOP to stop it",
                file=sys.stderr,
            )
    summary = engine.run(
        mode=mode,
        run_id=args.run_id,
        generations=generations,
        max_seconds=max_seconds,
        max_trials=max_trials,
        resume=not bool(getattr(args, "no_resume", False)),
    )
    if getattr(args, "json", False):
        _print_json(summary)
    else:
        print(_summary_text(summary))
    return 0 if summary.get("status") in ("completed", "stopped", "time_budget", "trial_budget", "paused", "exhausted", "no_candidate", "challenge_failed") else 1


def _cmd_status(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.all:
        runs_dir = config.path("runs_dir")
        rows = []
        for path in sorted(runs_dir.glob("*/ledger.sqlite3"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                item = run_status(config, path.parent.name)
                rows.append(
                    {
                        "run_id": item.get("run_id"),
                        "status": item.get("status"),
                        "trial_counts": item.get("trial_counts"),
                        "run_dir": item.get("run_dir"),
                    }
                )
            except Exception as exc:
                rows.append({"run_id": path.parent.name, "status": "unreadable", "error": str(exc)})
        if args.json:
            _print_json(rows)
        elif not rows:
            print(f"No runs found under {runs_dir}")
        else:
            for row in rows:
                print(f"{row['run_id']:<40} {row['status']}")
        return 0
    status = run_status(config, args.run_id)
    if args.json:
        _print_json(status)
    else:
        print(_status_text(status))
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    config = _load(args)
    result = stop_run(config, args.run_id, global_stop=args.global_stop)
    print(f"stop requested for {result['run_id']}: {result['stop_file']}")
    return 0


def _cmd_challenge(args: argparse.Namespace) -> int:
    config = _load(args)
    engine = Engine(config)
    summary = engine.run(
        mode="challenge",
        run_id=args.run_id,
        candidate_hash=args.candidate,
    )
    if args.json:
        _print_json(summary)
    else:
        print(_summary_text(summary))
        for opening in summary.get("challenge_openings", []):
            print(
                f"challenge opening {opening.get('opening_index')}: "
                f"candidate={opening.get('candidate_hash', '')[:12]} "
                f"cohort={opening.get('cohort')}"
            )
    return 0 if summary.get("status") == "completed" else 1


def _cmd_report(args: argparse.Namespace) -> int:
    config = _load(args)
    path = generate_report(config, args.run_id)
    print(path)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    config = _load(args)
    run_id = args.run_id or latest_run_id(config)
    if not run_id:
        raise ConfigError("no runs exist")
    run_dir = config.run_dir(run_id)
    ledger_path = run_dir / "ledger.sqlite3"
    if not ledger_path.exists():
        raise ConfigError(f"run not found: {run_id}")
    with Ledger(ledger_path) as ledger:
        candidate_hash = args.candidate
        if not candidate_hash:
            best = ledger.best_dev_candidate(run_id)
            if best:
                candidate_hash = best["candidate_hash"]
        if not candidate_hash:
            raise ConfigError("no candidate hash supplied and no confirmed candidate exists")
        candidates = ledger.find_candidates(run_id, candidate_hash, limit=args.limit)
        if not candidates:
            raise ConfigError(f"no candidate matches {candidate_hash!r}")
        payload = []
        for candidate in candidates:
            trials = ledger.trials_for_candidate(run_id, candidate["candidate_hash"])
            payload.append({"candidate": candidate, "trials": trials})
    if args.json:
        _print_json(payload)
    else:
        for item in payload:
            candidate = item["candidate"]
            print(f"Candidate: {candidate['candidate_hash']}")
            print(f"Level: {candidate['level']} generation={candidate['generation']} operator={candidate['operator']}")
            print(json.dumps(candidate["genome"], indent=2, sort_keys=True))
            for trial in item["trials"]:
                aggregate = trial.get("aggregate") or {}
                print(
                    f"  trial {trial['stage']:<8} {trial['status']:<10} "
                    f"mean={aggregate.get('mean')}"
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "smoke":
            return _cmd_run(args, "smoke")
        if args.command == "once":
            return _cmd_run(args, "once")
        if args.command == "continuous":
            return _cmd_run(args, "continuous")
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "stop":
            return _cmd_stop(args)
        if args.command == "challenge":
            return _cmd_challenge(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        raise ConfigError(f"unknown command: {args.command}")
    except AMDEBaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
