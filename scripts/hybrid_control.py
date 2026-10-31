#!/usr/bin/env python3
"""Start, inspect, stop and tail a detached AMDE hybrid supervisor.

The detached controller lets the caller return to the same PowerShell prompt;
stop is then requested with a marker and the supervisor performs cooperative
worker shutdown. It avoids terminating a .bat job with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amde.runtime.lock import default_pid_alive  # noqa: E402
from amde.util import atomic_write_json, atomic_write_text, read_json, sanitize_id, utc_now  # noqa: E402


def _paths(root: Path, campaign_id: str) -> dict[str, Path]:
    base = root / "reports" / "hybrid_campaigns" / campaign_id
    return {
        "base": base,
        "state": base / "supervisor_state.json",
        "launch": base / "launch.json",
        "console": base / "logs" / "supervisor_console.log",
        "stop": root / "runs" / "HYBRID_STOP",
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        result = read_json(path)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _pid_alive(pid: Any) -> bool:
    """Check process existence without sending a signal.

    ``os.kill(pid, 0)`` is a POSIX liveness probe, but on Windows Python maps
    ordinary signal values (including zero) to ``TerminateProcess``.  Reuse
    AMDE's platform-aware process probe instead: psutil when available, a
    Windows process handle on Windows, and signal-zero only on POSIX.
    """
    try:
        value = int(pid)
    except (TypeError, ValueError, OverflowError):
        return False
    return default_pid_alive(value)


def _tail(path: Path, count: int) -> str:
    if not path.is_file():
        return "<log not created>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max(1, count):])
    except OSError as exc:
        return f"<cannot read log: {exc}>"


def _start(root: Path, campaign_id: str, forwarded: list[str]) -> int:
    paths = _paths(root, campaign_id)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["console"].parent.mkdir(parents=True, exist_ok=True)
    state = _read_json(paths["state"])
    if state and _pid_alive(state.get("supervisor_pid")):
        print(f"Supervisor already running: pid={state['supervisor_pid']}")
        return 1
    launch = _read_json(paths["launch"])
    if launch and _pid_alive(launch.get("pid")):
        print(f"A previously launched supervisor is still running: pid={launch['pid']}")
        return 1

    supervisor = root / "scripts" / "hybrid_research_campaign.py"
    if not supervisor.is_file():
        raise FileNotFoundError(f"hybrid supervisor not found: {supervisor}")
    command = [sys.executable, "-u", str(supervisor), "--root", str(root), "--campaign-id", campaign_id]
    command.extend(forwarded)
    if any(item in {"--root", "--campaign-id"} for item in forwarded):
        raise ValueError("pass --root and --campaign-id to hybrid_control itself, not after --")

    with paths["console"].open("ab", buffering=0) as log_handle:
        log_handle.write(f"\n[{utc_now()}] detached launch: {command}\n".encode("utf-8", errors="replace"))
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            process = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=flags,
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
    atomic_write_json(paths["launch"], {
        "schema_version": "amde-hybrid-launch-v1",
        "campaign_id": campaign_id,
        "pid": process.pid,
        "command": command,
        "started_at": utc_now(),
        "console_log": str(paths["console"]),
    })
    time.sleep(0.5)
    code = process.poll()
    if code is not None:
        print(f"Supervisor exited during startup (code={code}). Log tail:")
        print(_tail(paths["console"], 60))
        return 2
    print(f"Started detached AMDE hybrid supervisor: pid={process.pid}")
    print(f"State: {paths['state']}")
    print(f"Console log: {paths['console']}")
    print("Use 'status' to inspect and 'stop' to request a cooperative shutdown.")
    return 0


def _status(root: Path, campaign_id: str) -> int:
    paths = _paths(root, campaign_id)
    state = _read_json(paths["state"])
    launch = _read_json(paths["launch"])
    if not state:
        print("Supervisor state not found. Has this campaign been started?")
        if launch:
            print(f"Last launch PID={launch.get('pid')} alive={_pid_alive(launch.get('pid'))}")
        return 2
    pid = state.get("supervisor_pid")
    print(f"Campaign: {state.get('campaign_id')}")
    print(f"Status: {state.get('status')}")
    print(f"Supervisor PID: {pid} (alive={_pid_alive(pid)})")
    print(f"Heartbeat: {state.get('heartbeat_at')}")
    print(f"Note: {state.get('note') or '-'}")
    for name, worker in (state.get("children") or {}).items():
        worker = worker or {}
        print(
            f"{name}: status={worker.get('status')} pid={worker.get('pid')} "
            f"exit={worker.get('last_exit_code')} restarts={worker.get('restarts')}"
        )
    return 0


def _stop(root: Path, campaign_id: str, wait_seconds: int) -> int:
    paths = _paths(root, campaign_id)
    state = _read_json(paths["state"])
    if not state:
        print("No supervisor state found; refusing to guess which workers should be stopped.")
        return 2
    pid = state.get("supervisor_pid")
    if not _pid_alive(pid):
        print(f"Supervisor PID {pid} is not running. Inspect state and worker locks before any resume.")
        return 2
    paths["stop"].parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["stop"], f"cooperative stop requested by hybrid_control at {utc_now()}\n")
    print(f"Stop requested for {campaign_id}; waiting up to {wait_seconds}s for checkpointed shutdown...")
    terminal = {"stopped", "resource_paused", "circuit_breaker", "supervisor_error"}
    deadline = time.monotonic() + max(1, wait_seconds)
    while time.monotonic() < deadline:
        current = _read_json(paths["state"]) or {}
        children = current.get("children") or {}
        children_clear = all(not _pid_alive((child or {}).get("pid")) for child in children.values())
        supervisor_clear = not _pid_alive(current.get("supervisor_pid"))
        if current.get("status") in terminal and children_clear and supervisor_clear:
            print(f"Shutdown complete: status={current.get('status')}")
            return 0 if current.get("status") == "stopped" else 1
        time.sleep(0.5)
    current = _read_json(paths["state"]) or {}
    print(f"Timed out; supervisor status={current.get('status')}. Do NOT kill it; inspect logs and locks.")
    print(_tail(paths["console"], 40))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop", "logs"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--wait-seconds", type=int, default=240)
    parser.add_argument("--tail", type=int, default=60)
    args, forwarded = parser.parse_known_args(argv)
    root = Path(args.root).resolve()
    campaign_id = sanitize_id(args.campaign_id, fallback="amde_hybrid")
    if len(campaign_id) > 32:
        parser.error("campaign id must be 32 characters or fewer")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    try:
        if args.action == "start":
            return _start(root, campaign_id, forwarded)
        if forwarded:
            parser.error(f"unexpected extra arguments for {args.action}: {forwarded}")
        if args.action == "status":
            return _status(root, campaign_id)
        if args.action == "stop":
            return _stop(root, campaign_id, args.wait_seconds)
        paths = _paths(root, campaign_id)
        print("=== SUPERVISOR ===")
        print(_tail(paths["console"], args.tail))
        print("=== CLASSICAL WORKER ===")
        print(_tail(paths["base"] / "logs" / "classical_worker.log", args.tail))
        print("=== QUANTUM WORKER ===")
        print(_tail(paths["base"] / "logs" / "quantum_worker.log", args.tail))
        return 0
    except (OSError, ValueError) as exc:
        print(f"hybrid control failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
