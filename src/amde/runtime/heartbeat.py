"""Heartbeat and cooperative stop files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..util import atomic_write_json, ensure_dir, read_json, utc_now


class HeartbeatWriter:
    def __init__(self, run_dir: Path, *, run_id: str, interval_seconds: float = 2.0) -> None:
        self.path = Path(run_dir) / "heartbeat.json"
        self.run_id = run_id
        self.interval_seconds = float(interval_seconds)
        self._last = 0.0
        self._base: dict[str, Any] = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at": utc_now(),
        }

    def write(self, payload: dict[str, Any], *, force: bool = False) -> None:
        import time

        now = time.time()
        if not force and now - self._last < self.interval_seconds:
            return
        data = dict(self._base)
        data.update(payload)
        data["updated_at"] = utc_now()
        atomic_write_json(self.path, data)
        self._last = now

    def read(self) -> dict[str, Any] | None:
        try:
            data = read_json(self.path)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def age_seconds(self) -> float | None:
        try:
            return max(0.0, __import__("time").time() - self.path.stat().st_mtime)
        except OSError:
            return None


class StopController:
    def __init__(self, run_dir: Path, *, stop_name: str = "STOP", global_runs_dir: Path | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.stop_name = stop_name
        self.global_runs_dir = Path(global_runs_dir) if global_runs_dir else None

    @property
    def run_stop_file(self) -> Path:
        return self.run_dir / self.stop_name

    @property
    def global_stop_file(self) -> Path | None:
        return self.global_runs_dir / self.stop_name if self.global_runs_dir else None

    def requested(self) -> bool:
        if self.run_stop_file.exists():
            return True
        global_file = self.global_stop_file
        return bool(global_file and global_file.exists())

    def reason(self) -> str:
        for path in (self.run_stop_file, self.global_stop_file):
            if path and path.exists():
                try:
                    data = read_json(path)
                    if isinstance(data, dict):
                        return str(data.get("reason", "stop requested"))
                except Exception:
                    pass
                return "stop file present"
        return ""

    def request(self, reason: str = "user requested stop", *, global_stop: bool = False) -> Path:
        ensure_dir(self.run_dir)
        path = self.run_stop_file
        atomic_write_json(path, {"requested_at": utc_now(), "pid": os.getpid(), "reason": reason})
        if global_stop and self.global_runs_dir:
            ensure_dir(self.global_runs_dir)
            atomic_write_json(self.global_stop_file, {"requested_at": utc_now(), "pid": os.getpid(), "reason": reason})
        return path

    def clear(self, *, clear_global: bool = False, clear_run: bool = True) -> None:
        if clear_run:
            try:
                self.run_stop_file.unlink(missing_ok=True)
            except OSError:
                pass
        if clear_global and self.global_stop_file:
            try:
                self.global_stop_file.unlink(missing_ok=True)
            except OSError:
                pass
