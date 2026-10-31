"""Single-instance lock with PID/heartbeat-based stale recovery."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable

from ..errors import LockError
from ..util import atomic_write_json, ensure_dir, read_json, safe_fsync, utc_now


def default_pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid) and psutil.Process(pid).is_running())
    except ImportError:
        pass
    except Exception:
        # Access denied can mean the process exists; treat it as alive rather
        # than stealing a lock from a privileged but real process.
        return True
    if os.name == "nt":
        # A process handle is pointer-sized on 64-bit Windows. Declare the
        # Win32 signatures explicitly; ctypes' default c_int return type can
        # truncate HANDLE values. SYNCHRONIZE is sufficient for a zero-time
        # WaitForSingleObject liveness probe and does not signal the process.
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            open_process.restype = wintypes.HANDLE
            wait_for_single_object = kernel32.WaitForSingleObject
            wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            wait_for_single_object.restype = wintypes.DWORD
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL

            SYNCHRONIZE = 0x00100000
            WAIT_OBJECT_0 = 0x00000000
            WAIT_TIMEOUT = 0x00000102
            handle = open_process(SYNCHRONIZE, False, int(pid))
            if not handle:
                # Invalid PID means dead. Access denied or other uncertain
                # errors are treated as alive to avoid stealing a live lock.
                return ctypes.get_last_error() not in {87, 1168}
            try:
                result = int(wait_for_single_object(handle, 0))
                if result == WAIT_OBJECT_0:
                    return False
                if result == WAIT_TIMEOUT:
                    return True
                # WAIT_FAILED/unknown: conservatively assume the process may
                # still exist; callers can inspect the associated heartbeat.
                return True
            finally:
                close_handle(handle)
        except Exception:
            # A failed platform probe must not authorize stealing a lock or
            # launching a duplicate supervisor.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


class SingleInstanceLock:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        stale_seconds: float = 300.0,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.stale_seconds = float(stale_seconds)
        self.pid_alive = pid_alive or default_pid_alive
        self.acquired = False
        self.data: dict[str, Any] = {}

    def _read_existing(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            raw = read_json(self.path)
            if not isinstance(raw, dict):
                return None, "lock file is not an object"
            return raw, None
        except FileNotFoundError:
            return None, None
        except Exception as exc:
            return None, f"lock file is unreadable: {exc}"

    def _is_stale(self, existing: dict[str, Any] | None, read_error: str | None, file_age: float) -> tuple[bool, str]:
        if existing is None:
            # An empty/partially written lock is only stolen after the stale
            # interval. This avoids a live process being preempted mid-write.
            if read_error and file_age >= self.stale_seconds:
                return True, read_error
            return False, read_error or "lock exists but is not readable yet"
        pid = existing.get("pid")
        if not isinstance(pid, int):
            return (file_age >= self.stale_seconds), "lock has no valid pid"
        if not self.pid_alive(pid):
            return True, f"lock owner pid {pid} is not alive"
        heartbeat = existing.get("heartbeat_at")
        age = file_age
        if isinstance(heartbeat, str):
            try:
                from datetime import datetime, timezone

                parsed = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - parsed).total_seconds()
            except Exception:
                age = file_age
        if age > self.stale_seconds:
            return True, f"lock owner pid {pid} heartbeat is stale ({age:.0f}s)"
        return False, f"another AMDE instance is running (pid={pid}, run={existing.get('run_id')})"

    def acquire(self) -> dict[str, Any]:
        ensure_dir(self.path.parent)
        for attempt in range(3):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                now = utc_now()
                payload = {
                    "schema_version": "1.0.0",
                    "pid": os.getpid(),
                    "run_id": self.run_id,
                    "host": socket.gethostname(),
                    "started_at": now,
                    "heartbeat_at": now,
                }
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    safe_fsync(handle.fileno())
                self.data = payload
                self.acquired = True
                return payload
            except FileExistsError:
                existing, read_error = self._read_existing()
                try:
                    file_age = time.time() - self.path.stat().st_mtime
                except OSError:
                    file_age = 0.0
                stale, reason = self._is_stale(existing, read_error, file_age)
                if stale:
                    archived = self.path.with_name(
                        f"{self.path.name}.stale.{int(time.time())}.{os.getpid()}"
                    )
                    try:
                        os.replace(self.path, archived)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise LockError(f"stale lock could not be archived: {exc}") from exc
                    continue
                raise LockError(reason)
        raise LockError("could not acquire lock after stale-recovery retries")

    def heartbeat(self) -> None:
        if not self.acquired:
            return
        self.data["heartbeat_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = read_json(self.path)
            if (
                isinstance(existing, dict)
                and existing.get("pid") == os.getpid()
                and existing.get("run_id") == self.run_id
            ):
                self.path.unlink(missing_ok=True)
        except Exception:
            pass
        finally:
            self.acquired = False

    def read(self) -> dict[str, Any] | None:
        try:
            value = read_json(self.path)
            return value if isinstance(value, dict) else None
        except Exception:
            return None
