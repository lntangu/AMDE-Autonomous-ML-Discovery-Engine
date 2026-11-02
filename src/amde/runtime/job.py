"""Windows Job Object and verified process-tree containment for hard deadline enforcement."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .lock import default_pid_alive

# Win32 Constants
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_ALL_ACCESS = 0x1F001F
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


if os.name == "nt":
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_PROCESS_ID_LIST(ctypes.Structure):
        _fields_ = [
            ("NumberOfAssignedProcesses", wintypes.DWORD),
            ("NumberOfProcessIdsInList", wintypes.DWORD),
            ("ProcessIdList", ctypes.c_size_t * 256),
        ]


class WindowsJobObject:
    """Manages an OS-level Windows Job Object to contain and atomically terminate process trees."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.handle: Any = None
        self._is_nt = os.name == "nt"

    def is_active(self) -> bool:
        return self.handle is not None and self.handle != 0

    def create(self, kill_on_close: bool = False) -> bool:
        if not self._is_nt:
            return False
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_job = kernel32.CreateJobObjectW
            create_job.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
            create_job.restype = wintypes.HANDLE
            self.handle = create_job(None, self.name)
            if not self.handle:
                self.handle = None
                return False

            if kill_on_close:
                info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                set_info = kernel32.SetInformationJobObject
                set_info.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
                set_info.restype = wintypes.BOOL
                set_info(
                    self.handle,
                    JobObjectExtendedLimitInformation,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                )
            return True
        except Exception:
            self.handle = None
            return False

    def assign_process(self, pid_or_handle: int) -> bool:
        if not self.is_active():
            return False
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            assign_job = kernel32.AssignProcessToJobObject
            assign_job.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            assign_job.restype = wintypes.BOOL

            # If it looks like a PID, open the process handle
            proc_handle = pid_or_handle
            opened = False
            if pid_or_handle > 0 and pid_or_handle < 0x10000000:
                open_proc = kernel32.OpenProcess
                open_proc.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
                open_proc.restype = wintypes.HANDLE
                proc_handle = open_proc(PROCESS_ALL_ACCESS, False, pid_or_handle)
                if not proc_handle:
                    # Retry with query and terminate rights
                    proc_handle = open_proc(
                        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, False, pid_or_handle
                    )
                if proc_handle:
                    opened = True

            if not proc_handle:
                return False

            try:
                res = bool(assign_job(self.handle, proc_handle))
                return res
            finally:
                if opened and proc_handle:
                    close_h = kernel32.CloseHandle
                    close_h.argtypes = (wintypes.HANDLE,)
                    close_h.restype = wintypes.BOOL
                    close_h(proc_handle)
        except Exception:
            return False

    def get_process_ids(self) -> list[int]:
        if not self.is_active():
            return []
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            query_job = kernel32.QueryInformationJobObject
            query_job.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            )
            query_job.restype = wintypes.BOOL

            pid_list = JOBOBJECT_BASIC_PROCESS_ID_LIST()
            ret_len = wintypes.DWORD(0)
            success = query_job(
                self.handle,
                JobObjectBasicProcessIdList,
                ctypes.byref(pid_list),
                ctypes.sizeof(pid_list),
                ctypes.byref(ret_len),
            )
            if success:
                count = min(pid_list.NumberOfProcessIdsInList, 256)
                return [int(pid_list.ProcessIdList[i]) for i in range(count)]
            return []
        except Exception:
            return []

    def terminate(self, exit_code: int = 1) -> bool:
        if not self.is_active():
            return False
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            term_job = kernel32.TerminateJobObject
            term_job.argtypes = (wintypes.HANDLE, wintypes.UINT)
            term_job.restype = wintypes.BOOL
            return bool(term_job(self.handle, exit_code))
        except Exception:
            return False

    def close(self) -> None:
        if self.is_active():
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                close_h = kernel32.CloseHandle
                close_h.argtypes = (wintypes.HANDLE,)
                close_h.restype = wintypes.BOOL
                close_h(self.handle)
            except Exception:
                pass
            finally:
                self.handle = None

    def __enter__(self) -> "WindowsJobObject":
        if not self.is_active():
            self.create()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def validate_process_identity(
    pid: int | None,
    expected_tokens: list[str] | None = None,
    min_create_time: float | None = None,
) -> tuple[bool, str]:
    """Validate process identity to prevent PID recycling issues before forced termination."""
    if pid is None or not isinstance(pid, int) or pid <= 0:
        return False, "invalid_pid"
    if not default_pid_alive(pid):
        return False, "process_not_running"

    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        if min_create_time is not None:
            # Process must have been created at or after min_create_time (with 1.0s clock tolerance)
            try:
                p_create = proc.create_time()
                if p_create < (min_create_time - 1.0):
                    return False, f"pid_reused: create_time {p_create} < {min_create_time}"
            except Exception:
                pass

        if expected_tokens:
            try:
                cmdline = " ".join(proc.cmdline() or []).lower()
                name = (proc.name() or "").lower()
                matched = any(tok.lower() in cmdline or tok.lower() in name for tok in expected_tokens)
                if not matched:
                    return False, f"pid_identity_mismatch: {name} {cmdline[:60]}"
            except Exception:
                # Access denied on system process means definitely not our worker
                return False, "access_denied_system_process"

        return True, "verified"
    except ImportError:
        return True, "psutil_unavailable_fallback"
    except Exception as exc:
        return False, f"inspection_error: {type(exc).__name__}: {exc}"


def terminate_process_tree(
    pid: int | None,
    exit_code: int = 1,
    job: WindowsJobObject | None = None,
    expected_tokens: list[str] | None = None,
    min_create_time: float | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Atomically terminates a process and all its descendants with verification and event metrics."""
    start_t = time.monotonic()
    result: dict[str, Any] = {
        "success": False,
        "target_pid": pid,
        "job_terminated": False,
        "descendants_identified": [],
        "pids_terminated": [],
        "mechanism": "none",
        "elapsed_seconds": 0.0,
        "errors": [],
    }

    # 1. Discover all descendants via psutil while processes are active
    all_pids: list[int] = [pid] if (pid is not None and pid > 0) else []
    if pid is not None and pid > 0:
        try:
            import psutil  # type: ignore

            if psutil.pid_exists(pid):
                root_proc = psutil.Process(pid)
                children = root_proc.children(recursive=True)
                for c in children:
                    if c.pid not in all_pids:
                        all_pids.append(c.pid)
        except Exception:
            pass

    # 2. Add any Job Object PIDs
    if job and job.is_active():
        job_pids = job.get_process_ids()
        for jp in job_pids:
            if jp not in all_pids:
                all_pids.append(jp)

    result["descendants_identified"] = list(all_pids)

    # 3. Terminate Job Object if present (fastest, atomic OS kernel termination)
    if job and job.is_active():
        term_ok = job.terminate(exit_code)
        result["job_terminated"] = term_ok
        if term_ok:
            result["mechanism"] = "job_object"

    # 4. Validate target process identity before direct OS-level taskkill
    if pid is not None and pid > 0:
        is_valid, reason = validate_process_identity(
            pid, expected_tokens=expected_tokens, min_create_time=min_create_time
        )
        if not is_valid:
            if reason == "process_not_running" or result["job_terminated"]:
                # Already exited or terminated via Job Object
                if pid in all_pids and not default_pid_alive(pid):
                    pass
            else:
                result["errors"].append(f"Identity validation failed: {reason}")
                if not result["job_terminated"]:
                    result["elapsed_seconds"] = time.monotonic() - start_t
                    return result
                # Do not run direct taskkill on this mismatched PID
                if pid in all_pids:
                    all_pids.remove(pid)

    if not all_pids:
        result["elapsed_seconds"] = time.monotonic() - start_t
        result["success"] = True
        return result

    # 5. OS-level termination fallback for all identified PIDs (tree-kill)
    if os.name == "nt":
        for p in all_pids:
            if default_pid_alive(p):
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(p), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result["mechanism"] == "none":
                        result["mechanism"] = "taskkill"
                except Exception:
                    pass
    else:
        try:
            import psutil  # type: ignore

            for p in reversed(all_pids):
                if default_pid_alive(p):
                    try:
                        psutil.Process(p).kill()
                    except Exception:
                        pass
            if result["mechanism"] == "none":
                result["mechanism"] = "psutil_kill"
        except Exception as exc:
            result["errors"].append(f"posix kill error: {exc}")

    # 6. Verification loop: wait up to timeout seconds for all PIDs to physically exit
    deadline = time.monotonic() + max(0.5, timeout)
    remaining = set(all_pids)
    while remaining and time.monotonic() < deadline:
        still_alive = set()
        for p in remaining:
            if default_pid_alive(p):
                still_alive.add(p)
            else:
                if p not in result["pids_terminated"]:
                    result["pids_terminated"].append(p)
        remaining = still_alive
        if remaining:
            time.sleep(0.05)

    result["success"] = len(remaining) == 0
    if remaining:
        result["errors"].append(f"Processes failed to exit: {list(remaining)}")
    result["elapsed_seconds"] = time.monotonic() - start_t
    return result
