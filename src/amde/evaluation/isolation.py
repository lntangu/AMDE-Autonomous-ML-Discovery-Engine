"""Parent-side process isolation for candidate evaluation.

Every evaluation is executed in a fresh spawned process. The parent enforces a
wall-clock timeout, an optional per-process RSS cap (when psutil is available),
and cooperative stop requests. Candidate failures and infrastructure failures
are classified separately by the worker and by the parent.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from .worker import child_entry

_POLL_SECONDS = 0.05
_MEMORY_SAMPLE_SECONDS = 0.15
_SUCCESSFUL_WORKER_EXIT_GRACE_SECONDS = 5.0


@dataclass
class WorkerJob:
    payload: dict[str, Any]
    timeout_seconds: float
    candidate: bool = True
    label: str = ""


@dataclass
class _Running:
    index: int
    job: WorkerJob
    process: Any
    connection: Any
    started: float
    deadline: float
    peak_rss_mb: float = 0.0
    psutil_process: Any = None
    next_memory_sample: float = 0.0


def _terminate(running: _Running) -> None:
    try:
        if running.process.is_alive():
            running.process.terminate()
            running.process.join(1.0)
    except Exception:
        pass
    try:
        if running.process.is_alive():
            running.process.kill()
            running.process.join(1.0)
    except Exception:
        pass
    try:
        running.connection.close()
    except Exception:
        pass


def _launch(index: int, job: WorkerJob, context: Any) -> _Running:
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=child_entry,
        args=(job.payload, child_conn),
        name=f"amde-eval-{job.label or index}",
        daemon=True,
    )
    try:
        process.start()
    except BaseException:
        # multiprocessing can fail before a worker exists (for example, a
        # Windows spawn/pickle error). Close both pipe endpoints in that path.
        for endpoint in (parent_conn, child_conn):
            try:
                endpoint.close()
            except Exception:
                pass
        try:
            process.close()
        except Exception:
            pass
        raise
    try:
        child_conn.close()
    except Exception:
        pass
    now = time.monotonic()
    return _Running(
        index=index,
        job=job,
        process=process,
        connection=parent_conn,
        started=now,
        deadline=now + max(0.1, float(job.timeout_seconds)),
    )


def _base_result(running: _Running, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "result": None,
        "error_type": None,
        "error_code": None,
        "message": "",
        "elapsed_seconds": max(0.0, time.monotonic() - running.started),
        "peak_rss_mb": running.peak_rss_mb or None,
        "exit_code": running.process.exitcode,
        "worker_pid": running.process.pid,
        "memory_cap_enforced": False,
        "worker_shutdown_forced": False,
    }


def _finish(
    running: _Running, status: str, *, worker_returned: bool = False, **updates: Any
) -> dict[str, Any]:
    result = _base_result(running, status)
    result.update(updates)
    forced_shutdown = False
    if worker_returned:
        # A worker sends its structured result before Python finishes process
        # shutdown. Give it a bounded grace period to exit naturally; Windows
        # process cleanup and native thread-pool teardown can exceed 0.3s. This
        # applies to valid results and structured worker errors alike.
        try:
            running.process.join(_SUCCESSFUL_WORKER_EXIT_GRACE_SECONDS)
        except Exception:
            pass
        if running.process.is_alive():
            forced_shutdown = True
            _terminate(running)
        else:
            try:
                running.connection.close()
            except Exception:
                pass
    else:
        # Parent-side timeout, stop, and resource-cap paths must remain bounded
        # and may terminate a worker that has not returned a message.
        _terminate(running)
    exit_code = running.process.exitcode
    result["exit_code"] = exit_code
    result["worker_shutdown_forced"] = forced_shutdown
    if forced_shutdown:
        note = (
            f"worker returned status {status!r} but did not exit within "
            f"{_SUCCESSFUL_WORKER_EXIT_GRACE_SECONDS:.1f}s; process was force-terminated"
        )
        existing = str(result.get("message") or "").strip()
        result["message"] = f"{existing}; {note}" if existing else note
    try:
        running.process.close()
    except Exception:
        pass
    return result


def run_jobs(
    jobs: list[WorkerJob],
    *,
    max_workers: int = 1,
    memory_cap_mb: int | None = None,
    stop_callback: Callable[[], bool] | None = None,
    poll_seconds: float = _POLL_SECONDS,
) -> list[dict[str, Any]]:
    """Run isolated jobs with bounded concurrency.

    Results preserve input order. A process is always terminated after its
    single result, so candidate code cannot survive between trials.
    """

    if not jobs:
        return []
    workers = max(1, min(int(max_workers), len(jobs)))
    context = multiprocessing.get_context("spawn")
    cap = int(memory_cap_mb or 0)
    psutil_module = None
    if cap > 0:
        try:
            import psutil  # type: ignore

            psutil_module = psutil
        except Exception:
            psutil_module = None
    memory_enforced = bool(cap > 0 and psutil_module is not None)
    # We still monitor RSS when psutil exists even without a hard cap, for the
    # ledger. It is explicitly reported as not enforced otherwise.

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending: list[_Running] = []
    next_index = 0
    stop_seen = False

    def note_stop() -> bool:
        nonlocal stop_seen
        if stop_callback is None or stop_seen:
            return stop_seen
        try:
            stop_seen = bool(stop_callback())
        except Exception:
            stop_seen = False
        return stop_seen

    try:
        while next_index < len(jobs) or pending:
            if not stop_seen and not note_stop():
                while len(pending) < workers and next_index < len(jobs):
                    running = _launch(next_index, jobs[next_index], context)
                    pending.append(running)
                    next_index += 1

            if stop_seen:
                for running in list(pending):
                    results[running.index] = _finish(
                        running,
                        "interrupted",
                        error_type="infrastructure",
                        error_code="stop_requested",
                        message="stop was requested while the evaluation was running",
                        memory_cap_enforced=memory_enforced,
                    )
                    pending.remove(running)
                for index in range(next_index, len(jobs)):
                    results[index] = {
                        "status": "interrupted",
                        "result": None,
                        "error_type": "infrastructure",
                        "error_code": "stop_requested",
                        "message": "job was not started because stop was requested",
                        "elapsed_seconds": 0.0,
                        "peak_rss_mb": None,
                        "exit_code": None,
                        "worker_pid": None,
                        "memory_cap_enforced": memory_enforced,
                    }
                next_index = len(jobs)
                continue

            now = time.monotonic()
            for running in list(pending):
                if running.psutil_process is None and psutil_module is not None:
                    try:
                        running.psutil_process = psutil_module.Process(running.process.pid)
                    except Exception:
                        running.psutil_process = None
                if running.psutil_process is not None and now >= running.next_memory_sample:
                    try:
                        rss_mb = running.psutil_process.memory_info().rss / (1024**2)
                        running.peak_rss_mb = max(running.peak_rss_mb, float(rss_mb))
                        if memory_enforced and rss_mb > cap:
                            results[running.index] = _finish(
                                running,
                                "memory",
                                error_type="candidate" if running.job.candidate else "infrastructure",
                                error_code="memory_cap",
                                message=f"worker RSS {rss_mb:.1f} MB exceeded cap {cap} MB",
                                memory_cap_enforced=True,
                            )
                            pending.remove(running)
                            continue
                    except Exception:
                        pass
                    running.next_memory_sample = now + _MEMORY_SAMPLE_SECONDS

                message: dict[str, Any] | None = None
                try:
                    if running.connection.poll(0):
                        message = running.connection.recv()
                except (EOFError, OSError):
                    message = None
                if message is not None:
                    status = str(message.get("status", "error"))
                    updates = {
                        "result": message.get("result"),
                        "error_type": message.get("error_type"),
                        "error_code": message.get("error_code"),
                        "message": message.get("message", ""),
                        "memory_cap_enforced": memory_enforced,
                    }
                    results[running.index] = _finish(running, status, worker_returned=True, **updates)
                    pending.remove(running)
                    continue

                if not running.process.is_alive():
                    # Give the pipe one final opportunity to deliver a result.
                    try:
                        if running.connection.poll(0.1):
                            message = running.connection.recv()
                    except (EOFError, OSError):
                        message = None
                    if message is not None:
                        status = str(message.get("status", "error"))
                        results[running.index] = _finish(
                            running,
                            status,
                            worker_returned=True,
                            result=message.get("result"),
                            error_type=message.get("error_type"),
                            error_code=message.get("error_code"),
                            message=message.get("message", ""),
                            memory_cap_enforced=memory_enforced,
                        )
                    else:
                        results[running.index] = _finish(
                            running,
                            "crashed",
                            error_type="infrastructure",
                            error_code="worker_exit",
                            message=f"worker exited with code {running.process.exitcode} without a result",
                            memory_cap_enforced=memory_enforced,
                        )
                    pending.remove(running)
                    continue

                if now > running.deadline:
                    results[running.index] = _finish(
                        running,
                        "timeout",
                        error_type="candidate" if running.job.candidate else "infrastructure",
                        error_code="timeout",
                        message=f"evaluation exceeded {running.job.timeout_seconds:.1f}s timeout",
                        memory_cap_enforced=memory_enforced,
                    )
                    pending.remove(running)
                    continue

            if pending:
                time.sleep(max(0.01, float(poll_seconds)))
    finally:
        for running in list(pending):
            _terminate(running)
            try:
                if not running.process.is_alive():
                    running.process.close()
            except Exception:
                pass

    for index, result in enumerate(results):
        if result is None:  # pragma: no cover - defensive
            results[index] = {
                "status": "crashed",
                "result": None,
                "error_type": "infrastructure",
                "error_code": "missing_worker_result",
                "message": f"no result was recorded for job {index}",
                "elapsed_seconds": 0.0,
                "peak_rss_mb": None,
                "exit_code": None,
                "worker_pid": None,
                "memory_cap_enforced": memory_enforced,
            }
    return [item for item in results if item is not None]
