# AMDE 1.2.1 — Windows detached-supervisor hotfix

AMDE 1.2.1 fixes a destructive Windows process-liveness check in the detached hybrid controller.

## Root cause

The 1.2.0 `scripts/hybrid_control.py` implementation used `os.kill(pid, 0)` to check whether a process existed. On POSIX this is a conventional non-destructive existence check. On Windows, Python maps non-console-event signal values to the `TerminateProcess` API; signal value zero is therefore not a harmless existence probe. Calling `status` could terminate the supervisor. A subsequent `stop` then observed a dead supervisor and refused to proceed, while its child workers could remain alive and hold log files open.

## Fix

The controller now delegates to `amde.runtime.lock.default_pid_alive`, which uses psutil when available, a Windows process handle when running on Windows, and `os.kill(pid, 0)` only on POSIX. A unit regression test checks that probing a live subprocess leaves it alive. The detached hybrid integration test covers `start -> status -> stop`.

## Qualification status

The source-level fix and regression suite are tested in the build environment. This release must still pass native Windows integration test 10 on the user's target machine before detached hybrid campaigns are considered qualified. No claim is made about 24-hour reliability, quantum advantage, or generalization from this hotfix.
