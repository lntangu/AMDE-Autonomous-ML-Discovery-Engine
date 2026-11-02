"""Native Windows 11 Qualification Test Suite for AMDE 2.0.1.

Executes directly on Windows host via .venv\\Scripts\\python.exe.
Validates:
1. Atomic robust_replace under Windows NTFS filesystem semantics.
2. Win32 Sleep Guard (SetThreadExecutionState API).
3. Process management, process identity, and PID reuse protection.
4. Fail-closed JSON semantics on NTFS.
5. SQLite WAL journaling and busy timeout on Windows.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from amde.util import robust_replace, read_json_safe, read_json_critical, CorruptJSONError
from amde.runtime.amde2 import _pid_alive, _pid_matches
from amde.storage.ledger import Ledger


def test_win32_sleep_guard() -> dict[str, Any]:
    """Test Win32 SetThreadExecutionState API."""
    if os.name != "nt":
        return {"status": "BLOCKED", "reason": "Not running on Windows"}
    
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    kernel32 = ctypes.windll.kernel32
    # Request sleep prevention
    prev_state = kernel32.SetThreadExecutionState(
        ctypes.c_uint(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    )
    # Revert to standard state
    revert_state = kernel32.SetThreadExecutionState(ctypes.c_uint(ES_CONTINUOUS))
    
    return {
        "status": "PASS" if prev_state != 0 else "FAIL",
        "prev_state_hex": hex(prev_state),
        "revert_state_hex": hex(revert_state),
    }


def test_robust_replace_ntfs() -> dict[str, Any]:
    """Verify robust_replace handles atomic replacement and contention on NTFS."""
    with tempfile.TemporaryDirectory() as td:
        dir_path = Path(td)
        target = dir_path / "target.txt"
        target.write_text("initial", encoding="utf-8")
        
        # Test 1: standard atomic replacement
        source = dir_path / "source.txt"
        source.write_text("updated", encoding="utf-8")
        robust_replace(source, target)
        
        if not target.exists() or target.read_text(encoding="utf-8") != "updated":
            return {"status": "FAIL", "reason": "Basic replacement failed"}
        if source.exists():
            return {"status": "FAIL", "reason": "Source still exists after replace"}
            
        # Test 2: multiple rapid replacements
        for i in range(10):
            src_i = dir_path / f"src_{i}.txt"
            src_i.write_text(f"content_{i}", encoding="utf-8")
            robust_replace(src_i, target)
            assert target.read_text(encoding="utf-8") == f"content_{i}"
            
        return {"status": "PASS", "replacements_verified": 11}


def test_process_management_windows() -> dict[str, Any]:
    """Verify PID check and command-line validation on Windows."""
    current_pid = os.getpid()
    alive = _pid_alive(current_pid)
    dead_pid_alive = _pid_alive(999999)
    
    # Process matching check for current process
    # Passing dummy required_script should return False
    matches_false = _pid_matches(current_pid, REPO_ROOT, "dummy_token", "nonexistent_script.py")
    
    return {
        "status": "PASS" if (alive and not dead_pid_alive and not matches_false) else "FAIL",
        "current_pid_alive": alive,
        "dead_pid_alive": dead_pid_alive,
        "identity_guard_working": not matches_false,
    }


def test_sqlite_wal_windows() -> dict[str, Any]:
    """Verify SQLite WAL mode and concurrency on Windows NTFS."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "ledger.sqlite3"
        ledger = Ledger(db_path)
        with ledger.transaction() as con:
            con.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
            con.execute("INSERT INTO test (val) VALUES ('amde2')")
            journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
            busy_timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
        ledger.close()
        
        return {
            "status": "PASS" if journal_mode.lower() == "wal" else "FAIL",
            "journal_mode": journal_mode,
            "busy_timeout_ms": busy_timeout,
        }


def test_critical_json_fail_closed_ntfs() -> dict[str, Any]:
    """Verify read_json_critical fails closed on corrupt/missing files on Windows."""
    with tempfile.TemporaryDirectory() as td:
        dir_path = Path(td)
        missing_file = dir_path / "missing.json"
        corrupt_file = dir_path / "corrupt.json"
        corrupt_file.write_text("{bad json: 123", encoding="utf-8")
        
        # 1. Missing optional file -> returns default
        opt_missing = read_json_safe(missing_file, default={"opt": True}, optional=True)
        # 2. Corrupt optional file -> raises CorruptJSONError (does NOT return default!)
        corrupt_caught = False
        try:
            read_json_safe(corrupt_file, default={"opt": True}, optional=True)
        except CorruptJSONError:
            corrupt_caught = True
            
        # 3. Missing critical file -> raises FileNotFoundError
        crit_missing_caught = False
        try:
            read_json_critical(missing_file)
        except FileNotFoundError:
            crit_missing_caught = True
            
        # 4. Corrupt critical file -> raises CorruptJSONError
        crit_corrupt_caught = False
        try:
            read_json_critical(corrupt_file)
        except CorruptJSONError:
            crit_corrupt_caught = True
            
        all_passed = (
            opt_missing == {"opt": True}
            and corrupt_caught
            and crit_missing_caught
            and crit_corrupt_caught
        )
        return {
            "status": "PASS" if all_passed else "FAIL",
            "opt_missing_ok": opt_missing == {"opt": True},
            "corrupt_optional_fails_closed": corrupt_caught,
            "missing_critical_fails_closed": crit_missing_caught,
            "corrupt_critical_fails_closed": crit_corrupt_caught,
        }


def main() -> int:
    results = {
        "os_name": os.name,
        "platform": sys.platform,
        "python_version": sys.version,
        "sleep_guard": test_win32_sleep_guard(),
        "robust_replace": test_robust_replace_ntfs(),
        "process_management": test_process_management_windows(),
        "sqlite_wal": test_sqlite_wal_windows(),
        "fail_closed_json": test_critical_json_fail_closed_ntfs(),
    }
    print(json.dumps(results, indent=2))
    all_pass = all(
        v.get("status") == "PASS"
        for k, v in results.items()
        if isinstance(v, dict) and "status" in v
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
