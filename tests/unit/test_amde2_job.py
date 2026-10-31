"""Unit tests for Windows Job Object and process-tree containment."""

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amde.runtime.job import (
    WindowsJobObject,
    terminate_process_tree,
    validate_process_identity,
)
from amde.runtime.lock import default_pid_alive


class WindowsJobObjectTests(unittest.TestCase):
    def test_job_object_lifecycle(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows Job Objects are native to Windows NT")

        job = WindowsJobObject(name=f"Local\\AMDE_Test_Lifecycle_{int(time.time()*1000)}")
        created = job.create(kill_on_close=False)
        self.assertTrue(created)
        self.assertTrue(job.is_active())

        # Spawn a sleeping child
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(default_pid_alive(proc.pid))
            assigned = job.assign_process(proc.pid)
            self.assertTrue(assigned)

            # Query process list
            pids = job.get_process_ids()
            self.assertIn(proc.pid, pids)

            # Terminate job
            term_ok = job.terminate(exit_code=42)
            self.assertTrue(term_ok)

            time.sleep(0.3)
            self.assertFalse(default_pid_alive(proc.pid))
            self.assertEqual(proc.poll(), 42)
        finally:
            if proc.poll() is None:
                proc.kill()
            job.close()
            self.assertFalse(job.is_active())

    def test_job_object_grandchild_containment(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows Job Objects are native to Windows NT")

        job = WindowsJobObject(name=f"Local\\AMDE_Test_Grandchild_{int(time.time()*1000)}")
        self.assertTrue(job.create(kill_on_close=False))

        # Parent spawns child, child spawns grandchild
        script = """
import subprocess, sys, time
p1 = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; p2 = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"])
time.sleep(60)
"""
        parent = subprocess.Popen([sys.executable, "-c", script])
        try:
            self.assertTrue(job.assign_process(parent.pid))
            time.sleep(1.0)  # wait for children to spawn

            import psutil  # type: ignore

            parent_proc = psutil.Process(parent.pid)
            descendants = parent_proc.children(recursive=True)
            self.assertGreaterEqual(len(descendants), 2)
            descendant_pids = [d.pid for d in descendants]

            # Terminate entire job
            self.assertTrue(job.terminate(exit_code=99))
            time.sleep(0.5)

            # Assert all descendants are dead
            self.assertFalse(default_pid_alive(parent.pid))
            for d_pid in descendant_pids:
                self.assertFalse(default_pid_alive(d_pid))
        finally:
            if parent.poll() is None:
                parent.kill()
            job.close()

    def test_process_identity_validation_prevents_collateral_termination(self) -> None:
        # Invalid PID
        ok, reason = validate_process_identity(-1)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_pid")

        # Dead PID
        ok, reason = validate_process_identity(99999999)
        self.assertFalse(ok)
        self.assertEqual(reason, "process_not_running")

        # Live Python process
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            # Expected token matched
            ok, reason = validate_process_identity(proc.pid, expected_tokens=["python"])
            self.assertTrue(ok)
            self.assertEqual(reason, "verified")

            # Mismatched expected token
            ok, reason = validate_process_identity(proc.pid, expected_tokens=["notepad.exe", "winword.exe"])
            self.assertFalse(ok)
            self.assertIn("pid_identity_mismatch", reason)

            # Creation time check: simulated recycled PID created before min_create_time
            ok, reason = validate_process_identity(proc.pid, min_create_time=time.time() + 100.0)
            self.assertFalse(ok)
            self.assertIn("pid_reused", reason)
        finally:
            proc.kill()
            proc.wait()

    def test_terminate_process_tree_safely(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            metrics = terminate_process_tree(
                proc.pid,
                exit_code=15,
                expected_tokens=["python"],
                timeout=3.0,
            )
            self.assertTrue(metrics["success"])
            self.assertFalse(default_pid_alive(proc.pid))
            self.assertIn(proc.pid, metrics["pids_terminated"])
            self.assertLess(metrics["elapsed_seconds"], 2.0)
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
