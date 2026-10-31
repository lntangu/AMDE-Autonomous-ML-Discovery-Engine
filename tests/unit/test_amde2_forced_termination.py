"""Comprehensive fault-injection test suite for production forced-termination fallback.

Covers all 8 required scenarios:
A. Cooperative worker exits before timeout (no force-kill).
B. Unresponsive worker ignores stop request (timeout expires, forced termination occurs).
C. Worker with child and grandchild processes (all campaign descendants terminate).
D. Worker PID becomes stale or identity validation fails (no unrelated process termination).
E. Manager shutdown timeout (timeout and escalation recorded).
F. Guardian orphan-recovery timeout (orphan worker detected and safely terminated).
G. Repeated stop requests (idempotent, race-safe).
H. Worker exits exactly around timeout boundary (race-safe classification).
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amde.runtime import amde2
from amde.runtime.job import (
    WindowsJobObject,
    terminate_process_tree,
    validate_process_identity,
)
from amde.runtime.lock import default_pid_alive


class ForcedTerminationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="amde2_ft_test_")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_scenario_a_cooperative_worker_exits_before_timeout(self) -> None:
        """Scenario A: Worker detects cooperative stop marker, exits gracefully before timeout."""
        cid = f"test_a_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        stop_file = paths["control"] / "worker.STOP"

        # Worker script loops checking stop file
        worker_code = f"""
import time, os
from pathlib import Path
stop_path = Path(r'{stop_file}')
for _ in range(100):
    if stop_path.exists():
        exit(0)
    time.sleep(0.1)
exit(1)
"""
        proc = subprocess.Popen([sys.executable, "-c", worker_code])
        try:
            worker = amde2.Worker(
                "classical", cid, paths["base"] / "state.json", paths["control"] / "lock",
                stop_file, paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "running"

            # Create mock manager with 3.0s grace period
            plan = {"stale_heartbeat_seconds": 60, "shutdown_grace_seconds": 3.0}
            mgr = amde2.AMDE2Manager(self.root, cid, plan)
            mgr.workers = [worker]

            # Request cooperative stop
            mgr._request_worker_stops("test_scenario_a")
            self.assertTrue(stop_file.exists())

            # Worker exits within 0.5s (before 3.0s timeout)
            proc.wait(timeout=2.0)
            self.assertEqual(proc.poll(), 0)

            # Heartbeat check should see process is exited and NOT trigger force kill
            mgr._monitor_worker_heartbeats()
            self.assertIsNone(worker.last_error)
            self.assertFalse(paths["events"].exists() and "forced_shutdown_timeout_reached" in paths["events"].read_text())
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_scenario_b_unresponsive_worker_forced_termination(self) -> None:
        """Scenario B: Unresponsive worker ignores stop request; timeout expires and forced kill triggers."""
        cid = f"test_b_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        stop_file = paths["control"] / "worker.STOP"

        # Worker script deliberately ignores stop file and sleeps
        worker_code = """
import time
while True:
    time.sleep(1)
"""
        proc = subprocess.Popen([sys.executable, "-c", worker_code])
        pid = proc.pid
        try:
            worker = amde2.Worker(
                "quantum", cid, paths["base"] / "state.json", paths["control"] / "lock",
                stop_file, paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "running"

            # Short grace period: 1.5 seconds
            plan = {"stale_heartbeat_seconds": 60, "shutdown_grace_seconds": 1.5}
            mgr = amde2.AMDE2Manager(self.root, cid, plan)
            mgr.workers = [worker]

            # 1. Request stop
            mgr._request_worker_stops("test_scenario_b")
            self.assertIsNotNone(worker.stale_stop_requested_at)

            # 2. Before timeout: worker still running, no force kill
            mgr._monitor_worker_heartbeats()
            self.assertTrue(default_pid_alive(pid))

            # 3. Wait for grace period to expire
            time.sleep(1.8)

            # 4. Trigger production monitor loop
            mgr._monitor_worker_heartbeats()

            # 5. Verify worker was physically terminated
            time.sleep(0.3)
            self.assertFalse(default_pid_alive(pid))
            self.assertIsNotNone(proc.poll())
            self.assertIn("forced shutdown timeout", worker.last_error)

            # 6. Verify events emitted
            events_text = paths["events"].read_text(encoding="utf-8")
            self.assertIn("forced_shutdown_timeout_reached", events_text)
            self.assertIn("forced_termination_requested", events_text)
            self.assertIn("forced_termination_completed", events_text)
            self.assertIn("process_tree_verified_exited", events_text)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_scenario_c_worker_with_child_and_grandchild(self) -> None:
        """Scenario C: Worker with child and grandchild processes is completely terminated."""
        cid = f"test_c_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        stop_file = paths["control"] / "worker.STOP"

        # Worker spawns child, child spawns grandchild
        worker_code = """
import subprocess, sys, time
p1 = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; p2 = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)"])
time.sleep(120)
"""
        proc = subprocess.Popen([sys.executable, "-c", worker_code])
        pid = proc.pid
        try:
            worker = amde2.Worker(
                "classical", cid, paths["base"] / "state.json", paths["control"] / "lock",
                stop_file, paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "running"

            plan = {"stale_heartbeat_seconds": 60, "shutdown_grace_seconds": 1.0, "plan_hash": "test_hash"}
            mgr = amde2.AMDE2Manager(self.root, cid, plan)
            mgr.workers = [worker]

            time.sleep(2.0)  # allow child and grandchild to fully initialize

            import psutil  # type: ignore
            root_p = psutil.Process(pid)
            children = root_p.children(recursive=True)
            child_pids = [c.pid for c in children]
            self.assertGreaterEqual(len(child_pids), 2)

            # Assign to job if Windows
            if mgr.job and mgr.job.is_active():
                mgr.job.assign_process(pid)

            mgr._request_worker_stops("test_scenario_c")
            time.sleep(1.2)
            mgr._monitor_worker_heartbeats()

            time.sleep(0.5)
            self.assertFalse(default_pid_alive(pid))
            for cp in child_pids:
                self.assertFalse(default_pid_alive(cp), f"Descendant {cp} should be terminated")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_scenario_d_stale_pid_identity_validation_prevents_kill(self) -> None:
        """Scenario D: PID identity validation rejects mismatched command token, prevents collateral kill."""
        # Validate that if PID belongs to e.g. current test runner or system, it won't be killed
        my_pid = os.getpid()
        ok, reason = validate_process_identity(my_pid, expected_tokens=["non_existent_token_12345"])
        self.assertFalse(ok)
        self.assertIn("pid_identity_mismatch", reason)

        # Calling terminate_process_tree on mismatched identity must abort safely without killing target
        res = terminate_process_tree(my_pid, exit_code=1, expected_tokens=["non_existent_token_12345"])
        self.assertFalse(res["success"])
        self.assertTrue(default_pid_alive(my_pid))
        self.assertIn("Identity validation failed", res["errors"][0])

    def test_scenario_e_manager_shutdown_timeout(self) -> None:
        """Scenario E: Manager shutdown escalation records timeout when worker does not stop."""
        cid = f"test_e_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        stop_file = paths["control"] / "worker.STOP"

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            worker = amde2.Worker(
                "quantum", cid, paths["base"] / "state.json", paths["control"] / "lock",
                stop_file, paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "running"

            plan = {"stale_heartbeat_seconds": 60, "shutdown_grace_seconds": 1.0, "plan_hash": "test_hash"}
            mgr = amde2.AMDE2Manager(self.root, cid, plan)
            mgr.workers = [worker]

            mgr._begin_shutdown("stopped", "test_shutdown_timeout")
            self.assertEqual(worker.status, "stopping")
            self.assertIsNotNone(worker.stale_stop_requested_at)

            time.sleep(1.2)
            mgr._monitor_worker_heartbeats()
            self.assertFalse(default_pid_alive(proc.pid))
            self.assertIn("forced shutdown timeout", worker.last_error)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_scenario_f_guardian_orphan_recovery_timeout(self) -> None:
        """Scenario F: Guardian detects orphan worker after manager exit, timeout expires, forces kill."""
        cid = f"test_f_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)
        paths["events"].parent.mkdir(parents=True, exist_ok=True)

        # Worker script representing an orphaned worker
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            wp = amde2._worker_paths(self.root, cid, "classical")
            wp["lock"].parent.mkdir(parents=True, exist_ok=True)
            amde2._atomic_json(wp["lock"], {"pid": proc.pid, "schema_version": "1.0.0", "updated_at": amde2.utc_now()})
            amde2._atomic_json(paths["state"], {
                "schema_version": "amde-2-state-v1", "campaign_id": cid, "status": "stopped",
                "workers": {"classical": {"pid": proc.pid, "alive": True}},
            })

            guardian = amde2.Guardian(self.root, cid)
            # Recover with short grace period: 1 second
            # Mock _pid_matches to identify proc as orphan
            orig_matches = amde2._pid_matches
            amde2._pid_matches = lambda pid, r, cid, s: pid == proc.pid and default_pid_alive(pid)
            try:
                ok = guardian._recover_orphan_workers(grace_seconds=1)
                self.assertTrue(ok)
                self.assertFalse(default_pid_alive(proc.pid))
            finally:
                amde2._pid_matches = orig_matches
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_scenario_g_repeated_stop_requests_idempotent(self) -> None:
        """Scenario G: Repeated stop requests do not cause duplicate races or reset monotonic grace clock."""
        cid = f"test_g_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            worker = amde2.Worker(
                "classical", cid, paths["base"] / "state.json", paths["control"] / "lock",
                paths["control"] / "worker.STOP", paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "running"

            mgr = amde2.AMDE2Manager(self.root, cid, {"shutdown_grace_seconds": 5.0})
            mgr.workers = [worker]

            mgr._request_worker_stops("first_request")
            first_clock = worker.stale_stop_requested_at
            self.assertIsNotNone(first_clock)

            time.sleep(0.2)
            # Repeated stop request
            mgr._request_worker_stops("second_request")
            # Clock must NOT be reset forward
            self.assertEqual(worker.stale_stop_requested_at, first_clock)
        finally:
            proc.kill()

    def test_scenario_h_worker_exits_around_timeout_boundary(self) -> None:
        """Scenario H: Worker exiting exactly around timeout boundary is handled race-safely."""
        cid = f"test_h_{int(time.time()*1000)%100000000}"
        paths = amde2._paths(self.root, cid)
        paths["control"].mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
        try:
            worker = amde2.Worker(
                "classical", cid, paths["base"] / "state.json", paths["control"] / "lock",
                paths["control"] / "worker.STOP", paths["base"] / "worker.log", []
            )
            worker.process = proc
            worker.status = "stopping"
            worker.stale_stop_requested_at = time.monotonic() - 10.0  # Already timed out

            mgr = amde2.AMDE2Manager(self.root, cid, {"shutdown_grace_seconds": 1.0})
            mgr.workers = [worker]

            # Let process exit right now
            proc.wait(timeout=2.0)

            # Monitor loop should see poll() is not None, not crash or report spurious error
            mgr._monitor_worker_heartbeats()
            self.assertIsNone(worker.last_error)
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
