from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from amde.util import read_json_safe


FAKE_WORKER = r'''from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--root', required=True); p.add_argument('--campaign-id', required=True)
p.add_argument('--campaign-stop-file', required=True)
a, _ = p.parse_known_args()
root = Path(a.root); cid = a.campaign_id; stop = Path(a.campaign_stop_file)
name = Path(sys.argv[0]).name
if name == 'research_campaign.py':
    base = root / 'reports' / 'campaigns' / cid
else:
    base = root / 'reports' / 'quantum_campaigns' / cid
base.mkdir(parents=True, exist_ok=True)
lock = base / 'campaign.lock'; state = base / 'state.json'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
old = {}
try: old = json.loads(state.read_text(encoding='utf-8'))
except Exception: pass
starts = int(old.get('starts', 0)) + 1

def write(status):
    data = {'campaign_id': cid, 'status': status, 'starts': starts, 'heartbeat_at': now(), 'updated_at': now(), 'next_round': 0, 'rounds': []}
    tmp = state.with_suffix('.tmp'); tmp.write_text(json.dumps(data), encoding='utf-8'); os.replace(tmp, state)
lock.write_text(json.dumps({'pid': os.getpid(), 'run_id': cid, 'heartbeat_at': now()}), encoding='utf-8')
write('running')
while not stop.exists():
    time.sleep(0.2); write('running')
write('stopped')
lock.unlink(missing_ok=True)
'''


class AMDE2LifecycleIntegrationTests(unittest.TestCase):
    def _call(self, root: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env.get("PYTHONPATH", ""))
        env["OMP_NUM_THREADS"] = "1"
        return subprocess.run(
            [sys.executable, str(root / "scripts" / "amde2.py"), *args, "--root", str(root)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False,
        )

    def test_campaign_start_status_stop_resume_is_campaign_scoped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amde2-life-") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts" / "amde2.py", scripts / "amde2.py")
            shutil.copy2(ROOT / "scripts" / "amde2_logrelay.py", scripts / "amde2_logrelay.py")
            (scripts / "research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            cid = "life_test_01"
            common = ("--campaign-id", cid)
            try:
                started = self._call(
                    root, "start", *common,
                    "--quantum-device", "cpu", "--min-free-gb", "1",
                    "--classical-max-gb", "1", "--quantum-max-gb", "1",
                    "--memory-reserve-gb", "0.5", "--health-interval-seconds", "2",
                    "--manager-stale-seconds", "60", "--stale-heartbeat-seconds", "120",
                    "--shutdown-grace-seconds", "30",
                )
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                state_path = root / "reports" / "amde2" / cid / "state.json"
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        if state.get("workers", {}).get("classical", {}).get("alive") and state.get("workers", {}).get("quantum", {}).get("alive"):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.2)
                else:
                    base = root / "reports" / "amde2" / cid
                    details = []
                    for candidate in (base / "state.json", base / "guardian.json", base / "logs" / "guardian.log", base / "logs" / "manager.log", base / "logs" / "classical.log", base / "logs" / "quantum.log"):
                        if candidate.exists():
                            details.append(f"\n--- {candidate.name} ---\n{candidate.read_text(encoding='utf-8', errors='replace')[-5000:]}")
                    self.fail("AMDE 2.0 workers did not both reach alive state" + "".join(details))

                stop1 = self._call(root, "stop", *common, "--wait-seconds", "30")
                self.assertEqual(stop1.returncode, 0, stop1.stdout + stop1.stderr)
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state.get("status"), "stopped")
                self.assertFalse((root / "runs" / "STOP").exists())
                classical_state = {}
                quantum_state = {}
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        classical_state = json.loads((root / "reports" / "campaigns" / f"{cid}_classical" / "state.json").read_text(encoding="utf-8"))
                        quantum_state = json.loads((root / "reports" / "quantum_campaigns" / f"{cid}_quantum" / "state.json").read_text(encoding="utf-8"))
                        if classical_state.get("status") == "stopped" and quantum_state.get("status") == "stopped":
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.1)
                self.assertEqual(classical_state.get("status"), "stopped")
                self.assertEqual(quantum_state.get("status"), "stopped")

                resumed = self._call(root, "resume", *common)
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        if state.get("status") == "running" and all(item.get("alive") for item in state.get("workers", {}).values()):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.2)
                else:
                    self.fail("AMDE 2.0 did not resume both workers")

                stop2 = self._call(root, "stop", *common, "--wait-seconds", "30")
                self.assertEqual(stop2.returncode, 0, stop2.stdout + stop2.stderr)
                c2 = json.loads((root / "reports" / "campaigns" / f"{cid}_classical" / "state.json").read_text(encoding="utf-8"))
                q2 = json.loads((root / "reports" / "quantum_campaigns" / f"{cid}_quantum" / "state.json").read_text(encoding="utf-8"))
                self.assertGreaterEqual(c2.get("starts", 0), 2)
                self.assertGreaterEqual(q2.get("starts", 0), 2)
            finally:
                # A failed assertion must not leave a detached guardian pointing
                # at a TemporaryDirectory that is about to be deleted.
                state_path = root / "reports" / "amde2" / cid / "state.json"
                if state_path.exists():
                    cleanup = self._call(root, "stop", *common, "--wait-seconds", "10", timeout=15)
                    if cleanup.returncode != 0:
                        self._call(root, "force-stop", *common, timeout=15)


    def test_guardian_recovers_after_manager_process_crash(self) -> None:
        import psutil
        with tempfile.TemporaryDirectory(prefix="amde2-recover-") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts" / "amde2.py", scripts / "amde2.py")
            shutil.copy2(ROOT / "scripts" / "amde2_logrelay.py", scripts / "amde2_logrelay.py")
            (scripts / "research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            cid = "recovery_test_01"
            common = ("--campaign-id", cid)
            state_path = root / "reports" / "amde2" / cid / "state.json"
            try:
                started = self._call(
                    root, "start", *common, "--quantum-device", "cpu", "--min-free-gb", "1",
                    "--classical-max-gb", "1", "--quantum-max-gb", "1", "--memory-reserve-gb", "0.5",
                    "--health-interval-seconds", "2", "--manager-stale-seconds", "60",
                    "--stale-heartbeat-seconds", "120", "--shutdown-grace-seconds", "30",
                )
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                deadline = time.monotonic() + 15
                first_manager_pid = None
                while time.monotonic() < deadline:
                    try:
                        state = read_json_safe(state_path, default={})
                        workers = state.get("workers", {})
                        first_manager_pid = state.get("manager_pid")
                        if first_manager_pid and len(workers) == 2 and all(v.get("alive") for v in workers.values()):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.2)
                else:
                    self.fail("initial manager/workers did not become healthy")

                psutil.Process(int(first_manager_pid)).kill()
                recovered = False
                deadline = time.monotonic() + 40
                state = {}
                while time.monotonic() < deadline:
                    try:
                        state = read_json_safe(state_path, default={})
                        workers = state.get("workers", {})
                        if (state.get("manager_pid") and state.get("manager_pid") != first_manager_pid
                                and state.get("status") == "running" and len(workers) == 2
                                and all(v.get("alive") for v in workers.values())):
                            recovered = True
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.25)
                self.assertTrue(recovered, "guardian did not restart the manager and both workers after a manager crash")
                self.assertGreaterEqual(state.get("manager_recoveries", 0), 1)
                stop = self._call(root, "stop", *common, "--wait-seconds", "30", timeout=40)
                self.assertEqual(stop.returncode, 0, stop.stdout + stop.stderr)
            finally:
                if state_path.exists():
                    cleanup = self._call(root, "stop", *common, "--wait-seconds", "10", timeout=15)
                    if cleanup.returncode != 0:
                        self._call(root, "force-stop", *common, timeout=15)

    def test_persistent_deadline_and_recovery_lifecycle(self) -> None:
        import psutil
        with tempfile.TemporaryDirectory(prefix="amde2-deadline-") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts" / "amde2.py", scripts / "amde2.py")
            shutil.copy2(ROOT / "scripts" / "amde2_logrelay.py", scripts / "amde2_logrelay.py")
            (scripts / "research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(FAKE_WORKER, encoding="utf-8")
            cid = "deadline_int_01"
            common = ("--campaign-id", cid)
            plan_path = root / "reports" / "amde2" / cid / "plan.json"
            state_path = root / "reports" / "amde2" / cid / "state.json"
            guardian_path = root / "reports" / "amde2" / cid / "control" / "guardian.json"

            result_log = {
                "campaign_id": cid,
                "steps": [],
                "verified_deadline_persistence": False,
                "manager_recovered": False,
                "guardian_restarted": False,
                "cooperative_shutdown_confirmed": False,
                "all_workers_cleanly_exited": False,
            }

            try:
                # 1. Start campaign with a 25-second persistent deadline (25 / 3600 hours)
                duration_hours = 25.0 / 3600.0
                start_t = time.time()
                started = self._call(
                    root, "start", *common,
                    "--duration-hours", f"{duration_hours:.6f}",
                    "--quantum-device", "cpu", "--min-free-gb", "1",
                    "--classical-max-gb", "1", "--quantum-max-gb", "1", "--memory-reserve-gb", "0.5",
                    "--health-interval-seconds", "2", "--manager-stale-seconds", "60",
                    "--stale-heartbeat-seconds", "120", "--shutdown-grace-seconds", "30",
                )
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                result_log["steps"].append({"step": "start_campaign", "exit_code": started.returncode})

                # 2. Verify deadline persistence in plan.json and state.json
                self.assertTrue(plan_path.exists())
                plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                initial_deadline = plan_data.get("deadline_utc")
                self.assertIsNotNone(initial_deadline)
                self.assertEqual(plan_data.get("campaign_type"), "timed")

                # Wait for workers to be healthy
                deadline = time.monotonic() + 15
                first_mgr_pid = None
                while time.monotonic() < deadline:
                    if state_path.exists():
                        try:
                            st = json.loads(state_path.read_text(encoding="utf-8"))
                            workers = st.get("workers", {})
                            first_mgr_pid = st.get("manager_pid")
                            if first_mgr_pid and len(workers) == 2 and all(v.get("alive") for v in workers.values()):
                                break
                        except Exception:
                            pass
                    time.sleep(0.2)
                self.assertIsNotNone(first_mgr_pid, "Manager or workers failed to reach running state")
                result_log["steps"].append({"step": "workers_online", "manager_pid": first_mgr_pid, "deadline_utc": initial_deadline})

                # 3. Simulate Manager crash & restart
                psutil.Process(int(first_mgr_pid)).kill()
                result_log["steps"].append({"step": "kill_manager", "killed_pid": first_mgr_pid})
                mgr_recovered = False
                deadline = time.monotonic() + 25
                second_mgr_pid = None
                while time.monotonic() < deadline:
                    if state_path.exists():
                        try:
                            st = json.loads(state_path.read_text(encoding="utf-8"))
                            new_pid = st.get("manager_pid")
                            if new_pid and new_pid != first_mgr_pid and st.get("status") == "running":
                                second_mgr_pid = new_pid
                                mgr_recovered = True
                                break
                        except Exception:
                            pass
                    time.sleep(0.2)
                self.assertTrue(mgr_recovered, "Manager did not recover after crash")
                result_log["manager_recovered"] = True
                result_log["steps"].append({"step": "manager_recovered", "new_manager_pid": second_mgr_pid, "deadline_utc_preserved": True})

                # 4. Simulate Guardian crash & restart
                guardian_data = json.loads(guardian_path.read_text(encoding="utf-8"))
                first_guardian_pid = guardian_data.get("pid")
                if first_guardian_pid and psutil.pid_exists(int(first_guardian_pid)):
                    psutil.Process(int(first_guardian_pid)).kill()
                    result_log["steps"].append({"step": "kill_guardian", "killed_pid": first_guardian_pid})

                # Launch replacement Guardian
                env = os.environ.copy()
                env["PYTHONPATH"] = str(SRC) + (os.pathsep + env.get("PYTHONPATH", ""))
                env["OMP_NUM_THREADS"] = "1"
                guardian_proc = subprocess.Popen(
                    [sys.executable, str(root / "scripts" / "amde2.py"), "guardian", *common, "--root", str(root)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                result_log["steps"].append({"step": "restart_guardian", "new_guardian_pid": guardian_proc.pid})
                result_log["guardian_restarted"] = True

                # Confirm deadline remains unchanged across both manager and guardian restarts
                plan_after = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(plan_after.get("deadline_utc"), initial_deadline)
                self.assertAlmostEqual(plan_after.get("duration_hours"), duration_hours, places=5)

                # 5. Wait for persistent deadline expiration and confirm cooperative shutdown
                shutdown_deadline = time.monotonic() + 35
                clean_shutdown = False
                final_state = {}
                while time.monotonic() < shutdown_deadline:
                    if state_path.exists():
                        try:
                            st = json.loads(state_path.read_text(encoding="utf-8"))
                            if st.get("status") == "stopped":
                                clean_shutdown = True
                                final_state = st
                                break
                        except Exception:
                            pass
                    time.sleep(0.3)
                self.assertTrue(clean_shutdown, f"Campaign did not stop at persistent deadline: {final_state}")
                result_log["cooperative_shutdown_confirmed"] = True
                result_log["verified_deadline_persistence"] = True

                # 6. Confirm all worker processes are cleanly exited (no orphans)
                workers = final_state.get("workers", {})
                all_exited = all(not v.get("alive") for v in workers.values())
                self.assertTrue(all_exited, f"Not all workers exited cleanly: {workers}")
                result_log["all_workers_cleanly_exited"] = True

                # 7. Verification check
                verified = self._call(root, "verify", *common)
                self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
                result_log["steps"].append({"step": "verify_campaign", "exit_code": verified.returncode})

            finally:
                # Cleanup if anything still running
                if state_path.exists():
                    self._call(root, "stop", *common, "--wait-seconds", "5", timeout=10)
                # Save test results to root PERSISTENT_DEADLINE_TEST_RESULTS.json
                out_path = ROOT / "PERSISTENT_DEADLINE_TEST_RESULTS.json"
                out_path.write_text(json.dumps(result_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
