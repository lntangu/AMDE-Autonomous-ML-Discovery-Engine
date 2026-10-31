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
                classical_state = json.loads((root / "reports" / "campaigns" / f"{cid}_classical" / "state.json").read_text(encoding="utf-8"))
                quantum_state = json.loads((root / "reports" / "quantum_campaigns" / f"{cid}_quantum" / "state.json").read_text(encoding="utf-8"))
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
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    workers = state.get("workers", {})
                    first_manager_pid = state.get("manager_pid")
                    if first_manager_pid and len(workers) == 2 and all(v.get("alive") for v in workers.values()):
                        break
                    time.sleep(0.2)
                else:
                    self.fail("initial manager/workers did not become healthy")

                psutil.Process(int(first_manager_pid)).kill()
                recovered = False
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    workers = state.get("workers", {})
                    if (state.get("manager_pid") and state.get("manager_pid") != first_manager_pid
                            and state.get("status") == "running" and len(workers) == 2
                            and all(v.get("alive") for v in workers.values())):
                        recovered = True
                        break
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


if __name__ == "__main__":
    unittest.main()
