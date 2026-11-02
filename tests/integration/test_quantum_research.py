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


class QuantumCampaignIntegrationTests(unittest.TestCase):
    def test_finite_quantum_campaign_records_all_methods_and_is_terminal(self):
        with tempfile.TemporaryDirectory(prefix="amde_qcampaign_") as temp:
            root = Path(temp)
            command = [
                sys.executable, str(ROOT / "scripts" / "quantum_research_campaign.py"),
                "--root", str(root), "--campaign-id", "qcampaign_e2e", "--rounds", "1",
                "--seed", "123", "--device", "cpu", "--epochs", "1", "--batch-size", "8",
                "--train-size", "16", "--val-size", "8", "--test-size", "8",
                "--max-configs", "3", "--max-qubits", "8", "--max-statevector-mb", "64",
                "--max-campaign-gb", "1", "--min-free-gb", "1",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + "\n" + completed.stderr)
            campaign = root / "reports" / "quantum_campaigns" / "qcampaign_e2e"
            state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "round_limit_reached")
            self.assertEqual(state["next_round"], 1)
            self.assertTrue((campaign / "leaderboard.csv").is_file())
            self.assertTrue((campaign / "REPORT.md").is_file())
            report = (campaign / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("- Status: `round_limit_reached`", report)
            import sqlite3
            con = sqlite3.connect(campaign / "quantum_ledger.sqlite3")
            try:
                statuses = dict(con.execute("SELECT status, COUNT(*) FROM model_trials GROUP BY status").fetchall())
                methods = {row[0] for row in con.execute("SELECT DISTINCT method FROM model_trials").fetchall()}
                maze = con.execute("SELECT status, result_json FROM maze_trials WHERE round_index=0").fetchone()
            finally:
                con.close()
            self.assertEqual(statuses, {"ok": 24})
            self.assertEqual(methods, {
                "mlp_cpu_baseline", "rbf_kernel_ridge_cpu_baseline",
                "quantum_fidelity_kernel_ridge", "vqc_statevector",
            })
            self.assertIsNotNone(maze)
            self.assertEqual(maze[0], "ok")
            maze_result = json.loads(maze[1])
            self.assertFalse(maze_result["grover"]["oracle_is_reversible_gate_circuit"])
            self.assertIn("No end-to-end quantum speedup", maze_result["interpretation"])

    def test_hybrid_supervisor_starts_both_workers_and_stops_cooperatively(self):
        with tempfile.TemporaryDirectory(prefix="amde_hybrid_supervisor_") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts" / "hybrid_research_campaign.py", scripts / "hybrid_research_campaign.py")
            (root / "runs").mkdir()
            worker_stub = r'''from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--campaign-id'); a,_=p.parse_known_args()
root=Path(a.root); cid=a.campaign_id
if cid.endswith('_classical'):
    base=root/'reports'/'campaigns'/cid; stop=root/'runs'/'STOP'
else:
    base=root/'reports'/'quantum_campaigns'/cid; stop=root/'runs'/'QUANTUM_STOP'
base.mkdir(parents=True, exist_ok=True)
(base/'campaign.lock').write_text(json.dumps({'pid':os.getpid(),'run_id':cid}))
state={'campaign_id':cid,'status':'running','pid':os.getpid()}
(base/'state.json').write_text(json.dumps(state))
while not stop.exists() and not (root/'runs'/'HYBRID_STOP').exists():
    state['heartbeat_at']=time.time(); (base/'state.json').write_text(json.dumps(state)); time.sleep(.1)
state['status']='stop_requested'; (base/'state.json').write_text(json.dumps(state))
(base/'campaign.lock').unlink(missing_ok=True)
'''
            (scripts / "research_campaign.py").write_text(worker_stub, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(worker_stub, encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "scripts" / "hybrid_research_campaign.py"),
                "--root", str(root), "--campaign-id", "hybrid_e2e", "--quantum-device", "cpu",
                "--hours-per-round", "0.01", "--max-trials-per-round", "20",
                "--classical-max-gb", "1", "--quantum-max-gb", "1", "--min-free-gb", "1",
                "--shutdown-grace-seconds", "15", "--max-restarts", "1",
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                classical_state = root / "reports" / "campaigns" / "hybrid_e2e_classical" / "state.json"
                quantum_state = root / "reports" / "quantum_campaigns" / "hybrid_e2e_quantum" / "state.json"
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and not (classical_state.exists() and quantum_state.exists()):
                    if process.poll() is not None:
                        break
                    time.sleep(.1)
                self.assertTrue(classical_state.exists(), "classical worker did not start")
                self.assertTrue(quantum_state.exists(), "quantum worker did not start")
                (root / "runs" / "HYBRID_STOP").write_text("integration-test-stop\n", encoding="utf-8")
                output, _ = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, msg=output)
                supervisor_state = json.loads((root / "reports" / "hybrid_campaigns" / "hybrid_e2e" / "supervisor_state.json").read_text(encoding="utf-8"))
                self.assertEqual(supervisor_state["status"], "stopped")
                self.assertEqual(supervisor_state["children"]["classical"]["status"], "stopped")
                self.assertEqual(supervisor_state["children"]["quantum"]["status"], "stopped")
                self.assertFalse((root / "reports" / "campaigns" / "hybrid_e2e_classical" / "campaign.lock").exists())
                self.assertFalse((root / "reports" / "quantum_campaigns" / "hybrid_e2e_quantum" / "campaign.lock").exists())
            finally:
                if process.poll() is None:
                    (root / "runs" / "HYBRID_STOP").write_text("test-cleanup-stop\n", encoding="utf-8")
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


    def test_hybrid_supervisor_treats_stopped_worker_as_cooperative_terminal_not_crash(self):
        with tempfile.TemporaryDirectory(prefix="amde_hybrid_stopped_") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            (root / "runs").mkdir()
            classical_stub = r'''from __future__ import annotations
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--campaign-id'); a,_=p.parse_known_args()
base=Path(a.root)/'reports'/'campaigns'/a.campaign_id
base.mkdir(parents=True, exist_ok=True)
(base/'state.json').write_text(json.dumps({'campaign_id':a.campaign_id,'status':'stopped'}), encoding='utf-8')
'''
            quantum_stub = r'''from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--campaign-id'); a,_=p.parse_known_args()
root=Path(a.root); base=root/'reports'/'quantum_campaigns'/a.campaign_id
base.mkdir(parents=True, exist_ok=True)
(base/'campaign.lock').write_text(json.dumps({'pid':os.getpid(),'run_id':a.campaign_id}), encoding='utf-8')
state={'campaign_id':a.campaign_id,'status':'running','pid':os.getpid()}
(base/'state.json').write_text(json.dumps(state), encoding='utf-8')
while not (root/'runs'/'QUANTUM_STOP').exists() and not (root/'runs'/'HYBRID_STOP').exists():
    time.sleep(.05)
state['status']='stop_requested'
(base/'state.json').write_text(json.dumps(state), encoding='utf-8')
(base/'campaign.lock').unlink(missing_ok=True)
'''
            (scripts / "research_campaign.py").write_text(classical_stub, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(quantum_stub, encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "scripts" / "hybrid_research_campaign.py"),
                "--root", str(root), "--campaign-id", "hybrid_stopped_e2e", "--quantum-device", "cpu",
                "--hours-per-round", "0.01", "--max-trials-per-round", "20",
                "--classical-max-gb", "1", "--quantum-max-gb", "1", "--min-free-gb", "1",
                "--shutdown-grace-seconds", "10", "--max-restarts", "2",
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                state_path = root / "reports" / "hybrid_campaigns" / "hybrid_stopped_e2e" / "supervisor_state.json"
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if state_path.exists():
                        try:
                            state = json.loads(state_path.read_text(encoding="utf-8"))
                            if state.get("status") == "stopped":
                                break
                        except (OSError, json.JSONDecodeError):
                            pass
                    if process.poll() is not None:
                        break
                    time.sleep(.1)
                output, _ = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, msg=output)
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "stopped")
                self.assertEqual(state["children"]["classical"]["last_exit_code"], 0)
                self.assertEqual(state["children"]["classical"]["restarts"], 0)
                self.assertEqual(state["children"]["classical"]["status"], "stopped")
                self.assertEqual(state["children"]["quantum"]["status"], "stopped")
                self.assertFalse((root / "reports" / "quantum_campaigns" / "hybrid_stopped_e2e_quantum" / "campaign.lock").exists())
            finally:
                if process.poll() is None:
                    (root / "runs" / "HYBRID_STOP").write_text("test-cleanup-stop\\n", encoding="utf-8")
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    def test_detached_hybrid_control_start_status_and_cooperative_stop(self):
        with tempfile.TemporaryDirectory(prefix="amde_hybrid_control_") as temp:
            root = Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts" / "hybrid_research_campaign.py", scripts / "hybrid_research_campaign.py")
            (root / "runs").mkdir()
            worker_stub = r'''from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--campaign-id'); a,_=p.parse_known_args()
root=Path(a.root); cid=a.campaign_id
if cid.endswith('_classical'):
    base=root/'reports'/'campaigns'/cid; stop=root/'runs'/'STOP'
else:
    base=root/'reports'/'quantum_campaigns'/cid; stop=root/'runs'/'QUANTUM_STOP'
base.mkdir(parents=True, exist_ok=True)
(base/'campaign.lock').write_text(json.dumps({'pid':os.getpid(),'run_id':cid}), encoding='utf-8')
state={'campaign_id':cid,'status':'running','pid':os.getpid()}
(base/'state.json').write_text(json.dumps(state), encoding='utf-8')
while not stop.exists() and not (root/'runs'/'HYBRID_STOP').exists():
    state['heartbeat_at']=time.time(); (base/'state.json').write_text(json.dumps(state), encoding='utf-8'); time.sleep(.05)
state['status']='stop_requested'; (base/'state.json').write_text(json.dumps(state), encoding='utf-8')
(base/'campaign.lock').unlink(missing_ok=True)
'''
            (scripts / "research_campaign.py").write_text(worker_stub, encoding="utf-8")
            (scripts / "quantum_research_campaign.py").write_text(worker_stub, encoding="utf-8")
            start_cmd = [
                sys.executable, str(ROOT / "scripts" / "hybrid_control.py"), "start",
                "--root", str(root), "--campaign-id", "hybrid_control_e2e", "--",
                "--quantum-device", "cpu", "--hours-per-round", "0.01",
                "--max-trials-per-round", "20", "--classical-max-gb", "1",
                "--quantum-max-gb", "1", "--min-free-gb", "1",
                "--shutdown-grace-seconds", "10", "--max-restarts", "1",
            ]
            started = subprocess.run(start_cmd, cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertEqual(started.returncode, 0, msg=started.stdout + "\\n" + started.stderr)
            status_path = root / "reports" / "hybrid_campaigns" / "hybrid_control_e2e" / "supervisor_state.json"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if status_path.exists():
                    try:
                        current = json.loads(status_path.read_text(encoding="utf-8"))
                        if current.get("status") == "running" and all(
                            (current.get("children") or {}).get(name, {}).get("pid")
                            for name in ("classical", "quantum")
                        ):
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(.1)
            status = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "hybrid_control.py"), "status",
                "--root", str(root), "--campaign-id", "hybrid_control_e2e",
            ], cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(status.returncode, 0, msg=status.stdout + "\\n" + status.stderr)
            stopped = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "hybrid_control.py"), "stop",
                "--root", str(root), "--campaign-id", "hybrid_control_e2e", "--wait-seconds", "20",
            ], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(stopped.returncode, 0, msg=stopped.stdout + "\\n" + stopped.stderr)
            final_state = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(final_state["status"], "stopped")
            self.assertTrue(all((item or {}).get("status") == "stopped" for item in final_state["children"].values()))
            self.assertTrue((root / "reports" / "hybrid_campaigns" / "hybrid_control_e2e" / "launch.json").is_file())


if __name__ == "__main__":
    unittest.main()
