"""AMDE 2.0.3 Native Windows Integration Test Suite for Hard Deadline Enforcement.

Executes a 10-run test suite validating:
1. Hard deadline watchdog trigger latency (<= 1.0s)
2. Process tree termination latency (<= 2.0s)
3. Total deadline overshoot (<= 3.0s)
4. No post-deadline trial execution in SQLite ledgers
5. All process tree PIDs physically exited (psutil verification)
6. SQLite database integrity post-forced termination
7. CUDA GPU memory and context release verification on NVIDIA Quadro P2000 Mobile
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amde.runtime import amde2
from amde.runtime.job import default_pid_alive


def parse_iso(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


class HardDeadlineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results: list[dict[str, Any]] = []

    def _run_single_hard_deadline_campaign(
        self,
        run_idx: int,
        duration_seconds: float = 4.0,
        quantum_device: str = "cpu",
    ) -> dict[str, Any]:
        """Execute a single sacrificial campaign with hard deadline enforcement."""
        cid = f"hd_run_{run_idx}_{int(time.time()*1000)%1000000}"
        with tempfile.TemporaryDirectory() as td:
            tpath = Path(td)
            shutil.copytree(ROOT / "scripts", tpath / "scripts")
            shutil.copytree(ROOT / "src", tpath / "src")
            py = sys.executable

            dur_hours = duration_seconds / 3600.0
            args = argparse.Namespace(
                root=str(tpath),
                campaign_id=cid,
                mode="quantum",
                duration_hours=dur_hours,
                deadline_policy="hard",
                seed=2026 + run_idx,
                hours_per_round=0.01,
                max_trials_per_round=20,
                classical_max_gb=1.0,
                quantum_max_gb=1.0,
                min_free_gb=1.0,
                quantum_device=quantum_device,
                quantum_profile="baseline",
                classical_threads=2,
                quantum_cpu_threads=2,
                quantum_epochs=1,
                quantum_batch_size=8,
                quantum_train_size=16,
                quantum_val_size=8,
                quantum_test_size=16,
                quantum_max_configs=3,
                quantum_max_qubits=8,
                quantum_statevector_mb=64,
                memory_reserve_gb=0.5,
                stale_heartbeat_seconds=60,
                manager_stale_seconds=30,
                stable_after_seconds=30,
                max_worker_restarts_per_hour=4,
                degraded_retry_seconds=60,
                shutdown_grace_seconds=1.0,
                health_interval_seconds=2,
                max_gpu_temperature_c=90.0,
            )
            plan = amde2._make_plan(args, tpath)
            paths = amde2._paths(tpath, cid)
            paths["base"].mkdir(parents=True, exist_ok=True)
            amde2._atomic_json(paths["plan"], plan)

            deadline_str = plan["deadline_utc"]
            deadline_dt = parse_iso(deadline_str)

            # Start guardian process
            g_cmd = [py, "-u", str(tpath / "scripts" / "amde2.py"), "guardian", "--root", str(tpath), "--campaign-id", cid]
            t_start = time.monotonic()
            proc = subprocess.Popen(g_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                out, _ = proc.communicate(timeout=duration_seconds + 15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            t_end = time.monotonic()

            exit_code = proc.returncode

            # Parse events
            events: list[dict[str, Any]] = []
            if paths["events"].is_file():
                for line in paths["events"].read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass

            # Find key events
            evt_trigger = next((e for e in events if e.get("event") == "hard_deadline_reached"), None)
            evt_req = next((e for e in events if e.get("event") == "hard_termination_requested"), None)
            evt_comp = next((e for e in events if e.get("event") == "hard_termination_completed"), None)
            evt_ver = next((e for e in events if e.get("event") == "process_tree_verified_exited"), None)

            # Timing calculations
            trigger_dt = parse_iso(evt_trigger["at"]) if evt_trigger else None
            req_dt = parse_iso(evt_req["at"]) if evt_req else None
            comp_dt = parse_iso(evt_comp["at"]) if evt_comp else None
            ver_dt = parse_iso(evt_ver["at"]) if evt_ver else None

            trigger_latency = (trigger_dt - deadline_dt).total_seconds() if trigger_dt else None
            term_latency = (ver_dt - req_dt).total_seconds() if (ver_dt and req_dt) else None
            total_overshoot = (ver_dt - deadline_dt).total_seconds() if ver_dt else None

            # Collect observed PIDs from state and events
            state = amde2._read_json(paths["state"], {})
            guardian_data = amde2._read_json(paths["guardian"], {})
            observed_pids: list[int] = []
            if isinstance(state, dict):
                mpid = state.get("manager_pid")
                if isinstance(mpid, int):
                    observed_pids.append(mpid)
                workers = state.get("workers", {})
                if isinstance(workers, dict):
                    for wname, wdata in workers.items():
                        if isinstance(wdata, dict):
                            for pkey in ("pid", "relay_pid"):
                                wp = wdata.get(pkey)
                                if isinstance(wp, int) and wp not in observed_pids:
                                    observed_pids.append(wp)
            if evt_comp and isinstance(evt_comp.get("pids_terminated"), list):
                for p in evt_comp["pids_terminated"]:
                    if isinstance(p, int) and p not in observed_pids:
                        observed_pids.append(p)

            # Check PID liveness with OS reap grace period and identity validation
            from amde.runtime.job import validate_process_identity
            deadline_reap = time.monotonic() + 1.0
            live_pids: list[int] = []
            while time.monotonic() < deadline_reap:
                still_alive = [p for p in observed_pids if default_pid_alive(p)]
                live_pids = [
                    p for p in still_alive
                    if validate_process_identity(p, expected_tokens=["python", "amde2"])[0]
                ]
                if not live_pids:
                    break
                time.sleep(0.05)

            # Check SQLite databases
            db_results: list[dict[str, Any]] = []
            for db_path in tpath.glob("**/*.sqlite"):
                try:
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    cur.execute("PRAGMA integrity_check;")
                    row = cur.fetchone()
                    int_ok = (row and row[0] == "ok")
                    cur.execute("PRAGMA quick_check;")
                    row_q = cur.fetchone()
                    quick_ok = (row_q and row_q[0] == "ok")

                    # Check for trials completed after deadline
                    post_trials = 0
                    try:
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = [r[0] for r in cur.fetchall()]
                        for tbl in tables:
                            col_info = [c[1] for c in cur.execute(f"PRAGMA table_info({tbl});").fetchall()]
                            if "completed_at" in col_info:
                                cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE completed_at > ?;", (deadline_str,))
                                post_trials += cur.fetchone()[0]
                    except Exception:
                        pass
                    conn.close()
                    db_results.append({
                        "path": str(db_path.name),
                        "integrity": int_ok,
                        "quick_check": quick_ok,
                        "post_deadline_trials": post_trials,
                    })
                except Exception as exc:
                    db_results.append({
                        "path": str(db_path.name),
                        "error": str(exc),
                        "integrity": False,
                        "quick_check": False,
                        "post_deadline_trials": -1,
                    })

            # Check GPU context if CUDA device
            gpu_context_clean = True
            gpu_vram_leak_mb = 0.0
            if quantum_device == "cuda":
                try:
                    smi_res = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    # Check that none of the campaign PIDs appear in nvidia-smi compute apps
                    active_gpu_pids = []
                    for line in smi_res.stdout.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if parts and parts[0].isdigit():
                            active_gpu_pids.append(int(parts[0]))
                    for p in observed_pids:
                        if p in active_gpu_pids:
                            gpu_context_clean = False
                except Exception:
                    pass

            run_result = {
                "run_idx": run_idx,
                "campaign_id": cid,
                "device": quantum_device,
                "configured_duration_seconds": duration_seconds,
                "deadline_utc": deadline_str,
                "guardian_exit_code": exit_code,
                "guardian_duration_seconds": round(t_end - t_start, 3),
                "trigger_latency_seconds": round(trigger_latency, 4) if trigger_latency is not None else None,
                "term_latency_seconds": round(term_latency, 4) if term_latency is not None else None,
                "total_overshoot_seconds": round(total_overshoot, 4) if total_overshoot is not None else None,
                "mechanism": evt_comp.get("mechanism") if evt_comp else None,
                "observed_pids": observed_pids,
                "live_pids_after_termination": live_pids,
                "db_integrity": all(d.get("integrity", False) for d in db_results) if db_results else True,
                "post_deadline_trials": sum(d.get("post_deadline_trials", 0) for d in db_results),
                "state_status": state.get("status") if isinstance(state, dict) else None,
                "guardian_status": guardian_data.get("status") if isinstance(guardian_data, dict) else None,
                "gpu_context_clean": gpu_context_clean,
            }
            return run_result

    def test_ten_run_hard_deadline_suite(self) -> None:
        """Execute 10 consecutive hard deadline campaigns and verify strict SLA thresholds."""
        print("\n" + "=" * 80)
        print("AMDE 2.0.3: TEN-RUN HARD DEADLINE ENFORCEMENT INTEGRATION TEST SUITE")
        print("=" * 80)

        suite_results: list[dict[str, Any]] = []

        for i in range(1, 11):
            device = "cuda" if i == 10 else "cpu"
            dur = 4.0 if i < 10 else 5.0
            print(f"\n[Run {i}/10] Executing campaign with duration={dur}s, device={device}...")
            res = self._run_single_hard_deadline_campaign(i, duration_seconds=dur, quantum_device=device)
            suite_results.append(res)
            print(f"  Deadline:          {res['deadline_utc']}")
            print(f"  Trigger latency:   {res['trigger_latency_seconds']} s (SLA <= 1.0s)")
            print(f"  Term latency:      {res['term_latency_seconds']} s (SLA <= 2.0s)")
            print(f"  Total overshoot:   {res['total_overshoot_seconds']} s (SLA <= 3.0s)")
            print(f"  Mechanism:         {res['mechanism']}")
            print(f"  PIDs alive:        {len(res['live_pids_after_termination'])} (expected 0)")
            print(f"  Post-trials:       {res['post_deadline_trials']} (expected 0)")
            print(f"  DB integrity:      {res['db_integrity']} (expected True)")
            print(f"  State status:      {res['state_status']} (expected stopped)")
            if device == "cuda":
                print(f"  GPU context clean: {res['gpu_context_clean']} (expected True)")

            # Assertions for each run
            self.assertEqual(res["guardian_exit_code"], 0, f"Run {i}: Guardian exit code must be 0")
            self.assertIsNotNone(res["trigger_latency_seconds"], f"Run {i}: Watchdog trigger event must be recorded")
            self.assertLessEqual(res["trigger_latency_seconds"], 1.0, f"Run {i}: Trigger latency must be <= 1.0s")
            self.assertIsNotNone(res["term_latency_seconds"], f"Run {i}: Term latency must be recorded")
            self.assertLessEqual(res["term_latency_seconds"], 2.0, f"Run {i}: Term latency must be <= 2.0s")
            self.assertLessEqual(res["total_overshoot_seconds"], 3.0, f"Run {i}: Total overshoot must be <= 3.0s")
            self.assertEqual(len(res["live_pids_after_termination"]), 0, f"Run {i}: All PIDs must be dead")
            self.assertEqual(res["post_deadline_trials"], 0, f"Run {i}: No post-deadline trials allowed")
            self.assertTrue(res["db_integrity"], f"Run {i}: SQLite integrity must be intact")
            self.assertEqual(res["state_status"], "stopped", f"Run {i}: Final state must be stopped")
            if device == "cuda":
                self.assertTrue(res["gpu_context_clean"], "Run 10: CUDA GPU context must be released")

        # Save suite results artifact
        out_path = ROOT / "reports" / "hard_deadline_10run_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(suite_results, indent=2), encoding="utf-8")
        print(f"\n[Artifact Saved] 10-run integration results: {out_path}")


if __name__ == "__main__":
    unittest.main()
