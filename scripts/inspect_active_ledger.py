#!/usr/bin/env python3
import sqlite3
import json
import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
db_path = ROOT / "reports" / "quantum_campaigns" / "amde2six003_quantum" / "quantum_ledger.sqlite3"

if not db_path.exists():
    # Fallback to local reports directory if present
    db_path = ROOT / "reports" / "quantum_ledger.sqlite3"

con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute('SELECT COUNT(*), MIN(started_at), MAX(ended_at) FROM model_trials WHERE status="ok"')
total_ok, min_start, max_end = cur.fetchone()
cur.execute('SELECT COUNT(*) FROM model_trials WHERE status!="ok"')
total_fail = cur.fetchone()[0]
cur.execute('SELECT COUNT(*), MIN(started_at), MAX(ended_at) FROM rounds WHERE status="completed"')
rounds_ok, r_min, r_max = cur.fetchone()
cur.execute('SELECT COUNT(DISTINCT task_id), COUNT(DISTINCT method), COUNT(DISTINCT config_json) FROM model_trials')
tasks, methods, configs = cur.fetchone()

t0 = datetime.datetime.fromisoformat(min_start.replace("Z", "+00:00"))
t1 = datetime.datetime.fromisoformat(max_end.replace("Z", "+00:00"))
elapsed_hours = (t1 - t0).total_seconds() / 3600.0
trials_per_hour = total_ok / elapsed_hours if elapsed_hours > 0 else 0

result = {
    "total_successful_trials": total_ok,
    "total_failed_trials": total_fail,
    "completed_rounds": rounds_ok,
    "unique_tasks": tasks,
    "unique_methods": methods,
    "unique_configurations": configs,
    "start_time": min_start,
    "latest_time": max_end,
    "elapsed_hours": round(elapsed_hours, 3),
    "empirical_trials_per_hour": round(trials_per_hour, 1),
}
print("--- QUANTUM CAMPAIGN LEDGER ---")
print(json.dumps(result, indent=2))

# Classical campaign inspection
classical_dir = ROOT / "reports" / "campaigns" / "amde2six003_classical"
if classical_dir.exists():
    state_file = classical_dir / "state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            cstate = json.load(f)
        rounds = cstate.get("rounds", [])
        total_classical_trials = sum(r.get("trials_completed", 0) for r in rounds)
        print("\n--- CLASSICAL CAMPAIGN STATE ---")
        print(json.dumps({
            "campaign_id": cstate.get("campaign_id"),
            "status": cstate.get("status"),
            "current_round": cstate.get("current_round"),
            "completed_rounds": len([r for r in rounds if r.get("status") == "completed"]),
            "total_trials_completed": total_classical_trials,
            "created_at": cstate.get("created_at"),
            "updated_at": cstate.get("updated_at"),
        }, indent=2))

