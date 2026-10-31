from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from amde.runtime import amde2
from amde.runtime.heartbeat import StopController


class AMDE2RuntimeTests(unittest.TestCase):
    def _plan_args(self, root: Path, campaign_id: str = "test_campaign") -> argparse.Namespace:
        return argparse.Namespace(
            root=str(root),
            campaign_id=campaign_id,
            seed=130,
            hours_per_round=0.05,
            max_trials_per_round=12,
            classical_max_gb=2.0,
            quantum_max_gb=2.0,
            min_free_gb=1.0,
            quantum_device="cpu",
            classical_threads=4,
            quantum_cpu_threads=2,
            quantum_epochs=2,
            quantum_batch_size=8,
            quantum_train_size=16,
            quantum_val_size=8,
            quantum_test_size=16,
            quantum_max_configs=3,
            quantum_max_qubits=8,
            quantum_statevector_mb=64,
            memory_reserve_gb=0.5,
            stale_heartbeat_seconds=120,
            manager_stale_seconds=60,
            stable_after_seconds=60,
            max_worker_restarts_per_hour=4,
            degraded_retry_seconds=60,
            shutdown_grace_seconds=30,
            health_interval_seconds=2,
            max_gpu_temperature_c=85.0,
        )

    def test_plan_checksum_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, args.campaign_id)
            paths["base"].mkdir(parents=True)
            amde2._atomic_json(paths["plan"], plan)
            self.assertEqual(amde2._read_plan(paths)["plan_hash"], plan["plan_hash"])
            altered = dict(plan)
            altered["seed"] = 999
            amde2._atomic_json(paths["plan"], altered)
            with self.assertRaisesRegex(ValueError, "checksum"):
                amde2._read_plan(paths)

    def test_isolated_campaign_stop_files_do_not_touch_legacy_or_other_campaigns(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "research_campaign.py").touch()
            (root / "scripts" / "quantum_research_campaign.py").touch()
            (root / "scripts" / "amde2_logrelay.py").touch()
            plan_a = amde2._make_plan(self._plan_args(root, "campaign_a"), root)
            manager_a = amde2.AMDE2Manager(root, "campaign_a", plan_a)
            manager_a.workers = manager_a._build_workers()
            manager_a._request_worker_stops("test stop")
            self.assertTrue(manager_a.paths["classical_stop"].is_file())
            self.assertTrue(manager_a.paths["quantum_stop"].is_file())
            self.assertFalse((root / "runs" / "STOP").exists())
            plan_b = amde2._make_plan(self._plan_args(root, "campaign_b"), root)
            paths_b = amde2._paths(root, "campaign_b")
            self.assertFalse(paths_b["stop"].exists())
            self.assertFalse(paths_b["classical_stop"].exists())
            self.assertFalse(paths_b["quantum_stop"].exists())

    def test_pid_identity_check_rejects_recycled_or_unrelated_pid(self) -> None:
        import os
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(amde2._pid_matches(os.getpid(), Path(temp), "not_this_campaign", "not_this_script.py"))
            self.assertFalse(amde2._pid_matches(0, Path(temp), "campaign", "amde2.py"))

    def test_isolated_stop_controller_ignores_global_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            global_runs = root / "runs"
            global_runs.mkdir()
            (global_runs / "STOP").write_text("legacy stop", encoding="utf-8")
            isolated = StopController(run_dir, global_runs_dir=None)
            shared = StopController(run_dir, global_runs_dir=global_runs)
            self.assertFalse(isolated.requested())
            self.assertTrue(shared.requested())
            isolated.request("local only")
            self.assertTrue(isolated.requested())
            self.assertTrue((global_runs / "STOP").is_file())

    def test_restart_budget_uses_persistable_epoch_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            manager = amde2.AMDE2Manager(root, args.campaign_id, plan)
            worker = amde2.Worker("classical", "test_campaign_classical", root / "state.json",
                                  root / "lock.json", root / "STOP", root / "worker.log", [])
            manager._schedule_restart(worker, 1, "failed")
            payload = manager._worker_status_payload(worker)
            self.assertEqual(payload["restarts"], 1)
            self.assertEqual(len(payload["failure_times_epoch"]), 1)
            self.assertGreater(payload["failure_times_epoch"][0], 1_600_000_000)


    def test_manager_watchdog_stop_does_not_create_user_campaign_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            manager = amde2.AMDE2Manager(root, args.campaign_id, plan)
            manager._request_worker_stops("watchdog test")
            guardian = amde2.Guardian(root, args.campaign_id)
            guardian._request_manager_stale_shutdown(12345, 300.0)
            self.assertFalse(manager.paths["stop"].exists())
            self.assertTrue(manager.paths["classical_stop"].exists())
            self.assertTrue(manager.paths["quantum_stop"].exists())

    def test_backup_and_sha256_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, args.campaign_id)
            paths["base"].mkdir(parents=True)
            paths["control"].mkdir(parents=True)
            amde2._atomic_json(paths["plan"], plan)
            amde2._atomic_json(paths["state"], {"campaign_id": args.campaign_id, "status": "stopped"})
            qdir = root / "reports" / "quantum_campaigns" / "test_campaign_quantum"
            qdir.mkdir(parents=True)
            db = qdir / "quantum_ledger.sqlite3"
            con = sqlite3.connect(db)
            try:
                con.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY, value TEXT)")
                con.execute("INSERT INTO sample(value) VALUES('durable')")
                con.commit()
            finally:
                con.close()
            (paths["base"] / "notes.txt").write_text("evidence", encoding="utf-8")
            (root / "config").mkdir(parents=True)
            (root / "config" / "default.json").write_text("{}", encoding="utf-8")
            sdir = root / "storage"
            sdir.mkdir()
            sdb = sdir / "discovery_archive.sqlite3"
            con = sqlite3.connect(sdb)
            try:
                con.execute("CREATE TABLE archive(id INTEGER PRIMARY KEY)")
                con.commit()
            finally:
                con.close()
            (root / "data").mkdir(parents=True)
            (root / "data" / "local_benchmark.txt").write_text("benchmark evidence", encoding="utf-8")
            output = Path(temp) / "campaign_backup.zip"
            backup_args = argparse.Namespace(root=str(root), campaign_id=args.campaign_id, output=str(output))
            self.assertEqual(amde2.backup_campaign(backup_args), 0)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".zip.partial").exists())
            import zipfile
            with zipfile.ZipFile(output) as archive:
                self.assertIn("data/local_benchmark.txt", archive.namelist())
            verify_args = argparse.Namespace(archive=str(output))
            self.assertEqual(amde2.verify_backup(verify_args), 0)
            import zipfile
            with zipfile.ZipFile(output) as archive:
                self.assertIn("config/default.json", archive.namelist())
                self.assertIn("storage/discovery_archive.sqlite3", archive.namelist())
            restored_root = Path(temp) / "restored_project"
            restore_args = argparse.Namespace(archive=str(output), root=str(restored_root), overwrite=False)
            self.assertEqual(amde2.restore_campaign(restore_args), 0)
            self.assertEqual((restored_root / "reports" / "amde2" / args.campaign_id / "notes.txt").read_text(encoding="utf-8"), "evidence")
            self.assertEqual((restored_root / "data" / "local_benchmark.txt").read_text(encoding="utf-8"), "benchmark evidence")
            self.assertTrue((restored_root / "reports" / "quantum_campaigns" / "test_campaign_quantum" / "quantum_ledger.sqlite3").is_file())
            restored_paths = amde2._paths(restored_root, args.campaign_id)
            restored_plan = amde2._read_plan(restored_paths)
            self.assertEqual(restored_plan["root"], str(restored_root.resolve()))
            restored_state = json.loads(restored_paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(restored_state["plan_hash"], restored_plan["plan_hash"])
            self.assertEqual(restored_state["status"], "stopped")
            self.assertTrue((restored_root / "config" / "default.json").is_file())
            with self.assertRaises(FileExistsError):
                amde2.restore_campaign(restore_args)

    def test_plan_mode_and_persistent_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "research_campaign.py").touch()
            (root / "scripts" / "quantum_research_campaign.py").touch()
            (root / "scripts" / "amde2_logrelay.py").touch()

            # 1. Timed run with persistent deadline
            args_timed = self._plan_args(root, "timed_test")
            args_timed.duration_hours = 6.0
            args_timed.mode = "hybrid"
            plan_timed = amde2._make_plan(args_timed, root)
            self.assertEqual(plan_timed["campaign_type"], "timed")
            self.assertEqual(plan_timed["duration_hours"], 6.0)
            self.assertIsNotNone(plan_timed["deadline_utc"])
            self.assertTrue(plan_timed["deadline_utc"].endswith("Z"))

            # 2. Infinite run
            args_inf = self._plan_args(root, "inf_test")
            args_inf.duration_hours = None
            args_inf.mode = "classical"
            plan_inf = amde2._make_plan(args_inf, root)
            self.assertEqual(plan_inf["campaign_type"], "infinite")
            self.assertIsNone(plan_inf["deadline_utc"])
            self.assertEqual(plan_inf["mode"], "classical")

            # 3. Worker filtering by mode
            mgr_classical = amde2.AMDE2Manager(root, "inf_test", plan_inf)
            workers_c = mgr_classical._build_workers()
            self.assertEqual(len(workers_c), 1)
            self.assertEqual(workers_c[0].name, "classical")

            args_q = self._plan_args(root, "q_test")
            args_q.mode = "quantum"
            plan_q = amde2._make_plan(args_q, root)
            mgr_q = amde2.AMDE2Manager(root, "q_test", plan_q)
            workers_q = mgr_q._build_workers()
            self.assertEqual(len(workers_q), 1)
            self.assertEqual(workers_q[0].name, "quantum")

    def test_schedule_restart_uses_monotonic_time(self) -> None:
        import time
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            manager = amde2.AMDE2Manager(root, args.campaign_id, plan)
            worker = amde2.Worker("classical", "test_classical", root / "state.json",
                                  root / "lock.json", root / "STOP", root / "worker.log", [])
            mono_before = time.monotonic()
            manager._schedule_restart(worker, 1, "failed")
            mono_after = time.monotonic()
            # next_start_at should be near monotonic time + delay (10s), NOT wall clock (1.7e9)
            self.assertGreaterEqual(worker.next_start_at, mono_before + 9.0)
            self.assertLessEqual(worker.next_start_at, mono_after + 15.0)

    def test_worker_exit_failure_classification(self) -> None:
        from unittest.mock import MagicMock
        from amde.errors import (
            EXIT_CONFIG_ERROR,
            EXIT_DATA_INTEGRITY,
            EXIT_RESOURCE_EXHAUSTION,
            EXIT_RUNTIME_ERROR,
            FAILURE_CATEGORY_CONFIG,
            FAILURE_CATEGORY_DATA_INTEGRITY,
            FAILURE_CATEGORY_RESOURCE,
            FAILURE_CATEGORY_RUNTIME,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root)
            plan = amde2._make_plan(args, root)
            manager = amde2.AMDE2Manager(root, args.campaign_id, plan)
            worker = amde2.Worker("quantum", "test_quantum", root / "state.json",
                                  root / "lock.json", root / "STOP", root / "worker.log", [])

            # 1. Configuration error during startup -> fatal needs_attention, NO restart
            worker.started_at_monotonic = time.monotonic()
            worker.restarts = 0
            amde2._atomic_json(worker.state_path, {"next_round": 0, "status": "running", "failure_category": FAILURE_CATEGORY_CONFIG})
            mock_proc = MagicMock()
            mock_proc.poll.return_value = EXIT_CONFIG_ERROR
            worker.process = mock_proc
            manager._schedule_restart = lambda w, c, s: self.fail("should not restart on configuration error")
            manager._handle_worker_exit(worker)
            self.assertEqual(worker.status, "needs_attention")
            self.assertEqual(manager.target_status, "needs_attention")

            # 2. Runtime error immediately after startup (duration < 5s) -> RESTARTS under budget
            worker.started_at_monotonic = time.monotonic() - 2.0
            worker.restarts = 0
            amde2._atomic_json(worker.state_path, {"next_round": 0, "status": "running", "failure_category": FAILURE_CATEGORY_RUNTIME})
            restarted = []
            manager._schedule_restart = lambda w, c, s: restarted.append((w.name, c, s))
            manager.target_status = None
            mock_proc.poll.return_value = EXIT_RUNTIME_ERROR
            worker.process = mock_proc
            manager._handle_worker_exit(worker)
            self.assertEqual(len(restarted), 1)
            self.assertEqual(restarted[0][1], EXIT_RUNTIME_ERROR)

            # 3. Runtime error during a later round (round 3) -> RESTARTS under budget
            worker.started_at_monotonic = time.monotonic() - 300.0
            worker.restarts = 1
            amde2._atomic_json(worker.state_path, {"next_round": 3, "status": "running", "failure_category": FAILURE_CATEGORY_RUNTIME})
            restarted.clear()
            manager.target_status = None
            mock_proc.poll.return_value = EXIT_RUNTIME_ERROR
            worker.process = mock_proc
            manager._handle_worker_exit(worker)
            self.assertEqual(len(restarted), 1)
            self.assertEqual(restarted[0][1], EXIT_RUNTIME_ERROR)

            # 4. Resource exhaustion (code 3 or category resource_exhaustion) -> PAUSES campaign
            worker.started_at_monotonic = time.monotonic() - 60.0
            amde2._atomic_json(worker.state_path, {"next_round": 1, "status": "running", "failure_category": FAILURE_CATEGORY_RESOURCE})
            manager._schedule_restart = lambda w, c, s: self.fail("should not restart on resource exhaustion")
            manager.target_status = None
            mock_proc.poll.return_value = EXIT_RESOURCE_EXHAUSTION
            worker.process = mock_proc
            manager._handle_worker_exit(worker)
            self.assertEqual(worker.status, "resource_paused")
            self.assertEqual(manager.target_status, "paused_resource")

            # 5. Data integrity failure -> fatal needs_attention, NEVER silently restart
            worker.started_at_monotonic = time.monotonic() - 60.0
            amde2._atomic_json(worker.state_path, {"next_round": 2, "status": "running", "failure_category": FAILURE_CATEGORY_DATA_INTEGRITY, "error": "corrupt sqlite ledger"})
            manager._schedule_restart = lambda w, c, s: self.fail("FATAL: must NEVER restart a data integrity failure!")
            manager.target_status = None
            mock_proc.poll.return_value = EXIT_DATA_INTEGRITY
            worker.process = mock_proc
            manager._handle_worker_exit(worker)
            self.assertEqual(worker.status, "needs_attention")
            self.assertEqual(manager.target_status, "needs_attention")
            self.assertIn("data integrity", worker.last_error.lower())

            # 6. Unknown exception (e.g. exit code 99) -> RESTARTS under budget
            worker.started_at_monotonic = time.monotonic() - 10.0
            amde2._atomic_json(worker.state_path, {"next_round": 1, "status": "running"})
            restarted.clear()
            manager._schedule_restart = lambda w, c, s: restarted.append((w.name, c, s))
            manager.target_status = None
            mock_proc.poll.return_value = 99
            worker.process = mock_proc
            manager._handle_worker_exit(worker)
            self.assertEqual(len(restarted), 1)
            self.assertEqual(restarted[0][1], 99)

    def test_start_campaign_directory_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "dir_test")
            # Pre-create directory (as a test harness or log directory might do)
            camp_dir = root / "reports" / "amde2" / "dir_test"
            (camp_dir / "logs").mkdir(parents=True)
            # Starting campaign should succeed because plan.json does not exist yet
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, "dir_test")
            self.assertFalse(paths["plan"].exists())
            paths["base"].mkdir(parents=True, exist_ok=True)
            amde2._atomic_json(paths["plan"], plan)
            self.assertTrue(paths["plan"].exists())
            # Attempting to start again with existing plan must raise FileExistsError
            with self.assertRaises(FileExistsError):
                amde2.start_campaign(args)

    def test_verify_campaign_detects_missing_or_corrupt_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "verify_test")
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, "verify_test")
            paths["base"].mkdir(parents=True)
            paths["control"].mkdir(parents=True)
            amde2._atomic_json(paths["plan"], plan)
            amde2._atomic_json(paths["state"], {"campaign_id": "verify_test", "plan_hash": plan["plan_hash"], "status": "running"})

            # Create a run directory that has completed trials but NO checkpoint
            run_dir = root / "runs" / "verify_test_classical_r00000"
            run_dir.mkdir(parents=True)
            db = run_dir / "ledger.sqlite3"
            con = sqlite3.connect(db)
            try:
                con.execute("CREATE TABLE trials(trial_id TEXT PRIMARY KEY, status TEXT)")
                con.execute("INSERT INTO trials(trial_id, status) VALUES('t1', 'ok')")
                con.commit()
            finally:
                con.close()

            # verify must FAIL because checkpoint is missing for run directory with trials
            ret = amde2.verify_campaign(argparse.Namespace(root=str(root), campaign_id="verify_test"))
            self.assertEqual(ret, 1)

            # Now add a valid checkpoint envelope
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir()
            state = {"round": 0, "status": "completed"}
            envelope = {
                "schema_version": "1.0.0",
                "checksum": amde2._json_hash(state),
                "state": state,
            }
            amde2._atomic_json(ckpt_dir / "latest.json", envelope)
            ret_ok = amde2.verify_campaign(argparse.Namespace(root=str(root), campaign_id="verify_test"))
            self.assertEqual(ret_ok, 0)

    def test_critical_json_fail_closed(self) -> None:
        from amde.util import CorruptJSONError, read_json_critical, read_json_safe
        from amde.errors import DataIntegrityError
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corrupt_file = root / "corrupt.json"
            corrupt_file.write_text("{bad json syntax: true,", encoding="utf-8")

            # Corrupt file must raise CorruptJSONError / DataIntegrityError, NOT return default!
            with self.assertRaises(CorruptJSONError):
                read_json_safe(corrupt_file, default={})

            with self.assertRaises(CorruptJSONError):
                read_json_critical(corrupt_file)

            # Missing optional file returns default
            missing_file = root / "missing.json"
            self.assertEqual(read_json_safe(missing_file, default={"opt": 1}), {"opt": 1})

            # Missing required file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                read_json_critical(missing_file)

            # Corrupt state.json in campaign directory must fail closed on Manager run without overwriting
            args = self._plan_args(root, "fail_closed_test")
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, "fail_closed_test")
            paths["base"].mkdir(parents=True)
            amde2._atomic_json(paths["plan"], plan)
            paths["state"].write_text("{corrupt state payload...", encoding="utf-8")
            manager = amde2.AMDE2Manager(root, "fail_closed_test", plan)
            with self.assertRaises(DataIntegrityError):
                manager.run()
            # Confirm file was NOT wiped/overwritten
            self.assertEqual(paths["state"].read_text(encoding="utf-8"), "{corrupt state payload...")

    def test_clock_and_backoff_across_reboot_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "reboot_test")
            plan = amde2._make_plan(args, root)
            paths = amde2._paths(root, "reboot_test")
            paths["base"].mkdir(parents=True)
            amde2._atomic_json(paths["plan"], plan)

            now_epoch = time.time()
            persisted_state = {
                "schema_version": amde2.SCHEMA,
                "campaign_id": "reboot_test",
                "plan_hash": plan["plan_hash"],
                "status": "degraded",
                "workers": {
                    "classical": {
                        "restarts": 3,
                        "consecutive_failures": 3,
                        "failure_times_epoch": [now_epoch - 900, now_epoch - 300, now_epoch - 20],
                    }
                }
            }
            amde2._atomic_json(paths["state"], persisted_state)

            scripts_dir = root / "scripts"
            scripts_dir.mkdir()
            for s in ("research_campaign.py", "quantum_research_campaign.py", "amde2_logrelay.py"):
                (scripts_dir / s).write_text("# dummy", encoding="utf-8")

            manager = amde2.AMDE2Manager(root, "reboot_test", plan)
            saved_workers = persisted_state["workers"]
            manager.workers = manager._build_workers()
            cw = next(w for w in manager.workers if w.name == "classical")
            saved = saved_workers["classical"]
            cw.restarts = saved["restarts"]
            cw.consecutive_failures = saved["consecutive_failures"]
            cw.failure_times = saved["failure_times_epoch"]
            delay = 40.0
            elapsed = 20.0
            remaining = delay - elapsed
            cw.next_start_at = time.monotonic() + remaining
            self.assertGreater(cw.next_start_at, time.monotonic() + 15.0)
            self.assertLessEqual(cw.next_start_at, time.monotonic() + 25.0)

    def test_sample_process_tree_telemetry(self) -> None:
        import os
        from amde.runtime.resources import sample_process_tree
        # Test current process
        current = sample_process_tree(os.getpid())
        self.assertTrue(current["alive"])
        self.assertEqual(current["pid"], os.getpid())
        self.assertGreaterEqual(current["rss_mb"], 0.0)
        self.assertGreaterEqual(current["working_set_mb"], 0.0)
        self.assertGreaterEqual(current["private_memory_mb"], 0.0)
        self.assertGreaterEqual(current["total_cpu_percent"], 0.0)
        self.assertTrue(current["sampling_complete"])
        self.assertIn(os.getpid(), current["observed_pids"])

        # Test non-existent process
        dead = sample_process_tree(99999999)
        self.assertFalse(dead["alive"])
        self.assertEqual(dead["cpu_percent"], 0.0)

        # Test None PID
        none_sample = sample_process_tree(None)
        self.assertFalse(none_sample["alive"])

    def test_plan_with_quantum_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "prof_test")
            args.quantum_profile = "medium"
            plan = amde2._make_plan(args, root)
            self.assertEqual(plan["quantum_profile"], "medium")
            self.assertEqual(plan["quantum_batch_size"], 64)
            self.assertEqual(plan["quantum_epochs"], 12)
            self.assertEqual(plan["quantum_train_size"], 256)


if __name__ == "__main__":
    unittest.main()
