"""Checkpoint integrity/recovery, single-instance locking, and stop files."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from amde.errors import CheckpointError, LockError
from amde.runtime.heartbeat import HeartbeatWriter, StopController
from amde.runtime.lock import SingleInstanceLock
from amde.storage.checkpoint import CheckpointManager
from amde.util import atomic_write_json, read_json


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.holder = tempfile.TemporaryDirectory(prefix="amde_checkpoint_")
        self.addCleanup(self.holder.cleanup)
        self.run_dir = Path(self.holder.name)
        self.manager = CheckpointManager(self.run_dir)

    def test_save_load_and_checksums(self) -> None:
        info = self.manager.save({"value": 1, "population": {}}, generation=0, status="test")
        self.assertTrue(info["checksum"])
        state = self.manager.load()
        self.assertEqual(state["value"], 1)
        self.assertFalse(state["_checkpoint_meta"]["recovered_from_backup"])
        listing = self.manager.list_checksums()
        self.assertTrue(any(item["valid"] for item in listing))

    def test_save_hashes_the_same_immutable_snapshot_it_persists(self) -> None:
        import amde.storage.checkpoint as checkpoint_module
        from amde.util import hash_json as real_hash_json

        source = {"value": 1, "nested": {"elapsed": 1.25}}

        def mutate_caller_after_snapshot(value):
            result = real_hash_json(value)
            source["nested"]["elapsed"] = 9.5
            return result

        with patch.object(checkpoint_module, "hash_json", side_effect=mutate_caller_after_snapshot):
            self.manager.save(source, generation=0, status="snapshot-race-test")
        loaded = self.manager.load()
        self.assertEqual(loaded["value"], 1)
        self.assertEqual(loaded["nested"]["elapsed"], 1.25)
        self.assertEqual(source["nested"]["elapsed"], 9.5)

    def test_backup_recovery_after_latest_corruption(self) -> None:
        self.manager.save({"value": 1}, generation=0)
        self.manager.save({"value": 2}, generation=1)
        self.assertTrue(list(self.manager.backup_dir.glob("gen_*.json")))
        latest = self.manager.latest
        payload = read_json(latest)
        payload["state"]["value"] = 999
        atomic_write_json(latest, payload)
        state = self.manager.load()
        self.assertEqual(state["value"], 1)
        self.assertTrue(state["_checkpoint_meta"]["recovered_from_backup"])

    def test_checkpoint_backups_are_bounded(self) -> None:
        manager = CheckpointManager(self.run_dir, max_backups=2)
        for generation in range(8):
            manager.save({"value": generation}, generation=generation)
        backups = list(manager.backup_dir.glob("gen_*.json"))
        self.assertLessEqual(len(backups), 2)
        self.assertEqual(manager.load()["value"], 7)
        self.assertTrue(all(item["valid"] for item in manager.list_checksums()))

    def test_tampered_checkpoint_without_valid_backup_fails_closed(self) -> None:
        self.manager.save({"value": 1}, generation=0)
        payload = read_json(self.manager.latest)
        payload["checksum"] = "0" * 64
        atomic_write_json(self.manager.latest, payload)
        with self.assertRaises(CheckpointError):
            self.manager.load()

    def test_unsupported_schema_is_rejected(self) -> None:
        self.manager.save({"value": 1}, generation=0)
        payload = read_json(self.manager.latest)
        payload["schema_version"] = "9.9.9"
        atomic_write_json(self.manager.latest, payload)
        with self.assertRaises(CheckpointError):
            self.manager.load()


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.holder = tempfile.TemporaryDirectory(prefix="amde_lock_")
        self.addCleanup(self.holder.cleanup)
        self.path = Path(self.holder.name) / ".amde.lock"

    def test_acquire_heartbeat_release(self) -> None:
        lock = SingleInstanceLock(self.path, run_id="run1", pid_alive=lambda pid: True)
        lock.acquire()
        self.assertTrue(lock.acquired)
        self.assertEqual(lock.read()["run_id"], "run1")
        lock.heartbeat()
        self.assertIn("heartbeat_at", lock.read())
        lock.release()
        self.assertFalse(self.path.exists())

    def test_second_live_instance_is_rejected(self) -> None:
        first = SingleInstanceLock(self.path, run_id="run1", pid_alive=lambda pid: True)
        first.acquire()
        self.addCleanup(first.release)
        second = SingleInstanceLock(self.path, run_id="run2", pid_alive=lambda pid: True)
        with self.assertRaises(LockError):
            second.acquire()

    def test_dead_owner_lock_is_stolen_and_archived(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": "1.0.0",
                "pid": 999999,
                "run_id": "dead",
                "heartbeat_at": "2000-01-01T00:00:00.000Z",
            },
        )
        lock = SingleInstanceLock(self.path, run_id="run2", stale_seconds=1, pid_alive=lambda pid: False)
        lock.acquire()
        self.addCleanup(lock.release)
        self.assertTrue(list(self.path.parent.glob(".amde.lock.stale.*")))
        self.assertEqual(lock.read()["run_id"], "run2")

    def test_stale_heartbeat_with_live_pid_is_stolen(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": "1.0.0",
                "pid": 1234,
                "run_id": "old",
                "heartbeat_at": "2000-01-01T00:00:00.000Z",
            },
        )
        lock = SingleInstanceLock(self.path, run_id="run3", stale_seconds=1, pid_alive=lambda pid: True)
        lock.acquire()
        self.addCleanup(lock.release)
        self.assertEqual(lock.read()["run_id"], "run3")


class HeartbeatStopTests(unittest.TestCase):
    def test_heartbeat_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = HeartbeatWriter(Path(tmp), run_id="run1", interval_seconds=0)
            writer.write({"status": "running", "generation": 2})
            data = writer.read()
            self.assertEqual(data["run_id"], "run1")
            self.assertEqual(data["generation"], 2)
            self.assertIsNotNone(writer.age_seconds())

    def test_stop_controller_run_and_global_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            run_dir = runs / "run1"
            controller = StopController(run_dir, stop_name="STOP", global_runs_dir=runs)
            self.assertFalse(controller.requested())
            path = controller.request("unit test")
            self.assertTrue(path.exists())
            self.assertTrue(controller.requested())
            self.assertEqual(controller.reason(), "unit test")
            controller.clear()
            self.assertFalse(controller.requested())
            controller.request("global", global_stop=True)
            self.assertTrue((runs / "STOP").exists())
            controller.clear(clear_global=True)
            self.assertFalse(controller.requested())


if __name__ == "__main__":
    unittest.main()
