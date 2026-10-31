from __future__ import annotations

import logging
import unittest
from unittest.mock import patch
from pathlib import Path

from amde.config import ResourcePolicy
from amde.orchestrator import Engine
from amde.runtime.resources import ResourceSampler, detect_hardware
from tests.support import make_project, fake_hardware


class RuntimeMonitoringTests(unittest.TestCase):
    def test_torch_probe_is_skipped_when_no_nvidia_gpu_is_detected(self) -> None:
        root, _ = make_project(self)
        with patch("amde.runtime.resources._nvidia_smi", return_value={"available": False, "reason": "not found"}), \
             patch("amde.runtime.resources._torch_probe") as torch_probe:
            info = detect_hardware(root, probe_torch=True)
        torch_probe.assert_not_called()
        self.assertFalse(info.gpu_available)
        self.assertFalse(info.torch["cuda_available"])
        self.assertIn("skipped", info.torch["tensor_smoke"]["reason"])

    def test_engine_log_rotation_bounds_long_run_logs(self) -> None:
        root, config = make_project(self)
        config.set("logging.level", "INFO")
        config.set("logging.max_bytes", 262144)
        config.set("logging.backup_count", 2)
        config.set("runtime.device", "cpu")
        config.refresh_hashes()
        engine = Engine(config, hardware=fake_hardware(logical_cores=4, ram_available_mb=4096))
        run_dir = root / "runs" / "rotation_test"
        run_dir.mkdir(parents=True)
        logger = engine._configure_logging(run_dir)
        # Keep the regression test quiet while retaining the real rotating file handler.
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, "baseFilename"):
                handler.setLevel(logging.ERROR)
        try:
            message = "x" * 40000
            for index in range(12):
                logger.info("rotation-test %s %s", index, message)
        finally:
            engine._close_logging()
        logs = list(run_dir.glob("engine.log*"))
        self.assertGreaterEqual(len(logs), 2)
        self.assertLessEqual(len(logs), 3)
        self.assertLessEqual(sum(path.stat().st_size for path in logs), 700000)

    def test_hardware_and_resource_sampler_report_memory_and_pressure(self) -> None:
        root, _ = make_project(self)
        hw = detect_hardware(root, probe_torch=False)
        self.assertGreaterEqual(hw.logical_cores, 1)
        self.assertGreater(hw.disk_free_gb, 0.0)
        self.assertIn(hw.memory_source, ("psutil", "windows-api", "proc-meminfo", "unavailable"))
        policy = ResourcePolicy.load(root)
        sampler = ResourceSampler(policy, root, gpu_active=True)
        snap = sampler.snapshot(force=True)
        self.assertIn("ts", snap)
        self.assertIn("disk_free_gb", snap)
        self.assertIn("memory_source", snap)
        reasons = sampler.pressure(
            {
                "ram_total_mb": 32768.0,
                "ram_available_mb": 256.0,
                "disk_free_gb": 0.5,
                "gpu_temperature_c": 92.0,
                "gpu_memory_free_mb": 128.0,
            }
        )
        self.assertEqual(len(reasons), 4)


if __name__ == "__main__":
    unittest.main()
