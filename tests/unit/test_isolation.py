"""Spawned-process isolation, timeout, stop, and memory-cap behavior."""

from __future__ import annotations

import importlib.util
import os
import random
import unittest

from amde.discovery.factory import random_genome
from amde.errors import InfrastructureError
from amde.evaluation.isolation import WorkerJob, run_jobs
from amde.evaluation.worker import dev_worker
from tests.support import make_config


def _payload(
    config,
    *,
    genome=None,
    baseline: str | None = "majority_mean",
    seeds: list[int] | None = None,
    epochs_scale: float = 0.15,
    max_steps: int | None = None,
) -> dict:
    return {
        "kind": "dev",
        "include_challenge": False,
        "genome": genome.to_dict() if genome is not None else None,
        "baseline": baseline,
        "task_specs": config.task_specs()[:1],
        "split_config": dict(config.get("benchmarks.split")),
        "seeds": list(seeds or [1]),
        "epochs_scale": float(epochs_scale),
        "safety": dict(config.get("safety")),
        "max_steps": int(max_steps or config.get("evaluation.max_steps", 300)),
        "parameter_limit": int(config.get("evaluation.parameter_limit", 20000)),
        "ci_level": float(config.get("evaluation.promotion.ci_level", 0.8)),
    }


class IsolationTests(unittest.TestCase):
    def test_success_runs_in_a_different_process(self) -> None:
        config = make_config(self)
        result = run_jobs(
            [WorkerJob(_payload(config), timeout_seconds=30, candidate=False, label="unit")],
            max_workers=1,
        )[0]
        self.assertEqual(result["status"], "ok", result)
        self.assertIsNotNone(result["worker_pid"])
        self.assertNotEqual(result["worker_pid"], os.getpid())
        self.assertEqual(result["exit_code"], 0, result)
        self.assertFalse(result["worker_shutdown_forced"], result)
        self.assertFalse(result["memory_cap_enforced"])
        aggregate = result["result"]["aggregate"]
        self.assertEqual(aggregate["n_tasks"], 1)
        self.assertEqual(aggregate["n_seeds"], 1)
        self.assertFalse(result["result"]["challenge_used"])

    def test_timeout_is_enforced_by_the_parent(self) -> None:
        config = make_config(self)
        # Use a deliberately expensive candidate rather than a trivial baseline:
        # the latter can legitimately finish before the minimum 100 ms timeout
        # on fast/spawn-warm hosts, making this test flaky across environments.
        genome = random_genome("C", random.Random(7), safety=dict(config.get("safety")))
        payload = _payload(
            config,
            genome=genome,
            baseline=None,
            epochs_scale=50.0,
            max_steps=20000,
        )
        result = run_jobs(
            [WorkerJob(payload, timeout_seconds=0.1, candidate=True, label="timeout")],
            max_workers=1,
            poll_seconds=0.01,
        )[0]
        self.assertEqual(result["status"], "timeout", result)
        self.assertEqual(result["error_type"], "candidate")
        self.assertEqual(result["error_code"], "timeout")
        self.assertIsNotNone(result["worker_pid"])

    def test_stop_callback_prevents_worker_start(self) -> None:
        config = make_config(self)
        results = run_jobs(
            [
                WorkerJob(_payload(config), timeout_seconds=30),
                WorkerJob(_payload(config), timeout_seconds=30),
            ],
            max_workers=2,
            stop_callback=lambda: True,
        )
        self.assertEqual([item["status"] for item in results], ["interrupted", "interrupted"])
        for item in results:
            self.assertEqual(item["error_code"], "stop_requested")
            self.assertIsNone(item["worker_pid"])

    def test_infrastructure_errors_are_not_candidate_errors(self) -> None:
        result = run_jobs(
            [WorkerJob({"kind": "not-a-worker"}, timeout_seconds=10)],
            max_workers=1,
        )[0]
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["error_type"], "infrastructure")
        self.assertIn("InfrastructureError", str(result["error_code"]))
        self.assertEqual(result["exit_code"], 0, result)
        self.assertFalse(result["worker_shutdown_forced"], result)

    def test_dev_worker_rejects_challenge_payloads_directly(self) -> None:
        with self.assertRaises(InfrastructureError):
            dev_worker({"include_challenge": True})

    def test_empty_job_list_is_a_no_op(self) -> None:
        self.assertEqual(run_jobs([], max_workers=4), [])

    @unittest.skipUnless(importlib.util.find_spec("psutil") is not None, "psutil is not installed")
    def test_memory_cap_is_enforced_when_psutil_is_available(self) -> None:
        config = make_config(self)
        genome = random_genome("C", random.Random(7), safety=dict(config.get("safety")))
        result = run_jobs(
            [
                WorkerJob(
                    _payload(
                        config,
                        genome=genome,
                        baseline=None,
                        epochs_scale=50.0,
                        max_steps=20000,
                    ),
                    timeout_seconds=30,
                    candidate=True,
                    label="memory",
                )
            ],
            max_workers=1,
            memory_cap_mb=1,
        )[0]
        self.assertEqual(result["status"], "memory", result)
        self.assertEqual(result["error_code"], "memory_cap")
        self.assertTrue(result["memory_cap_enforced"])
        self.assertGreater(result["peak_rss_mb"] or 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
