"""Evaluator request boundaries, cache keys, and aggregate validation."""

from __future__ import annotations

import copy
import os
import unittest

from amde.config import METRIC_VERSION
from amde.discovery.factory import default_genome
from amde.discovery.genome import Genome
from amde.errors import ConfigError
from amde.evaluation.evaluator import Evaluator, evaluation_cache_key
from amde.evaluation.metrics import METRIC_HASH, aggregate_scores


def _aggregate(task_ids: list[str], seeds: list[int], mean: float = 0.5) -> dict:
    records = []
    for task_id in task_ids:
        for seed in seeds:
            records.append(
                {
                    "task_id": task_id,
                    "family": f"family_{task_id}",
                    "suite": "classical",
                    "kind": "regression",
                    "seed": int(seed),
                    "score": float(mean),
                    "elapsed_seconds": 0.001,
                }
            )
    return aggregate_scores(records, ci_level=0.8)


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = self._make_config()
        self.evaluator = Evaluator(self.config)
        self.specs = self.config.task_specs()[:2]
        self.seeds = [1, 2]

    def _make_config(self):
        from tests.support import make_config

        return make_config(self)

    def _request(self, **overrides):
        values = {
            "genome": None,
            "baseline": "majority_mean",
            "stage": "quick",
            "task_specs": self.specs,
            "seeds": self.seeds,
            "epochs_scale": 0.2,
            "timeout_seconds": 30.0,
            "manifest_hash": "manifest-1",
            "metadata": {"note": "unit"},
        }
        values.update(overrides)
        return self.evaluator.build_request(**values)

    def test_payload_is_dev_only_and_contains_no_challenge_data(self) -> None:
        request = self._request()
        payload = self.evaluator.payload(request)
        self.assertEqual(payload["kind"], "dev")
        self.assertFalse(payload["include_challenge"])
        self.assertFalse(payload["challenge_used"] if "challenge_used" in payload else False)
        self.assertNotIn("challenge", payload)
        self.assertNotIn("tasks", payload)
        self.assertIsNone(payload["genome"])
        self.assertEqual(payload["baseline"], "majority_mean")
        self.assertEqual([spec["task_id"] for spec in payload["task_specs"]], ["tab_linear", "tab_xor"])
        for spec in payload["task_specs"]:
            self.assertFalse({"x", "y", "input", "output", "challenge"} & set(spec))

    def test_request_identity_rules_fail_closed(self) -> None:
        genome = Genome(default_genome("A"))
        with self.assertRaises(ConfigError):
            self._request(genome=genome, baseline="majority_mean")
        with self.assertRaises(ConfigError):
            self._request(genome=None, baseline="linear_sgd")
        with self.assertRaises(ConfigError):
            self._request(task_specs=[])

    def test_cache_key_is_stable_and_protocol_sensitive(self) -> None:
        base = dict(
            protocol_hash="protocol",
            manifest_hash="manifest",
            stage="quick",
            identity="candidate",
            task_ids=["a", "b"],
            seeds=[1, 2],
            epochs_scale=0.25,
            metric_hash="metric",
            dataset_version="dataset",
            engine_version="engine",
        )
        first = evaluation_cache_key(**base)
        reordered = dict(base, task_ids=["b", "a"], seeds=[2, 1])
        self.assertEqual(first, evaluation_cache_key(**reordered))
        for key, value in (
            ("protocol_hash", "protocol-2"),
            ("manifest_hash", "manifest-2"),
            ("stage", "confirm"),
            ("identity", "other"),
            ("task_ids", ["a"]),
            ("seeds", [1]),
            ("epochs_scale", 1.0),
            ("metric_hash", "metric-2"),
            ("dataset_version", "dataset-2"),
            ("engine_version", "engine-2"),
            ("device", "cuda"),
        ):
            with self.subTest(key=key):
                changed = dict(base)
                changed[key] = value
                self.assertNotEqual(first, evaluation_cache_key(**changed))

    def test_valid_aggregate_is_normalized(self) -> None:
        request = self._request()
        aggregate = _aggregate(request.task_ids, request.seeds)
        result = self.evaluator._normalize(
            {"status": "ok", "result": {"aggregate": aggregate}},
            request,
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertIs(result["aggregate"], aggregate)
        self.assertEqual(result["cache_key"], request.cache_key)
        self.assertEqual(result["stage"], "quick")
        self.assertFalse(result["cache_hit"])

    def test_malformed_or_leaky_aggregates_are_protocol_errors(self) -> None:
        request = self._request()
        aggregate = _aggregate(request.task_ids, request.seeds)

        def normalized(value: dict) -> dict:
            return self.evaluator._normalize({"status": "ok", "result": {"aggregate": value}}, request)

        with_challenge = copy.deepcopy(aggregate)
        with_challenge["challenge"] = {"score": 0.9}
        self.assertEqual(normalized(with_challenge)["status"], "protocol_error")

        with_examples = copy.deepcopy(aggregate)
        with_examples["records"][0]["x"] = [1.0, 2.0]
        with_examples["records"][0]["y"] = 1.0
        self.assertEqual(normalized(with_examples)["status"], "protocol_error")

        missing_task = copy.deepcopy(aggregate)
        missing_task["per_task"].pop(request.task_ids[0])
        self.assertEqual(normalized(missing_task)["status"], "protocol_error")

        missing_seed = copy.deepcopy(aggregate)
        first_task = next(iter(missing_seed["per_task"].values()))
        first_task["scores"].pop(int(request.seeds[0]))
        self.assertEqual(normalized(missing_seed)["status"], "protocol_error")

        out_of_range = copy.deepcopy(aggregate)
        first_task = next(iter(out_of_range["per_task"].values()))
        first_task["scores"][int(request.seeds[0])] = 1.5
        self.assertEqual(normalized(out_of_range)["status"], "protocol_error")

        wrong_metric = copy.deepcopy(aggregate)
        wrong_metric["metric_hash"] = "0" * 64
        self.assertEqual(normalized(wrong_metric)["status"], "protocol_error")

        wrong_version = copy.deepcopy(aggregate)
        wrong_version["metric_version"] = METRIC_VERSION + "-old"
        self.assertEqual(normalized(wrong_version)["status"], "protocol_error")

    def test_worker_failures_are_classified_not_zero_filled(self) -> None:
        request = self._request()
        cases = [
            ({"status": "error", "error_type": "candidate", "error_code": "bad"}, "invalid"),
            ({"status": "error", "error_type": "infrastructure", "error_code": "bad"}, "error"),
            ({"status": "timeout", "error_code": "timeout"}, "timeout"),
            ({"status": "memory", "error_code": "memory_cap"}, "memory"),
            ({"status": "interrupted", "error_code": "stop_requested"}, "interrupted"),
            ({"status": "mystery"}, "error"),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                result = self.evaluator._normalize(raw, request)
                self.assertEqual(result["status"], expected)
                self.assertIsNone(result["aggregate"])
        unknown = self.evaluator._normalize({"status": "mystery"}, request)
        self.assertEqual(unknown["error_type"], "infrastructure")
        self.assertEqual(unknown["error_code"], "worker_failure")

    def test_real_majority_evaluation_in_a_spawned_process(self) -> None:
        request = self._request()
        result = self.evaluator.evaluate(request)
        self.assertEqual(result["status"], "ok", result)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["metric_hash"], METRIC_HASH)
        self.assertEqual(aggregate["metric_version"], METRIC_VERSION)
        self.assertEqual(set(aggregate["per_task"]), set(request.task_ids))
        self.assertEqual(aggregate["n_seeds"], len(request.seeds))
        self.assertEqual(len(aggregate["records"]), len(self.specs) * len(self.seeds))
        self.assertNotEqual(result["worker_pid"], os.getpid())
        self.assertFalse(result["cache_hit"])
        for record in aggregate["records"]:
            self.assertFalse({"x", "y", "input", "output", "examples"} & set(record))

    def test_candidate_parameter_limit_is_invalid_not_infrastructure(self) -> None:
        self.config.set("evaluation.parameter_limit", 0)
        evaluator = Evaluator(self.config)
        request = evaluator.build_request(
            genome=Genome(default_genome("A")),
            baseline=None,
            stage="quick",
            task_specs=self.specs[:1],
            seeds=[1],
            epochs_scale=0.2,
            timeout_seconds=30.0,
            manifest_hash="manifest-1",
        )
        result = evaluator.evaluate(request)
        self.assertEqual(result["status"], "invalid", result)
        self.assertEqual(result["error_type"], "candidate")
        self.assertEqual(result["error_code"], "parameter_limit")
        self.assertIsNone(result["aggregate"])


if __name__ == "__main__":
    unittest.main()
