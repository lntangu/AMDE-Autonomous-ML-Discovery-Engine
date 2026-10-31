from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amde.research.ablation import audit_ablation_results, build_ablation_plan
from amde.research.hypotheses import (HypothesisRegistry, audit_hypothesis_batch,
                                      evaluate_hypothesis, validate_hypothesis)
from amde.research.statistics import holm_bonferroni, power_analysis


def hypothesis():
    return {
        "hypothesis_id": "sparse_update_gain_001",
        "claim": "A sparse-aware update rule improves validation performance.",
        "mechanism": "The update rule reduces noisy coordinate updates for sparse gradients.",
        "population": "Tabular sparse benchmark families not used during discovery.",
        "intervention": {"optimizer": "sparse_aware"},
        "comparator": {"optimizer": "adamw"},
        "outcome": "Family-balanced held-out score difference.",
        "direction": "greater",
        "minimum_effect": 0.05,
        "alpha": 0.05,
        "expected_sd": 0.02,
        "target_power": 0.80,
        "benchmark_families": ["sparse_a", "sparse_b", "sparse_c", "sparse_d", "sparse_e", "sparse_f"],
        "seeds": [11, 23, 37, 53, 71],
        "exclusion_criteria": ["Invalid runs are not imputed as zero."],
        "analysis_plan": "Paired family-level bootstrap and exact sign-flip test.",
    }


class ResearchToolkitTests(unittest.TestCase):
    def test_hypothesis_registration_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = HypothesisRegistry(Path(directory) / "hypotheses.sqlite3")
            first = registry.register(hypothesis())
            second = registry.register(hypothesis())
            self.assertTrue(first["inserted"])
            self.assertFalse(second["inserted"])
            changed = hypothesis()
            changed["minimum_effect"] = 0.02
            with self.assertRaisesRegex(ValueError, "different frozen protocol"):
                registry.register(changed)

    def test_hypothesis_evaluation_supports_a_preregistered_effect(self):
        spec = hypothesis()
        result = evaluate_hypothesis(spec, {"sparse_a": .08, "sparse_b": .07,
                                            "sparse_c": .09, "sparse_d": .06,
                                            "sparse_e": .08, "sparse_f": .07},
                                     bootstrap_repetitions=1000)
        self.assertEqual(result["status"], "supported")
        with tempfile.TemporaryDirectory() as directory:
            registry = HypothesisRegistry(Path(directory) / "hypotheses.sqlite3")
            registry.register(spec)
            self.assertEqual(len(registry.record_result(spec["hypothesis_id"], result)), 64)

    def test_hypothesis_requires_exact_family_coverage(self):
        with self.assertRaisesRegex(ValueError, "family coverage mismatch"):
            evaluate_hypothesis(hypothesis(), {"sparse_a": .1})

    def test_hypothesis_batch_audit_applies_holm_to_complete_prespecified_family(self):
        results = []
        for index, p_value in enumerate((0.01, 0.04, 0.03), start=1):
            results.append({
                "schema_version": "amde-hypothesis-result-v1",
                "hypothesis_id": f"hypothesis_{index}",
                "hypothesis_sha256": str(index) * 64,
                "status": "supported",
                "test": {"p_value": p_value},
            })
        payload = {"schema_version": "amde-hypothesis-batch-v1",
                   "expected_hypothesis_ids": ["hypothesis_1", "hypothesis_2", "hypothesis_3"],
                   "results": results}
        audit = audit_hypothesis_batch(payload, alpha=0.05)
        self.assertEqual(audit["correction"], "Holm-Bonferroni")
        self.assertEqual(audit["hypotheses"]["hypothesis_1"]["p_adjusted"], 0.03)
        self.assertFalse(audit["hypotheses"]["hypothesis_2"]["reject"])
        self.assertEqual(audit["hypotheses"]["hypothesis_1"]["familywise_decision"], "supported_after_holm")
        underpowered = {**payload, "results": [
            {**results[0], "status": "inconclusive_underpowered"}, results[1], results[2]
        ]}
        underpowered_audit = audit_hypothesis_batch(underpowered, alpha=0.05)
        self.assertEqual(underpowered_audit["hypotheses"]["hypothesis_1"]["familywise_decision"],
                         "inconclusive_underpowered")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            audit_hypothesis_batch({**payload, "results": results[:2]}, alpha=0.05)

    def test_ablation_plan_is_deterministic_and_bounded(self):
        plan = build_ablation_plan({"optimizer": "adam", "lr": .01},
                                   {"optimizer": ["adam", "sgd", "lion"],
                                    "lr": [.001, .01]}, mode="full_factorial",
                                   seeds=[1, 2, 3])
        self.assertEqual(len(plan["variants"]), 6)
        self.assertEqual(plan["protocol_sha256"], build_ablation_plan(
            {"optimizer": "adam", "lr": .01},
            {"optimizer": ["adam", "sgd", "lion"], "lr": [.001, .01]},
            mode="full_factorial", seeds=[1, 2, 3])["protocol_sha256"])
        with self.assertRaisesRegex(ValueError, "above max_variants"):
            build_ablation_plan({}, {"a": list(range(10)), "b": list(range(10))},
                                mode="full_factorial", seeds=[1, 2, 3], max_variants=20)

    def test_ablation_audit_pairs_by_family_and_seed_and_holm_adjusts(self):
        rows = []
        for family_index, family in enumerate(("f1", "f2", "f3", "f4", "f5", "f6", "f7")):
            for seed in (1, 2, 3):
                control = 0.50 + family_index * .01 + seed * .001
                rows.append({"variant_id": "control", "family_id": family, "seed": seed, "score": control})
                rows.append({"variant_id": "variant_a", "family_id": family, "seed": seed, "score": control + .08})
                rows.append({"variant_id": "variant_b", "family_id": family, "seed": seed, "score": control + .01})
        result = audit_ablation_results({"schema_version": "amde-ablation-results-v1",
                                         "control_variant_id": "control", "results": rows},
                                        bootstrap_repetitions=1000)
        self.assertEqual(result["correction"], "Holm-Bonferroni")
        self.assertTrue(result["variants"]["variant_a"]["multiple_testing"]["reject"])
        adjusted = holm_bonferroni({"a": .01, "b": .04, "c": .03})
        self.assertEqual(adjusted["a"]["p_adjusted"], .03)

    def test_power_analysis_is_explicit_and_deterministic(self):
        result = power_analysis(minimum_effect=0.05, expected_sd=0.02,
                                alpha=0.05, target_power=0.80)
        self.assertEqual(result["method"], "paired_mean_normal_approximation")
        self.assertGreaterEqual(result["recommended_independent_families"], 6)
        self.assertGreaterEqual(result["exact_sign_flip_resolution_floor"], 6)
        self.assertIn("ignores finite-sample", result["assumption"])

    def test_one_sided_exact_test_resolution_floor_tracks_alpha(self):
        result = power_analysis(minimum_effect=0.1, expected_sd=0.2,
                                alpha=0.01, target_power=0.8, direction="greater")
        self.assertEqual(result["exact_sign_flip_resolution_floor"], 7)

    def test_hypothesis_is_not_supported_when_underpowered(self):
        spec = hypothesis()
        spec["expected_sd"] = 0.20
        result = evaluate_hypothesis(spec, {key: 0.20 for key in spec["benchmark_families"]},
                                     bootstrap_repetitions=1000)
        self.assertEqual(result["status"], "inconclusive_underpowered")
        self.assertFalse(result["power_design_adequate"])

    def test_ablation_rejects_missing_pairs_and_too_few_families(self):
        rows = []
        for family in ("f1", "f2", "f3", "f4", "f5", "f6"):
            rows.append({"variant_id": "control", "family_id": family, "seed": 1, "score": .5})
            rows.append({"variant_id": "candidate", "family_id": family, "seed": 1, "score": .6})
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete/mismatched paired coverage"):
            audit_ablation_results({"schema_version": "amde-ablation-results-v1",
                                    "control_variant_id": "control", "results": rows},
                                   bootstrap_repetitions=1000)


if __name__ == "__main__":
    unittest.main()
