from __future__ import annotations

import unittest

from amde.evaluation.generalization import audit_generalization


class GeneralizationAuditTests(unittest.TestCase):
    def setUp(self):
        self.families = [f"family_{index:02d}" for index in range(12)]
        self.candidate = {family: 0.72 + index * 0.001 for index, family in enumerate(self.families)}
        self.baseline = {family: 0.67 + index * 0.001 for index, family in enumerate(self.families)}
        self.metadata = {
            family: {
                "source": "external",
                "domain": f"domain_{index % 3}",
                "seen_during_search": False,
            }
            for index, family in enumerate(self.families)
        }

    def test_strong_paired_external_holdout_evidence_passes_scoped_gate(self):
        result = audit_generalization(
            self.candidate,
            self.baseline,
            family_metadata=self.metadata,
            preregistered=True,
            evaluation_used_for_selection=False,
            bootstrap_repetitions=2000,
            seed=9,
        )
        self.assertTrue(result["internal_generalization_gate"])
        self.assertTrue(result["external_generalization_gate"])
        self.assertEqual(result["evidence_tier"], "multi_domain_external_holdout_support")
        self.assertGreater(result["bootstrap_ci_95"][0], 0.0)
        self.assertEqual(result["family_wins"], 12)

    def test_selection_contamination_blocks_generalization_gate(self):
        result = audit_generalization(
            self.candidate,
            self.baseline,
            family_metadata=self.metadata,
            preregistered=True,
            evaluation_used_for_selection=True,
            bootstrap_repetitions=1000,
        )
        self.assertFalse(result["internal_generalization_gate"])
        self.assertFalse(result["external_generalization_gate"])

    def test_family_mismatch_is_rejected(self):
        baseline = dict(self.baseline)
        baseline.pop(self.families[-1])
        with self.assertRaises(ValueError):
            audit_generalization(
                self.candidate,
                baseline,
                family_metadata=self.metadata,
            )

    def test_missing_metadata_is_rejected(self):
        metadata = dict(self.metadata)
        metadata.pop(self.families[0])
        with self.assertRaises(ValueError):
            audit_generalization(
                self.candidate,
                self.baseline,
                family_metadata=metadata,
            )


if __name__ == "__main__":
    unittest.main()
