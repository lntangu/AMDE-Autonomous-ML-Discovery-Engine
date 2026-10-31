from __future__ import annotations

import math
import unittest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from amde.quantum.profiles import (
    AdaptiveComputeGovernor,
    ComputeProfile,
    GovernorLimits,
    PROFILE_BASELINE,
    PROFILE_MEDIUM,
    PROFILE_HIGH,
    PROFILE_CATALOG,
)
from amde.quantum.statevector import (
    _CNOT_CACHE,
    _SIGNS_CACHE,
    apply_cnot,
    expectation_z,
    initial_state,
)
from amde.quantum.vqc import VQCConfig, train_vqc


@unittest.skipIf(torch is None, "PyTorch optional dependency is unavailable")
class QuantumProfilesAndGovernorTests(unittest.TestCase):
    def test_cnot_and_signs_caching(self):
        state = initial_state(3, device="cpu")
        _CNOT_CACHE.clear()
        _SIGNS_CACHE.clear()

        # First call populates cache
        out1 = apply_cnot(state, 3, 0, 1)
        self.assertIn((3, 0, 1, str(state.device)), _CNOT_CACHE)

        # Second call reuses cached permutation
        out2 = apply_cnot(state, 3, 0, 1)
        self.assertTrue(torch.equal(out1, out2))

        # Test expectation_z caching
        z1 = expectation_z(state, 3, 0)
        self.assertIn((3, 0, str(state.device)), _SIGNS_CACHE)
        z2 = expectation_z(state, 3, 0)
        self.assertTrue(torch.equal(z1, z2))

    def test_compute_profiles_provenance_and_immutability(self):
        self.assertEqual(len(PROFILE_CATALOG), 3)
        self.assertIn("BASELINE", PROFILE_CATALOG)
        self.assertIn("MEDIUM", PROFILE_CATALOG)
        self.assertIn("HIGH", PROFILE_CATALOG)

        base_dict = PROFILE_BASELINE.to_dict()
        med_dict = PROFILE_MEDIUM.to_dict()
        high_dict = PROFILE_HIGH.to_dict()

        # Protocols and hashes must be distinct to prevent false scientific equivalence
        self.assertNotEqual(base_dict["protocol_id"], med_dict["protocol_id"])
        self.assertNotEqual(med_dict["protocol_id"], high_dict["protocol_id"])
        self.assertNotEqual(base_dict["profile_hash"], med_dict["profile_hash"])

        self.assertEqual(base_dict["batch_size"], 16)
        self.assertEqual(med_dict["batch_size"], 64)
        self.assertEqual(high_dict["batch_size"], 128)

    def test_adaptive_governor_hysteresis(self):
        limits = GovernorLimits(
            max_gpu_temperature_c=75.0,
            temp_hysteresis_c=5.0,
            min_free_vram_mb=1000.0,
            vram_hysteresis_mb=500.0,
        )
        gov = AdaptiveComputeGovernor(limits=limits)

        # 1. Normal state -> selects HIGH profile
        prof = gov.select_adaptive_profile(gpu_temp_c=60.0, free_vram_mb=3500.0)
        self.assertEqual(prof.name, "HIGH")
        self.assertFalse(gov.is_throttled)

        # 2. Overheating -> triggers throttle, falls back to BASELINE
        prof_hot = gov.select_adaptive_profile(gpu_temp_c=76.0, free_vram_mb=3500.0)
        self.assertTrue(gov.is_throttled)
        self.assertEqual(prof_hot.name, "BASELINE")
        self.assertIn("GPU temperature", str(gov.throttle_reason))

        # 3. Intermediate temp (72.0C): inside hysteresis band (75 - 5 = 70C)
        # Governor must REMAIN throttled to prevent oscillation
        prof_hysteresis = gov.select_adaptive_profile(gpu_temp_c=72.0, free_vram_mb=3500.0)
        self.assertTrue(gov.is_throttled)
        self.assertEqual(prof_hysteresis.name, "BASELINE")

        # 4. Temperature drops below 70C -> recovers
        prof_cool = gov.select_adaptive_profile(gpu_temp_c=68.0, free_vram_mb=3500.0)
        self.assertFalse(gov.is_throttled)
        self.assertEqual(prof_cool.name, "HIGH")

    def test_vqc_training_optimizations_preserve_metrics(self):
        cfg = VQCConfig(n_qubits=4, n_layers=1, epochs=3, batch_size=8)
        tx = [[0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.1, 0.3], [-0.4, 0.1, 0.2, 0.5], [0.2, -0.3, 0.1, -0.1]]
        ty = [0.0, 1.0, 1.0, 0.0]
        res = train_vqc(tx, ty, tx[:2], ty[:2], tx[:2], ty[:2], config=cfg, device="cpu", seed=42)

        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["loss_history"]), 3)
        self.assertTrue(math.isfinite(res["val_balanced_accuracy"]))
        self.assertGreater(res["val_balanced_accuracy"], 0.0)
