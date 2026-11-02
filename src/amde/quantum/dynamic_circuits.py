"""AMDE 3.0 Dynamic Parameterized Quantum Circuit Synthesis Engine.

Fixes BOT-04 (Quantum Configuration Freeze):
- Replaces static preset slicing with dynamic variational circuit synthesis.
- Explores circuit depths, rotation gate topologies, and entanglement structures.
- Generates unique cryptographic circuit hashes for every mutation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Dict, List, Optional


class DynamicQuantumCircuitSpace:
    """Dynamic search space for parameterized quantum circuits (PQC) and kernels."""

    QUBIT_OPTIONS = [2, 4, 6, 8]
    FEATURE_MAPS = ["angle_ry", "angle_ry_rz", "iqp_entangled"]
    ENTANGLEMENTS = ["linear", "ring", "all_to_all"]
    ROTATION_GATES = ["ry", "rx_ry", "u3"]

    def __init__(self, max_qubits: int = 8):
        self.max_qubits = max_qubits

    def sample_circuit(self, rng: random.Random, n_features: int = 4) -> dict[str, Any]:
        """Synthesizes a new dynamic variational quantum circuit configuration."""
        qubits = min(self.max_qubits, max(2, min(n_features, 8)))
        depth = rng.choice([1, 2, 3, 4])
        config = {
            "family": "quantum_vqc",
            "n_qubits": qubits,
            "depth": depth,
            "feature_map": rng.choice(self.FEATURE_MAPS),
            "entanglement": rng.choice(self.ENTANGLEMENTS),
            "rotation_gate": rng.choice(self.ROTATION_GATES),
            "learning_rate": round(rng.choice([0.005, 0.01, 0.02, 0.04]), 4),
            "batch_size": rng.choice([16, 32, 64]),
        }
        config["circuit_hash"] = self.compute_hash(config)
        return config

    def mutate_circuit(self, config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        """Mutates quantum circuit architecture and optimization parameters dynamically."""
        mutated = copy.deepcopy(config)
        mut_type = rng.choice(["depth", "entanglement", "lr", "feature_map"])

        if mut_type == "depth":
            mutated["depth"] = max(1, min(4, mutated["depth"] + rng.choice([-1, 1])))
        elif mut_type == "entanglement":
            mutated["entanglement"] = rng.choice(self.ENTANGLEMENTS)
        elif mut_type == "feature_map":
            mutated["feature_map"] = rng.choice(self.FEATURE_MAPS)
        else:
            mutated["learning_rate"] = round(max(0.001, min(0.1, mutated["learning_rate"] * rng.choice([0.5, 0.8, 1.25, 2.0]))), 4)

        mutated["circuit_hash"] = self.compute_hash(mutated)
        return mutated

    @staticmethod
    def compute_hash(config: dict[str, Any]) -> str:
        clean = {k: v for k, v in config.items() if k != "circuit_hash"}
        canon = json.dumps(clean, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]
