from __future__ import annotations

import math
import unittest

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from amde.quantum.maze import bfs_shortest_path, run_maze_grover_demo
from amde.quantum.statevector import (
    QuantumBackendError,
    StatevectorBudget,
    apply_cnot,
    apply_h,
    apply_ry,
    apply_rz,
    expectation_z,
    grover_search_statevector,
    initial_state,
)
from amde.quantum.vqc import VQCConfig, VariationalQuantumClassifier, train_vqc
from amde.quantum.kernel import train_quantum_kernel, train_rbf_kernel_baseline


@unittest.skipIf(torch is None, "PyTorch optional dependency is unavailable")
class StatevectorTests(unittest.TestCase):
    def test_single_qubit_gates_preserve_norm(self):
        state = initial_state(3, device="cpu")
        state = apply_h(state, 3, 0)
        state = apply_ry(state, 3, 1, 0.71)
        state = apply_rz(state, 3, 2, -1.13)
        norm = torch.sum(state.abs().square()).item()
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_bell_pair_expectation_and_cnot(self):
        state = initial_state(2, device="cpu")
        state = apply_h(state, 2, 0)
        state = apply_cnot(state, 2, 0, 1)
        probabilities = state.abs().square().squeeze(0)
        nonzero = {i for i, value in enumerate(probabilities.tolist()) if value > 1e-6}
        self.assertEqual(nonzero, {0, 3})
        self.assertAlmostEqual(float(expectation_z(state, 2, 0).item()), 0.0, places=6)
        state = apply_cnot(state, 2, 0, 1)
        self.assertAlmostEqual(float((state - torch.tensor([[1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0]], dtype=torch.complex64)).abs().max().item()), 0.0, places=5)

    def test_budget_rejects_excessive_allocation(self):
        budget = StatevectorBudget(max_qubits=12, max_statevector_mb=1)
        with self.assertRaises(QuantumBackendError):
            budget.validate(12, batch_size=100)

    def test_grover_amplifies_marked_state(self):
        result = grover_search_statevector(3, [5], device="cpu")
        self.assertGreater(result["marked_probability"], 0.9)
        self.assertEqual(result["oracle_queries"], result["iterations"])
        self.assertTrue(result["top_measurement_is_marked"])
        self.assertIn("statevector", result["simulation_semantics"])

    def test_maze_demo_reports_query_model_limits(self):
        result = run_maze_grover_demo(device="cpu", max_qubits=20, max_statevector_mb=64)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["grover"]["measured_path_is_valid"])
        self.assertFalse(result["grover"]["oracle_is_reversible_gate_circuit"])
        self.assertGreater(result["grover"]["oracle_precompute_seconds"], 0.0)
        self.assertIn("No end-to-end quantum speedup", result["interpretation"])
        self.assertEqual(result["bfs"]["distance"], bfs_shortest_path(result["maze"]).distance)


@unittest.skipIf(torch is None, "PyTorch optional dependency is unavailable")
class VQCTests(unittest.TestCase):
    def test_parameter_shift_matches_autograd(self):
        model = VariationalQuantumClassifier(VQCConfig(n_qubits=3, n_layers=1), device="cpu", seed=9)
        x = torch.tensor([[0.1, -0.4, 0.2], [0.7, 0.3, -0.2], [-0.5, 0.2, 0.9]], dtype=torch.float32)
        y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(x), y)
        auto = torch.autograd.grad(loss, model.theta)[0]
        shifted, _ = model.parameter_shift_grad(x, y)
        self.assertLess(float((auto - shifted).abs().max().item()), 2e-5)

    def test_vqc_trains_and_returns_finite_metrics(self):
        x = [[-1.0, -0.8], [-0.8, -0.9], [-0.5, -0.7], [0.8, 0.9], [1.0, 0.7], [0.7, 0.6],
             [-0.9, 0.2], [0.9, -0.2], [-0.7, 0.1], [0.7, -0.1], [-0.2, -0.9], [0.2, 0.9]]
        y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        result = train_vqc(
            x, y, x, y, x, y,
            config=VQCConfig(n_qubits=2, n_layers=1, epochs=2, batch_size=4),
            device="cpu", seed=7,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["device"], "cpu")
        self.assertTrue(math.isfinite(result["test"]["balanced_accuracy"]))
        self.assertFalse(result["physical_qpu"])

    def test_balanced_accuracy_uses_positive_and_negative_class_denominators(self):
        from amde.quantum.vqc import _balanced_accuracy
        logits = torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=torch.float32)
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        # Negative recall = 1.0; positive recall = 0.5; balanced accuracy = 0.75.
        self.assertAlmostEqual(_balanced_accuracy(logits, labels), 0.75, places=6)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            VQCConfig(n_qubits=1).validate()
        with self.assertRaises(ValueError):
            VQCConfig(n_qubits=4, optimizer="unknown").validate()


@unittest.skipIf(torch is None, "PyTorch optional dependency is unavailable")
class QuantumKernelTests(unittest.TestCase):
    def test_quantum_and_rbf_kernel_baselines_are_finite(self):
        x = [[-1.0, -0.8], [-0.8, -0.9], [-0.5, -0.7], [0.8, 0.9], [1.0, 0.7], [0.7, 0.6],
             [-0.9, 0.2], [0.9, -0.2], [-0.7, 0.1], [0.7, -0.1], [-0.2, -0.9], [0.2, 0.9]]
        y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        quantum = train_quantum_kernel(x, y, x, y, x, y, n_qubits=4, device="cpu")
        classical = train_rbf_kernel_baseline(x, y, x, y, x, y)
        self.assertTrue(math.isfinite(quantum["test"]["balanced_accuracy"]))
        self.assertTrue(math.isfinite(classical["test"]["balanced_accuracy"]))
        self.assertEqual(quantum["method"], "quantum_fidelity_kernel_ridge")
        self.assertEqual(classical["method"], "rbf_kernel_ridge_cpu_baseline")


if __name__ == "__main__":
    unittest.main()
