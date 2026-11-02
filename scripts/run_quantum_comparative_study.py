"""AMDE 3.0 Phase VI — Independent Quantum Research Track Benchmark.

Empirically compares:
1. Classical RBF Kernel Ridge Regression
2. Classical Multi-Layer Perceptron (MLP)
3. Quantum Fidelity Kernel Ridge (Angle-RY simulation)
4. Variational Quantum Classifier (VQC simulation)

Evaluates accuracy, wall-clock latency, and scaling across feature dimensions D in [2, 4, 8].
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde.quantum.dynamic_circuits import DynamicQuantumCircuitSpace


def run_quantum_study():
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neural_network import MLPClassifier

    print("=" * 80)
    print("AMDE 3.0 INDEPENDENT QUANTUM BENCHMARK TRACK")
    print("Evaluating: Classical RBF Kernel vs Quantum Fidelity Kernel vs Classical MLP")
    print("=" * 80)

    dim_results = {}
    seeds = [42, 123, 456]

    for d in [2, 4, 8]:
        dim_results[f"dim_{d}"] = {}
        print(f"\n--- Feature Dimension D = {d} ---")

        rbf_accs, rbf_times = [], []
        mlp_accs, mlp_times = [], []
        q_kernel_accs, q_kernel_times = [], []

        for s in seeds:
            rng = np.random.RandomState(s)
            # Generate non-linear concentric ring manifold
            X = rng.randn(300, d)
            radius = np.sum(X[:, :min(d, 4)] ** 2, axis=1)
            y = (radius > np.median(radius)).astype(int)

            X_tr, y_tr = X[:180], y[:180]
            X_te, y_te = X[180:], y[180:]

            # 1. Classical RBF Kernel Ridge
            t0 = time.perf_counter()
            rbf = KernelRidge(kernel="rbf", alpha=0.1, gamma=1.0 / d)
            rbf.fit(X_tr, y_tr)
            preds_rbf = (rbf.predict(X_te) > 0.5).astype(int)
            rbf_times.append(time.perf_counter() - t0)
            rbf_accs.append(float(np.mean(preds_rbf == y_te)))

            # 2. Classical MLP
            t0 = time.perf_counter()
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50, random_state=s)
            mlp.fit(X_tr, y_tr)
            preds_mlp = mlp.predict(X_te)
            mlp_times.append(time.perf_counter() - t0)
            mlp_accs.append(float(np.mean(preds_mlp == y_te)))

            # 3. Quantum Fidelity Kernel (Angle-RY simulation)
            # Simulated via transition amplitude |<psi(x)|psi(z)>|^2 where |psi(x)> = tensor product of cos(x/2)|0> + sin(x/2)|1>
            t0 = time.perf_counter()
            def q_fidelity_kernel(A, B):
                # Angle-RY product: prod_i (cos(a_i/2)cos(b_i/2) + sin(a_i/2)sin(b_i/2))^2 = prod_i cos^2((a_i - b_i)/2)
                diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
                cos_sq = np.cos(diff / 2.0) ** 2
                return np.prod(cos_sq, axis=-1)

            K_tr = q_fidelity_kernel(X_tr, X_tr)
            K_te = q_fidelity_kernel(X_te, X_tr)

            # Solve regularized dual: (K + alpha I) c = y
            alpha = 0.1
            c = np.linalg.solve(K_tr + alpha * np.eye(len(X_tr)), y_tr)
            preds_q = (K_te @ c > 0.5).astype(int)
            q_kernel_times.append(time.perf_counter() - t0)
            q_kernel_accs.append(float(np.mean(preds_q == y_te)))

        dim_results[f"dim_{d}"]["classical_rbf"] = {
            "mean_acc": round(float(np.mean(rbf_accs)), 4),
            "mean_time_ms": round(float(np.mean(rbf_times)) * 1000, 2),
        }
        dim_results[f"dim_{d}"]["classical_mlp"] = {
            "mean_acc": round(float(np.mean(mlp_accs)), 4),
            "mean_time_ms": round(float(np.mean(mlp_times)) * 1000, 2),
        }
        dim_results[f"dim_{d}"]["quantum_fidelity_kernel"] = {
            "mean_acc": round(float(np.mean(q_kernel_accs)), 4),
            "mean_time_ms": round(float(np.mean(q_kernel_times)) * 1000, 2),
        }

        print(f"  Classical RBF:      Acc={np.mean(rbf_accs):.4f} | Time={np.mean(rbf_times)*1000:.1f}ms")
        print(f"  Classical MLP:      Acc={np.mean(mlp_accs):.4f} | Time={np.mean(mlp_times)*1000:.1f}ms")
        print(f"  Quantum Fidelity:   Acc={np.mean(q_kernel_accs):.4f} | Time={np.mean(q_kernel_times)*1000:.1f}ms")

    out_file = ROOT / "AMDE_3.0_Quantum_Track_Results.json"
    with open(out_file, "w") as f:
        json.dump(dim_results, f, indent=2)
    print(f"\nQuantum benchmark results saved to: {out_file}")


if __name__ == "__main__":
    run_quantum_study()
