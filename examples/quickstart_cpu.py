"""AMDE 4.0.1 Quickstart CPU Example.

A minimal, deterministic, and self-contained script demonstrating:
1. Bounded ML candidate generation over a typed hierarchical search space.
2. In-process pipeline evaluation across stratified train/validation folds.
3. Cryptographic tamper-evident provenance logging with EventLedger.
4. Multi-objective trade-off tracking via ParetoArchive.

Run time: ~5-10 seconds on a standard modern CPU.
No GPU, external network access, or credentials required.
"""

from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

# Ensure worktree src takes precedence
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amde.archive.pareto import ParetoArchive
from amde.core.contracts import (
    Candidate,
    EvaluationRequest,
    FidelityLevel,
    ResourceBudget,
)
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.ledger.event_ledger import EventLedger
from amde.space.search_space import SearchSpace
from amde.strategy.tpe import TPEStrategy


def generate_synthetic_tabular_data(n_samples: int = 150, n_features: int = 4, seed: int = 42):
    """Generates a small deterministic binary classification problem."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Non-linear decision boundary
    y = ((X[:, 0] * X[:, 1] + X[:, 2] ** 2) > 0.5).astype(int)
    split = int(0.7 * n_samples)
    return (X[:split], y[:split]), (X[split:], y[split:])


def main() -> int:
    print("=" * 70)
    print("  AMDE 4.0.1 — 60-Second CPU Discovery & Verification Quickstart")
    print("=" * 70)

    start_time = time.monotonic()
    rng = random.Random(42)

    # 1. Dataset & Search Space Definition
    print("\n[1/5] Initializing synthetic classification task & search space...")
    (X_train, y_train), (X_val, y_val) = generate_synthetic_tabular_data()
    space = SearchSpace(n_features=4, is_classification=True)
    print(f"      Train samples: {len(X_train)}, Val samples: {len(X_val)}, Features: 4")
    print(f"      Model families supported: {list(space.MODEL_FAMILIES)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        ledger_path = Path(temp_dir) / "quickstart_ledger.sqlite3"

        # 2. Setup Provenance Ledger and Multi-Objective Pareto Archive
        print("\n[2/5] Initializing tamper-evident EventLedger and ParetoArchive...")
        ledger = EventLedger(ledger_path)
        pareto = ParetoArchive(capacity=10)

        # 3. Candidate Generation with Factorized TPE
        print("\n[3/5] Proposing candidates with factorized Parzen estimator (TPE)...")
        strategy = TPEStrategy(space, gamma_quantile=0.3, min_observations=2)
        candidates: list[Candidate] = strategy.propose(n=4, rng=rng, archives={})
        print(f"      Generated {len(candidates)} candidate configurations.")

        # 4. Evaluation via Standard Pipeline
        print("\n[4/5] Evaluating candidates under strict resource budget...")
        evaluator = StandardEvaluationPipeline()
        budget = ResourceBudget(max_wallclock_seconds=5.0, max_memory_mb=512.0)
        fidelity = FidelityLevel.full_fidelity()

        for idx, candidate in enumerate(candidates, start=1):
            request = EvaluationRequest(
                request_id=f"req_{candidate.canonical_hash[:8]}_{idx}",
                candidate=candidate,
                dataset_id="synthetic_tabular_binary",
                split_hash="deterministic_70_30_split",
                seed=42 + idx,
                fidelity=fidelity,
                resource_budget=budget,
            )
            result = evaluator.execute(request, X_train, y_train, X_val, y_val, isolated=False)

            # Record feedback to drive subsequent TPE proposal updates
            strategy.record_observation(candidate, result.primary_metric)

            # Log to cryptographic ledger
            ledger.append_event(
                event_id=f"eval_{candidate.canonical_hash[:8]}_{idx}",
                experiment_id="quickstart_exp_01",
                event_type="candidate_evaluation",
                payload={
                    "candidate_hash": candidate.canonical_hash,
                    "fidelity": result.fidelity.name,
                    "primary_metric": result.primary_metric,
                    "train_duration_s": result.train_duration_seconds,
                    "inference_latency_ms": result.inference_latency_ms,
                    "status": result.status,
                    "model_family": candidate.genome.get("model", {}).get("family", "linear"),
                },
            )

            # Insert into multi-objective Pareto archive (loss = -metric, latency_ms, memory_mb)
            is_new_or_dominant = pareto.update(
                candidate=candidate,
                loss=-float(result.primary_metric),
                latency=float(result.inference_latency_ms),
                memory=float(result.peak_memory_mb),
            )

            status_str = "dominant" if is_new_or_dominant else "dominated"
            print(f"      Candidate {idx} [{candidate.canonical_hash[:8]}]: "
                  f"Score={result.primary_metric:.4f}, Latency={result.inference_latency_ms:.2f}ms, "
                  f"Model={candidate.genome.get('model', {}).get('family', 'linear')} ({status_str})")

        # 5. Cryptographic Chain Verification
        print("\n[5/5] Verifying ledger provenance & Pareto frontier...")
        is_chain_valid, chain_msg = ledger.verify_chain()
        ledger_count = len(ledger.get_events("quickstart_exp_01"))
        frontier_size = len(pareto.frontier)

        print(f"      Ledger records: {ledger_count} events | SHA-256 chain: {chain_msg}")
        print(f"      Pareto frontier: {frontier_size} non-dominated configurations preserved")

        assert is_chain_valid, f"Cryptographic hash chain verification failed: {chain_msg}"
        assert ledger_count == len(candidates), "Mismatch between evaluations and ledger entries!"

    elapsed = time.monotonic() - start_time
    print("\n" + "=" * 70)
    print(f"  SUCCESS: Quickstart finished in {elapsed:.2f} seconds.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
