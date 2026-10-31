# AMDE 2.0.1 Scientific Benchmark & Evaluation Protocol

## 1. Scope & Research Objective

AMDE 2.0.1 evaluates machine learning and quantum-inspired algorithmic discoveries across standard benchmark tasks under strict, preregistered protocols.

The goal is to determine whether open-ended, evolutionary search yields statistically significant improvements over established baselines without data leakage, metric overfitting, or false performance claims.

---

## 2. Experimental Rigor & Anti-Leakage Safeguards

### 2.1 Sealed Test Set Policy
1. **Three-Way Data Split**: Every benchmark dataset is partitioned into:
   - **Training Set (60%)**: Used by candidate algorithms for parameter fitting.
   - **Validation Set (20%)**: Used during discovery rounds for Pareto fitness selection.
   - **Sealed Test Set (20%)**: Evaluated strictly once per selected Pareto candidate at round completion. Candidates are never selected or tuned based on the sealed split.
2. **Leakage Prevention**: All preprocessing transformations (e.g., standard scalers, quantile transforms, imputers) are fit strictly on the training fold and applied to validation/test folds without re-estimating parameters.

### 2.2 Baseline Registry
All candidate recipes are compared against canonical baselines configured under identical resource budgets:
- **Classical Tabular Baselines**:
  - Logistic Regression (L2 regularized)
  - Random Forest Classifier (100 estimators, max depth 8)
  - Gradient Boosted Decision Trees (LightGBM / HistGradientBoosting)
  - Multilayer Perceptron (2 hidden layers, ReLU activation)
- **Quantum Machine Learning Baselines**:
  - Angle-encoded Variational Quantum Classifier (VQC) with Hardware-Efficient Ansatz (HEA)
  - Amplitude-encoded Quantum Kernel Estimator (QKM) with Support Vector Machine (QSVM)
  - Classical Simulated Annealing baseline for discrete circuit structure optimization

---

## 3. Evaluation Metrics & Statistical Testing

Candidate models are scored across multiple orthogonal objectives:
1. **Discriminative Quality**:
   - Classification Accuracy
   - Macro-averaged F1 Score ($F_1$)
   - Area Under ROC Curve (ROC-AUC)
2. **Calibration Quality**:
   - Brier Score
   - Expected Calibration Error (ECE)
3. **Efficiency & Complexity**:
   - Inference latency per sample (milliseconds)
   - Train wall-clock time (seconds)
   - Model parameter count / Circuit gate depth
4. **Statistical Significance**:
   - Resampled 10-fold cross-validation.
   - Paired Student's t-test and non-parametric Wilcoxon signed-rank test ($\alpha = 0.05$).
   - Benjamini-Hochberg False Discovery Rate (FDR) correction across multiple comparisons.

---

## 4. Hardware Standard & Reproducibility Constraints

- **Reference Platform**: Intel Core i7-8850H (6 cores / 12 threads), 32 GB RAM, NVIDIA Quadro P2000 (4 GB GDDR5, CUDA 12.6, PyTorch 2.14.0+cu126), Windows 11.
- **RNG Determinism**: Every round records explicit random seeds for NumPy, PyTorch, Python random, and scikit-learn. Resumed runs preserve the exact RNG sequence.
- **Quantum Simulation Protocol**:
  - All quantum computations are executed via statevector matrix simulation in PyTorch CUDA.
  - Statevectors are simulated up to 8 qubits ($2^8 = 256$ complex amplitudes) within bounded VRAM.
  - Results are labeled as *classical statevector simulation*, not physical quantum device execution.
