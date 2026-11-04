# AMDE 4.0 — STATISTICAL SIGNIFICANCE & SCIENTIFIC LIMITATIONS
**Mandatory Scientific Disclosure Document**  
**Distribution Version:** 4.0.2  

---

## 1. Executive Statistical Disclosure

> **No statistically significant superiority was detected under the tested conditions.**

In accordance with strict scientific integrity principles and preregistered hypothesis testing protocols, AMDE does not claim AutoML superiority over standard baseline search algorithms under the evaluated conditions.

---

## 2. Hypothesis Testing Methodology

Paired two-sided Wilcoxon signed-rank non-parametric tests were performed comparing AMDE 4.0 (EGADA) against each baseline across 50 paired task-seed evaluations. Multiplicity was controlled using the Holm-Bonferroni step-down correction at the family-wise error rate $\alpha = 0.05$.

### Test Results

| Comparison | Number of Pairs | Wilcoxon $W$ | Raw $p$-value | Holm-Bonferroni $p$-value | Significant ($\alpha = 0.05$)? |
|---|---|---|---|---|---|
| AMDE 4.0 vs AMDE 3.0-Lite | 50 | 123.5 | 0.2936 | 1.0000 | **No** |
| AMDE 4.0 vs AMDE 2.0.3 Baseline | 50 | 150.0 | 0.5162 | 1.0000 | **No** |
| AMDE 4.0 vs Bayesian TPE | 50 | 154.0 | 0.8188 | 1.0000 | **No** |
| AMDE 4.0 vs Random Search | 50 | 154.0 | 0.8188 | 0.8188 | **No** |

---

## 3. Methodological Context & Scope

1. **Evaluation Budget Scope:** All benchmark runs were constrained to an evaluation budget of 50 candidates per task. Under this budget, simple random search and TPE sample the parameter distribution effectively on small-to-medium tabular datasets.
2. **Search Space Density:** When the search space consists of standard scikit-learn estimators (RandomForest, GradientBoosting, LogisticRegression, MLP, SVM), well-tuned default hyperparameters frequently lie near optimal empirical validation performance.
3. **No Free Lunch Theorem:** In finite discrete spaces without strong problem-specific structural priors, no general search meta-policy can uniformly outperform stochastic random sampling across all data distributions.
