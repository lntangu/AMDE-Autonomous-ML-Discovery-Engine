# AMDE 4.0.1 — Frequently Asked Questions (FAQ)

---

### Q1: What is AMDE in simple terms?
**A:** AMDE (Autonomous Machine Learning Discovery Engine) is an open-source, evidence-oriented research framework designed to study how machine learning pipelines can be searched, evaluated, and audited under strict data isolation, resource budgets, and multi-objective trade-offs.

---

### Q2: How is AMDE different from Optuna, Auto-Sklearn, or FLAML?
**A:** Mature frameworks like Optuna and FLAML are engineered for fast, production hyperparameter optimization. In contrast, AMDE is built as an **auditable research workbench** focused on:
1. **Cryptographic Provenance:** Append-only SQLite event ledger with SHA-256 state chaining.
2. **Evaluation Identity Decoupling:** Caches and rungs strictly separate the candidate's structural genome from its evaluation context.
3. **Multi-Objective Trade-Offs:** Exact and bounded Pareto archives that track inference latency and memory alongside predictive score.
4. **Data Isolation Invariants:** Sealed challenge boundaries that prevent accidental leakage during search iterations.

---

### Q3: Does AMDE beat Random Search or Bayesian Optimization?
**A:** **No statistically significant superiority has been detected.** In our 90-run confirmatory benchmark across 6 datasets and 3 random seeds (50 trials per run), AMDE 4.0 achieved comparable accuracy ($0.9871$ vs $0.9832$ and $0.9809$), but paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction yielded $p \ge 0.85$. AMDE is valuable for its execution reliability and auditable contracts, not for claims of algorithmic dominance.

---

### Q4: How does AMDE prevent data leakage?
**A:** AMDE enforces data isolation across two distinct layers:
1. **Evaluation Pipeline:** Preprocessors (scalers, imputers, feature selectors) are strictly fit on the training split before transforming the validation split. Fitting across the joint dataset is forbidden by contract.
2. **Sealed Holdout Gate:** The held-out test split is locked behind a single-use authorization token (`open_once`). Candidates evaluated during the search phase cannot query or introspect this data until search terminates.

---

### Q5: Is the sealed holdout an OS-level sandbox?
**A:** **No.** It is an in-process Python architectural contract designed to prevent accidental leakage in search loops. It does not provide OS sandboxing or hardware memory encryption (SGX/SEV).

---

### Q6: What does "tamper-evident" ledger mean?
**A:** AMDE logs every candidate evaluation, promotion, and checkpoint into an SQLite database where each row's hash incorporates the previous row's hash:
$$\text{event\_hash} = \text{SHA256}(\text{seq} \parallel \text{prev\_hash} \parallel \text{event\_id} \parallel \text{experiment\_id} \parallel \text{event\_type} \parallel \text{payload} \parallel \text{timestamp})$$
If any row is altered, deleted, or reordered, calling `ledger.verify_chain()` immediately flags the discrepancy. However, it is not "tamper-proof" because it lacks external cryptographic timestamps or digital signatures.

---

### Q7: Why does the Wilcoxon test report $N=18, W=17$, but $p=0.5139$?
**A:** In 9 of the 18 paired runs, AMDE and the baseline produced identical test scores ($\Delta = 0$). Under standard Wilcoxon signed-rank conventions (`zero_method='wilcox'`), zero differences are omitted. This leaves an effective sample size of $N_{\text{effective}} = 9$, where the maximum possible rank sum is 45 and the expected mean under the null hypothesis is $22.5$. The observed statistic of $17.0$ is close to the expected mean, yielding an exact non-significant $p$-value of $0.5139$.

---

### Q8: What modalities does AMDE support?
**A:** AMDE currently supports **classical tabular classification and regression** (tree ensembles, multi-layer perceptrons, linear models, and kernel ridge regression). Computer vision, natural language processing, and audio modalities are not supported.

---

### Q9: How can I run the quickstart?
**A:** Clone the repository, create a virtual environment, and run:
```bash
pip install -e .[dev,tabular]
python examples/quickstart_cpu.py
```
The quickstart completes in ~5 to 10 seconds on a standard CPU. See [REPRODUCE_IN_5_MINUTES.md](REPRODUCE_IN_5_MINUTES.md) for full instructions.

---

### Q10: How can I review or challenge a claim?
**A:** We actively welcome independent technical review! Please consult [INDEPENDENT_REVIEW_GUIDE.md](INDEPENDENT_REVIEW_GUIDE.md) and open an issue using the [Algorithm Correctness Review](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/issues/new?template=algorithm_correctness_review.md) or [Statistical Concern](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/issues/new?template=statistical_concern.md) template.
