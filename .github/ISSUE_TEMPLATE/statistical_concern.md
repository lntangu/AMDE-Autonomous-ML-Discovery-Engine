---
name: Statistical or Benchmark Concern
description: Report an issue with statistical significance testing, baseline fairness, multiple testing corrections, or benchmark design
labels: ["statistics", "benchmarks"]
---

### Target Benchmark / Comparison
- **Benchmark Run / Task:** (e.g. Confirmatory P0 Benchmark, `random_search` vs `amde_40`)
- **Reported Metric / Test:** (e.g. Wilcoxon signed-rank test, Holm-Bonferroni correction, ANOVA)

### Statistical Concern
Describe the issue (e.g. paired unit mismatch, zero-tie handling, sample size N, power analysis, baseline budget inequality):

### Re-analysis / Calculation Evidence
```python
# Provide Python / R script or raw differences used for re-calculation
```

### Reference Artifact
- **File Path in Repository:** (e.g. `results/amde4_benchmarks/confirmatory_p0/AMDE_4.0_Statistical_Significance.json`)

### Suggested Correction
Suggested statistical methodology, reporting change, or benchmark re-run:
