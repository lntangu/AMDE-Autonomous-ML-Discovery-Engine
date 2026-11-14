# AMDE v5.1.0 Central Defect Register

- **Total Confirmed Defects**: 10
- **Critical (P0)**: 4
- **High (P1)**: 2
- **Medium (P1)**: 4

| ID | Category | Severity | File | Root Cause | Proposed Fix | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DEF-01` | Numerical Integrity | **Critical** | `src/amde/preprocessing/tabular.py` | Uncapped float64 values (1e308, -1e308) overflow into +inf/-inf upon float32 conversion, causing SimpleImputer to fail with ValueError. | Clip extreme float64 values to [-3.4e38, 3.4e38] or convert overflowed values to NaN prior to imputation. | **Open** |
| `DEF-02` | Categorical Architecture | **Critical** | `src/amde/preprocessing/tabular.py, src/amde/engine.py` | Unconstrained OneHotEncoder(sparse_output=False) creates dense matrices (10k features, 2.73 GB RAM). All candidates timeout, leaving incumbent_score as -inf which crashes canonical_json(). | Implement adaptive encoding (Frequency/OOF Target Encoding for cardinality > 50, OneHot for <= 50) and sanitize incumbent score. | **Open** |
| `DEF-03` | AutoML Search Architecture | **Critical** | `src/amde/wrapper.py, src/amde/engine.py` | On large datasets (>50k rows like Covertype), AMDE evaluates slow models on the entire dataset, exhausting time budget on 1-2 candidates. | Implement progressive stratified subsampling (15k coarse, 40k refined, final full refit) and prioritize histogram tree models. | **Open** |
| `DEF-04` | Runtime Governance | **Critical** | `src/amde/wrapper.py, src/amde/engine.py` | Search timeout is checked only at the start of candidate loops. Running candidates overrun budget without interruption. | Implement central monotonic clock, propagate remaining candidate budget, and reserve 15% budget for finalization. | **Open** |
| `DEF-05` | Data Splitting & Edge Cases | **High** | `src/amde/wrapper.py` | train_test_split(..., stratify=y) fails when test_size < n_classes or class frequency < 2. | Add min_class_count guard and fallback to unstratified split when stratification is impossible. | **Open** |
| `DEF-06` | ML Correctness | **High** | `src/amde/wrapper.py, src/amde/evaluation/pipeline.py` | Need strict fold-local transformation in CV and sealed holdout test isolation. | Enforce fold-local preprocessing in internal CV pipelines. | **Open** |
| `DEF-07` | Observability & Code Quality | **Medium** | `src/amde/evaluation/isolation.py, src/amde/runtime/lock.py, src/amde/wrapper.py` | Silent except: pass suppresses errors without logging or diagnostics. | Replace silent pass with structured logger.debug/logger.warning and failure tracking. | **Open** |
| `DEF-08` | Performance & DataFrame Integrity | **Medium** | `src/amde/preprocessing/tabular.py` | Column-by-column assignment in a loop on wide DataFrames fragments memory. | Construct DataFrame in a single batch using dictionary or pd.concat. | **Open** |
| `DEF-09` | Test Reliability | **Medium** | `src/amde/evaluation/isolation.py` | Timeout status mismatch (returned timeout instead of error). | Map timeout to appropriate status field in isolation result. | **Open** |
| `DEF-10` | Test Reliability | **Medium** | `tests/unit/test_campaign.py` | In two-round test, candidate_archive length assertion fails when generations=0. | Ensure candidate archive updates even with zero generations or align assertion. | **Open** |
