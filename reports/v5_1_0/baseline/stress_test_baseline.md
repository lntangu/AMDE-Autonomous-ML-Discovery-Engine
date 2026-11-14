# AMDE v5.1.0 12-Case Stress Test Results

- **Total Cases**: 12
- **Passed**: 9/12
- **Failed**: 3/12

| Case ID | Description | Status | Time (s) | Peak RAM (MB) | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `G01` | Standard Heterogeneous (Num + Cat + Missing) | **PASS** | 18.53s | 0.0MB | Clean execution |
| `G02` | High Cardinality 10K (20k rows, 10k categories) | ❌ **FAIL** | 56.15s | 0.0MB | ValidationError: Invalid non-finite float value in canonical data: -inf |
| `G03` | Extreme Class Imbalance (99.5% vs 0.5%) | **PASS** | 8.39s | 0.0MB | Clean execution |
| `G04` | Inf and Float64 Max (+/-inf, 1e308, -1e308) | ❌ **FAIL** | 0.02s | 0.0MB | ValueError: Input X contains infinity or a value too large for dtype('float32'). |
| `G05` | All Missing Columns (100% NaN/None) | **PASS** | 7.73s | 0.0MB | Clean execution |
| `G06` | Tiny Dataset Small N (N=15, 3 classes) | ❌ **FAIL** | 0.0s | 0.0MB | ValueError: The test_size = 2 should be greater or equal to the number of classes = 3 |
| `G07` | High Dimensionality Wide (N=100, D=300) | **PASS** | 17.05s | 0.0MB | Clean execution |
| `G08` | Disjoint Unseen Categories at Inference | **PASS** | 7.37s | 0.0MB | Clean execution |
| `G09` | Sentinel Collision ('__missing__', 'nan', 'null') | **PASS** | 10.49s | 0.0MB | Clean execution |
| `G10` | Zero-Variance Constant Features | **PASS** | 12.35s | 0.0MB | Clean execution |
| `G11` | High Class Count (20 distinct classes) | **PASS** | 13.64s | 0.0MB | Clean execution |
| `G12` | Mixed Nullable Extension Dtypes (Int64, Float64, boolean) | **PASS** | 13.18s | 0.0MB | Clean execution |
