# AMDE v5.1.0 12-Case Stress Test Results

- **Total Cases**: 12
- **Passed**: 12/12
- **Failed**: 0/12

| Case ID | Description | Status | Time (s) | Peak RAM (MB) | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `G01` | Standard Heterogeneous (Num + Cat + Missing) | **PASS** | 17.14s | 0.0MB | Clean execution |
| `G02` | High Cardinality 10K (20k rows, 10k categories) | **PASS** | 16.02s | 0.0MB | Clean execution |
| `G03` | Extreme Class Imbalance (99.5% vs 0.5%) | **PASS** | 7.27s | 0.0MB | Clean execution |
| `G04` | Inf and Float64 Max (+/-inf, 1e308, -1e308) | **PASS** | 5.49s | 0.0MB | Clean execution |
| `G05` | All Missing Columns (100% NaN/None) | **PASS** | 7.17s | 0.0MB | Clean execution |
| `G06` | Tiny Dataset Small N (N=15, 3 classes) | **PASS** | 13.29s | 0.0MB | Clean execution |
| `G07` | High Dimensionality Wide (N=100, D=300) | **PASS** | 17.76s | 0.0MB | Clean execution |
| `G08` | Disjoint Unseen Categories at Inference | **PASS** | 5.88s | 0.0MB | Clean execution |
| `G09` | Sentinel Collision ('__missing__', 'nan', 'null') | **PASS** | 16.52s | 0.0MB | Clean execution |
| `G10` | Zero-Variance Constant Features | **PASS** | 16.52s | 0.0MB | Clean execution |
| `G11` | High Class Count (20 distinct classes) | **PASS** | 13.49s | 0.0MB | Clean execution |
| `G12` | Mixed Nullable Extension Dtypes (Int64, Float64, boolean) | **PASS** | 11.23s | 0.0MB | Clean execution |
