# OpenML-CC18 Real-Data Benchmark Revalidation Report

## Executive Summary
- **Datasets Evaluated**: 6/6
- **AMDE Completions**: 6/6
- **AMDE Crashes**: 0/6
- **Acceptance Gate G Status**: PASS

## Evaluated Dataset Metadata
| ID | Name | Samples | Features | Categorical | Numeric | Classes | Missing % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `31` | **credit-g** | 1000 | 20 | 13 | 7 | 2 | 0.0% |
| `37` | **diabetes** | 768 | 8 | 0 | 8 | 2 | 0.0% |
| `38` | **sick** | 3772 | 29 | 22 | 7 | 2 | 5.54% |
| `54` | **vehicle** | 846 | 18 | 0 | 18 | 4 | 0.0% |
| `1464` | **blood-transfusion-service-center** | 748 | 4 | 0 | 4 | 2 | 0.0% |
| `40975` | **car** | 1728 | 6 | 6 | 0 | 4 | 0.0% |

## Benchmark Results (Seed = 42, Budget = 30s per dataset)
| Dataset ID | Dataset Name | System | Status | Weighted F1 | Accuracy | Fit Time (s) | Memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `31` | credit-g | AMDE 5.0 Super Learner | **COMPLETED** | 0.7559 | 0.7650 | 119.92s | 555.91 |
| `31` | credit-g | FLAML v2.8.0 | **COMPLETED** | 0.6862 | 0.7150 | 32.36s | 556.87 |
| `31` | credit-g | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
| `37` | diabetes | AMDE 5.0 Super Learner | **COMPLETED** | 0.7274 | 0.7338 | 30.65s | 557.09 |
| `37` | diabetes | FLAML v2.8.0 | **COMPLETED** | 0.7372 | 0.7468 | 33.0s | 557.45 |
| `37` | diabetes | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
| `38` | sick | AMDE 5.0 Super Learner | **COMPLETED** | 0.9920 | 0.9921 | 35.48s | 568.97 |
| `38` | sick | FLAML v2.8.0 | **COMPLETED** | 0.9786 | 0.9801 | 30.82s | 593.01 |
| `38` | sick | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
| `54` | vehicle | AMDE 5.0 Super Learner | **COMPLETED** | 0.8276 | 0.8294 | 50.76s | 617.49 |
| `54` | vehicle | FLAML v2.8.0 | **COMPLETED** | 0.6653 | 0.6882 | 31.38s | 615.71 |
| `54` | vehicle | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
| `1464` | blood-transfusion-service-center | AMDE 5.0 Super Learner | **COMPLETED** | 0.7642 | 0.7933 | 10.81s | 617.77 |
| `1464` | blood-transfusion-service-center | FLAML v2.8.0 | **COMPLETED** | 0.7384 | 0.7800 | 32.49s | 616.56 |
| `1464` | blood-transfusion-service-center | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
| `40975` | car | AMDE 5.0 Super Learner | **COMPLETED** | 0.9943 | 0.9942 | 34.35s | 619.99 |
| `40975` | car | FLAML v2.8.0 | **COMPLETED** | 0.7908 | 0.8179 | 32.54s | 620.09 |
| `40975` | car | AutoGluon v1.6.3 | *BLOCKED* | 0.0000 | 0.0000 | 0.0s | 0.0 |
