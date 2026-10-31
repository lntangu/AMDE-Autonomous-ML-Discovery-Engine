> **Release qualification note (0.2.3):** this report contains the previous 0.2.1 campaign's reported figures. They were not re-run as part of the 0.2.3 hardening pass and must be treated as historical development-set evidence, not an independently replicated claim of generalization. The reported N=3 seeds are exploratory only. No new-algorithm discovery, OOD generalization, CUDA result, ARC score, or 24-hour soak is established by this report.

# BENCHMARK & CONTROLLED EXPERIMENT REPORT — AMDE 0.2.1

**Ngày thực hiện:** 2026-10-29  
**Phiên bản Engine:** `0.2.1`  
**Protocol Hash cố định:** `94174f7ea5f98383532e69cee155794aaa58426c23a02e66531c4c395778ff00`  
**Data Manifest Hash cố định:** `0144b20565b6f5dd096e43c4a7845554bbf17233fe1f335d2c679cffa09c10b3`  
**Split Manifest Hash cố định:** `d7dbdfd471c32265c10b2871968dc0648e53e9bd948da8fead65448a1f059ede`  
**Metric Version & Hash:** `metric-v1` (`ddc20478ef0a08c7fa1c5c87704de7984ebe4c175e08cbb0c911c0ab8a3f1340`)  
**Search RNG Protocol:** `amde-search-rng-v2-seed-only`

## 1. Cấu trúc bộ bài toán và quy tắc tách dữ liệu

Hệ thống đánh giá trên **9 họ bài toán tổng hợp có tính tất định** (`dataset_version = "generated-v1"`), chia thành 2 nhóm:
- **Classical suite (7 nhiệm vụ):** `tab_linear` (regression), `tab_quadratic` (regression), `tab_xor` (classification), `tab_blobs` (classification), `tab_noisy_linear` (classification), `tab_circles` (classification), `tab_imbalanced` (classification).
- **Compositional suite (2 nhiệm vụ có dịch chuyển phân phối ở tập challenge):** `comp_regime` (`regime_switch_regression`, đảo quy tắc chuyển chế độ ở tập challenge) và `comp_shifted_xor` (`shifted_xor_classification`, dịch ngưỡng phân lớp ở tập challenge).

## 2. Kết quả Baseline bất biến (Open Development Suite)

| Baseline | Mô tả | Điểm trung bình Dev (`mean`) | Họ nhiệm vụ kém nhất (`worst_family`) | Điểm họ kém nhất | Trạng thái |
|---|---|---:|---|---:|---|
| `majority_mean` | Dự đoán lớp đa số (classification) hoặc trung bình mẫu `train` (regression) | `0.333333` | `linear` | `0.000000` | `ok` |
| `linear_sgd` | Mô hình tuyến tính huấn luyện bằng SGD chuẩn (Level A mặc định) | `0.378667` | `regime_switch` | `0.000000` | `ok` |
| `mlp_small` | Mạng MLP 1 tầng ẩn (`hidden=[8]`, `tanh`, optimizer `adam`, `lr=0.01`, `epochs=35`) | **`0.445461`** | `regime_switch` | `0.000000` | `ok` (Strongest Baseline) |

## 3. Thí nghiệm đối chứng 3 chiến lược × 3 hạt giống × 3 thế hệ (`reports/control_experiment_3seeds_3gens.json`)

- **Lệnh chạy:** `PYTHONPATH=src python3 scripts/control_experiment.py --generations 3 --seeds 11 23 37 --out reports/control_experiment_3seeds_3gens.json`
- **Ngân sách cố định cho mỗi lần chạy:** 3 thế hệ × (6 quick proposals + 2 full confirmations) = **24 candidate trials/run** (+ 3 baseline evaluations). Tổng cộng 9 runs = **216 candidate trials** (`101.845 giây`).

### 3.1. Bảng kết quả chi tiết theo từng Seed (`N = 3 seeds`)

| Seed | Chiến lược (`variant`) | Điểm Dev tốt nhất (`best_score`) | Chênh lệch so với `mlp_small` (`0.445461`) | Cấp độ ứng viên tốt nhất (`best_level`) | Candidate Hash (12 ký tự đầu) | Thời gian chạy (s) |
|---:|---|---:|---:|:---:|---|---:|
| **11** | `portfolio` | `0.531633` | `+0.086172` | `A` | `039624fb3bc9` | `12.08` |
| **11** | `random_only` | `0.531633` | `+0.086172` | `A` | `039624fb3bc9` | `11.87` |
| **11** | `hpo_A` | **`0.532293`** | `+0.086832` | `A` | `c23c154bb263` | `10.54` |
| **23** | `portfolio` | **`0.579078`** | `+0.133617` | `C` | `82d6ca2c371e` | `10.03` |
| **23** | `random_only` | `0.555576` | `+0.110114` | `B` | `ae81529f90df` | `10.52` |
| **23** | `hpo_A` | `0.503789` | `+0.058328` | `A` | `f69f2fc95250` | `11.55` |
| **37** | `portfolio` | **`0.540533`** | `+0.095072` | `B` | `f59fc89b1e60` | `12.24` |
| **37** | `random_only` | **`0.540533`** | `+0.095072` | `B` | `f59fc89b1e60` | `11.56` |
| **37** | `hpo_A` | `0.536690` | `+0.091229` | `A` | `e17fd5d93787` | `11.45` |

### 3.2. Thống kê tổng hợp và so sánh ghép cặp (Paired Comparisons)

| Chiến lược (`variant`) | Số lần chạy thành công | Điểm trung bình (`mean_best_score`) | Độ lệch chuẩn mẫu (`sample_sd`) | So sánh ghép cặp (`portfolio - variant`) | Thắng / Hòa / Thua (trên 3 seeds) |
|---|---:|---:|---:|---:|:---:|
| **`portfolio`** (5 chiến lược, Levels A–D) | `3 / 3` | **`0.550415`** | `0.025219` | — | — |
| **`random_only`** (Random search, Levels A–D) | `3 / 3` | `0.542581` | `0.012102` | `+0.007834` | `1 thắng / 2 hòa / 0 thua` |
| **`hpo_A`** (Random HPO chỉ ở Level A) | `3 / 3` | `0.524257` | `0.017862` | `+0.026157` | `2 thắng / 0 hòa / 1 thua` |

## 4. Kết quả kiểm tra trên tập Sealed Challenge (5-Generation Run)

| Đối tượng | Điểm trung bình Dev Confirm | Điểm trung bình Sealed Challenge | Khoảng tin cậy 80% trên Challenge (`[ci_low, ci_high]`) | Độ lệch chuẩn giữa các Seed (`std_seeds`) |
|---|---:|---:|---:|---:|
| **Ứng viên Incumbent (`c774117484a2`, Level B)** | `0.531321` | **`0.467948`** | `[0.432972, 0.502924]` | `0.047271` |
| **Baseline mạnh nhất (`mlp_small`)** | `0.445461` | `0.439493` | `[0.406027, 0.472959]` | `0.045230` |
| **Chênh lệch (`Candidate - Baseline`)** | `+0.085860` | **`+0.028455`** | — | — |

## 5. Giới hạn diễn giải khoa học

- Cỡ mẫu `N = 3 seeds` còn nhỏ và sử dụng chung tập `dev` cho chọn lọc; kết quả trên là bằng chứng kỹ thuật về hành vi tìm kiếm dưới ngân sách khớp nhau, **không phải** bằng chứng tổng quát hóa sang họ nhiệm vụ hoàn toàn mới.
- Các ứng viên dẫn đầu thuộc Level A, B, C (tinh chỉnh siêu tham số, chuẩn hóa, chọn optimizer Adam và kiến trúc MLP), không phải phát minh ra thuật toán học hoàn toàn mới ở Level D.
