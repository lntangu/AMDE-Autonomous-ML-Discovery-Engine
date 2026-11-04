# AMDE 4.0 — Báo Cáo Audit Hiện Trạng Trước Khi Thực Hiện Demo Thực Tế
**Tài liệu:** `reports/real_world_demo/INITIAL_AUDIT.md`  
**Ngày lập:** 2026-11-03  
**Tác giả:** Principal ML Systems & Reproducibility Engineering Team  
**Mục đích:** Đánh giá hiện trạng toàn diện về Git, kiến trúc, module, benchmark cũ và điều kiện thực nghiệm trước khi thiết kế bài toán ML demo thực tế, bộ benchmark đối chứng và Google Colab demo.

---

## 1. Trạng Thái Thực Tế Của Dự Án

| Thuộc tính | Giá trị được xác minh | Chi tiết & Bằng chứng |
| :--- | :--- | :--- |
| **Repository URL** | `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine` | Remote public, unauthenticated read xác nhận 100%. |
| **Canonical Default Branch** | `release-amde-final` | Nhánh mặc định chính thức trên GitHub. |
| **Commit Baseline trên Remote** | `72be93013907b8296e7918f7c1383cba6bae4cf4` | Merge commit của PR #4 (P0 correctness closure). |
| **Working Tree Branch** | `experiment/real-world-evidence-demo` | Tạo mới từ commit `2629e2e` (PR #5 trên `origin/release-amde-final`). |
| **Release Tag Ổn Định Nhất** | `v4.0.1` (`0a4176f1048cac69d0b150b16dad5ae9b1f6cefe`) | 4 release assets gồm wheel, sdist, closure doc và checksums. |
| **Phiên bản Package** | `4.0.1` (`pyproject.toml`, `src/amde/__init__.py`) | Đã cài đặt editable mode trong clean venv. |
| **Trạng thái CI Matrix** | **100% Green (Workflow Run 41613002049)** | 4/4 jobs PASS (Ubuntu & Windows trên Python 3.10 & 3.14). |

---

## 2. Commit & Release Được Chọn Để Thực Nghiệm

- **Commit được chọn:** `2629e2e` (trên nhánh `experiment/real-world-evidence-demo`).
- **Lý do lựa chọn:**
  1. Commit `2629e2e` kế thừa trực tiếp từ commit `72be930` (merge commit của `fix/amde4-p0-correctness-closure` vào `release-amde-final`).
  2. Chứa đầy đủ tất cả 6 bản sửa lỗi P0 cốt lõi đã được kiểm định:
     - Tích lũy lịch sử quan sát TPE (`record_feedback` $\rightarrow$ `record_observation`).
     - Ánh xạ kiểu hình đồng nhất trong Search Space (MLP activations, optimizers, learning rate).
     - Bảo vệ cổng holdout đơn dùng với token mã hóa (`SealedHoldoutTest`).
     - Tính toàn vẹn chuỗi hash SHA-256 trong SQLite WAL Event Ledger (bổ sung `experiment_id`).
     - Lưu trữ chính xác các giải pháp non-dominated trong ParetoArchive (hypervolume 2D/3D).
     - Cơ chế hard timeout và giám sát bộ nhớ thực tế RSS qua `psutil`.
  3. Bản phát hành `v4.0.1` (commit `0a4176f`) là baseline lịch sử được phát hành trước khi sửa lỗi P0, do đó việc chạy benchmark trên nhánh `release-amde-final` / `experiment/real-world-evidence-demo` là bắt buộc để phản ánh đúng khả năng kỹ thuật thực tế của hệ thống.

---

## 3. Các Thành Phần Kiến Trúc Đã Được Kiểm Thử & Có Thể Sử Dụng

Qua kiểm tra trực tiếp mã nguồn trong `src/amde/` và kết quả chạy unit test hồi quy (`tests.unit.test_amde4_p0_correctness`), các thành phần sau hoạt động ổn định và sẵn sàng:

1. **`SearchSpace` (`amde.space.search_space`):**
   - Hỗ trợ 4 họ mô hình phân loại:
     - `tree_ensemble`: `RandomForestClassifier`, `GradientBoostingClassifier` (số cây 10–100, max_depth 2–8, min_samples_split 2–10).
     - `mlp`: `MLPClassifier` (1–3 tầng ẩn, độ rộng 16–128, kích hoạt relu/tanh/logistic, solver adam/sgd_momentum/lbfgs, learning_rate 1e-4..1e-1, weight_decay 1e-6..1e-2).
     - `linear`: `LogisticRegression` (C 1e-3..1e2, penalty l2/none, solver lbfgs).
     - `kernel_ridge`: `SVC` (C 1e-4..1e4, kernel rbf/linear/poly, gamma 1e-3..1e1).
   - Tiền xử lý: Imputer (mean/median/none), Scaler (standard/robust/minmax/none), Feature Selection (`SelectKBest` với `f_classif`).
   - Tạo mã hash bất biến $H(C) = \text{SHA256}(\text{canonical\_json}(G))$.

2. **`StandardEvaluationPipeline` (`amde.evaluation.pipeline`):**
   - Thực thi pipeline 10 bước chuẩn hóa.
   - Tiền xử lý được fit **chỉ trên tập train** ($X_{\text{train}}$), tuyệt đối không rò rỉ thông tin validation hay test.
   - Đo lường tài nguyên thực tế: Wall-clock train duration, inference latency (ms/100 mẫu), bộ nhớ RSS process (`psutil`).
   - Tính toán đầy đủ metrics: Accuracy, Macro F1, Balanced Accuracy.

3. **`SealedHoldoutTest` (`amde.evaluation.sealed`):**
   - Đóng gói tập dữ liệu kiểm tra 20% đằng sau token bảo mật một lần (`uuid.uuid4().hex`).
   - Chỉ cho phép đánh giá candidate incumbent một lần duy nhất sau khi hoàn thành tìm kiếm; kích hoạt `EvaluationError` nếu tái sử dụng.

4. **`EventLedger` (`amde.ledger.event_ledger`):**
   - Ghi nhận tuần tự mọi sự kiện vào SQLite ở chế độ Write-Ahead Logging (WAL).
   - Chuỗi hash SHA-256 bảo đảm tính toàn vẹn cục bộ: $H_i = \text{SHA256}(H_{i-1} \Vert \text{payload} \Vert \text{experiment\_id})$.

5. **`ParetoArchive` (`amde.archive.pareto`):**
   - Duy trì tập nghiệm non-dominated chính xác theo 3 mục tiêu: Validation Loss (cần giảm), Inference Latency (ms, cần giảm), Peak Memory (MB, cần giảm).
   - Tính toán exact 2D Hypervolume và 3D Hypervolume.

6. **Chiến Lược Tìm Kiếm & Bộ Điều Khiển:**
   - `RandomSearchStrategy`: Lấy mẫu ngẫu nhiên đồng nhất từ `SearchSpace`.
   - `TPEStrategy`: Mô hình hóa mật độ Parzen có điều kiện $l(x)/g(x)$, tích lũy lịch sử đánh giá chính xác, lấy mẫu tối đa hóa Expected Improvement proxy.
   - `EGADAController`: Meta-controller thích ứng với Discounted Thompson Sampling và mở rộng phương sai cho các arm chưa chọn.
   - `CandidateDeduplicator`: Ngăn chặn đánh giá lặp lại cùng một genome hình thái, tiết kiệm chi phí huấn luyện.

---

## 4. Các Lỗi & Giới Hạn Cần Chú Ý Trong Quá Trình Demo

1. **Giới Hạn Bộ Phân Tích ARFF Trong Môi Trường Mới (Python 3.14):**
   - Thử nghiệm tải dataset từ OpenML (`fetch_openml('spambase')`) gặp lỗi `TypeError: expected string or bytes-like object, got 'int'` trong parser ARFF của `scikit-learn 1.9.1` khi chạy cùng `pandas 3.1.0` trên Python 3.14.
   - *Biện pháp phòng ngừa:* Sử dụng dataset tích hợp sẵn trong `scikit-learn` (như `load_digits`, `load_breast_cancer`, `load_wine`) để bảo đảm 100% khả năng chạy offline, không bị ảnh hưởng bởi lỗi parser bên ngoài và không phụ thuộc kết nối mạng.
2. **Cảnh Báo Tính Năng Hằng Số (`constant_features_idx`):**
   - Trên một số dataset có viền pixel giá trị 0 không đổi (như `digits`), `SelectKBest` phát sinh cảnh báo `UserWarning: Features are constant`. Đây là hành vi bình thường của scikit-learn, cần bọc bộ lọc `warnings.filterwarnings('ignore')` trong demo để giữ log sạch sẽ.
3. **Cảnh Báo Hội Tụ Của MLP/Logistic Regression:**
   - Khi chạy ở ngân sách nhỏ (ví dụ 50 epochs), một số mô hình MLP hoặc LogisticRegression có thể phát sinh `ConvergenceWarning`. Pipeline AMDE xử lý an toàn cảnh báo này và ghi nhận metric thực tế tại epoch dừng.
4. **Phần Cứng & Tài Nguyên:**
   - Môi trường tham chiếu là CPU (Intel Core i7-8850H 6C/12T, Pascal GPU P2000 không có Tensor Cores). Demo phải thiết kế theo hướng **CPU-First** để chạy mượt mà trên cả máy tính cá nhân lẫn Google Colab (CPU instance).

---

## 5. Đánh Giá Các Benchmark Cũ

1. **Benchmark AMDE 4.0 (250 Lượt Chạy Lịch Sử):**
   - 5 phương pháp $\times$ 10 bài toán $\times$ 5 seeds $\times$ 50 evaluations.
   - Được thực hiện trước khi có bản vá lỗi P0 (thời điểm TPE chưa tích lũy đầy đủ observation history qua API cũ).
   - *Kết luận:* Không dùng số liệu này làm kết quả của AMDE 4.0.1 đã sửa lỗi.
2. **Chiến Dịch Requalification 90 Lượt Chạy (P0 Confirmatory Benchmark):**
   - Chạy trên 9 bài toán với 18 cặp đối chứng thực tế.
   - Kiểm định Wilcoxon signed-rank:
     - So với Random Search: $\Delta = +0.0003, W = 17.0, N_{\text{eff}} = 9, p = 0.5139$ (Holm-Bonferroni $p = 1.0000$).
     - So với Bayesian TPE: $\Delta = +0.0016, W = 23.0, N_{\text{eff}} = 11, p = 0.2811$ (Holm-Bonferroni $p = 0.8433$).
   - Kết quả này chứng minh: Trên các bài toán dạng bảng cơ bản với 50 evaluations, AMDE **không vượt trội có ý nghĩa thống kê** về độ chính xác phân loại so với Random Search hay TPE.
3. **Yêu Cầu Cho Benchmark Mới:**
   - Thực hiện một bài toán ML thực tế độc lập, minh bạch, so sánh công bằng giữa:
     - **Method A:** Random Search
     - **Method B:** TPE (Triển khai TPE đã kiểm chứng)
     - **Method C:** AMDE 4.0 (Toàn bộ kiến trúc gồm EGADA + Deduplication + Pareto + Ledger)
   - Đo lường đầy đủ: Best-so-far trajectory, Wall-clock time, Model fitting time, Search overhead, Tỷ lệ trùng lặp, Bộ nhớ RSS, và Kết quả trên Sealed Holdout Test.

---

## 6. Những Kết Luận Chưa Được Chứng Minh (Tuyệt Đối Không Tuyên Bố)

1. *Không tuyên bố AMDE vượt trội hơn Optuna hay Random Search về độ chính xác.* (Dữ liệu thực nghiệm xác nhận $p > 0.05$).
2. *Không tuyên bố tăng tốc GPU / CUDA / Tensor Cores.* (Mã nguồn chạy scikit-learn trên CPU).
3. *Không tuyên bố tiết kiệm VRAM.* (Chưa từng có thực nghiệm đo lường so sánh VRAM hợp lệ).
4. *Không tuyên bố thuật toán Quantum AI / VQE / Mô phỏng lượng tử.* (Mã nguồn tập trung vào classical tabular ML).
5. *Không nhầm lẫn việc chạy thành công với hiệu quả tối ưu.*

---

## 7. Đề Xuất Bài Toán Demo Khả Thi Nhất

Sau khi khảo sát các dataset và chạy thử nghiệm pilot:
- **Bài toán được chọn:** **Phân loại chữ số viết tay (Optical Recognition of Handwritten Digits - Digits)**.
- **Nguồn:** UCI Machine Learning Repository / NIST / `sklearn.datasets.load_digits`.
- **Kích thước:** **1,797 mẫu**, **64 đặc trưng số** (ảnh 8x8 pixel), **10 lớp** (chữ số từ 0 đến 9, phân bố cân bằng ~180 mẫu/lớp).
- **Phân chia dữ liệu:**
  - Tập phát triển (80% = 1,437 mẫu):
    - Huấn luyện ($X_{\text{train}}$): **1,077 mẫu** (60% tổng thể).
    - Thẩm định ($X_{\text{val}}$): **360 mẫu** (20% tổng thể).
  - Tập kiểm tra niêm phong ($X_{\text{test}}$): **360 mẫu** (20% tổng thể, khóa bằng token).
- **Metric tối ưu:** Multi-class Classification Accuracy (đồng thời theo dõi Macro F1 và Balanced Accuracy).
- **Lý do lựa chọn:**
  1. *Thực tế và phi tầm thường:* Bài toán 10 lớp với 64 đặc trưng đòi hỏi sự phối hợp giữa lựa chọn mô hình (MLP, Random Forest, Gradient Boosting, SVC) và tiền xử lý (Scaler, Feature Selection). Một cấu hình tệ (ví dụ SVC với gamma không phù hợp) chỉ đạt 10% (đoán mò), trong khi cấu hình tốt đạt > 98%.
  2. *Độc lập:* Khác với benchmark cũ chỉ dùng 600 mẫu cắt xén, benchmark mới sử dụng toàn bộ 1,797 mẫu nguyên bản.
  3. *Tốc độ và tính khả thi trên CPU:* Pilot 20 evaluations chạy xong trong **12.59 giây** trên CPU. 5 seeds $\times$ 30 evaluations sẽ hoàn thành trong khoảng 4–5 phút.
  4. *Tương thích hoàn hảo với Google Colab:* Không cần tải file nặng, không lo lỗi mạng, không cần GPU trả phí, chạy mượt mà trên môi trường Colab CPU miễn phí.
