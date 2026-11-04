# AMDE 4.0.1 — Bản Nháp Truyền Thông & Công Bố (Tiếng Việt)

> [!IMPORTANT]
> **DRAFT — HUMAN REVIEW REQUIRED / BẢN NHÁP CẦN DUYỆT THỦ CÔNG**  
> Mọi nội dung dưới đây do agent chuẩn bị để hỗ trợ maintainer. Maintainer phải tự xem xét, chỉnh sửa và quyết định trước khi đăng tải lên bất kỳ kênh nào. Không được đăng tự động.

---

## A. Hero Copy Cho README / Giới Thiệu Ngắn (30 Giây)

**AMDE (Autonomous Machine Learning Discovery Engine)** là nền tảng nghiên cứu thực nghiệm mã nguồn mở, tập trung vào việc nghiên cứu tìm kiếm ứng viên học máy, phân bổ tài nguyên đa độ trung thực (multi-fidelity), và bảo toàn đường biên Pareto đa mục tiêu trên dữ liệu dạng bảng (tabular ML).

- **Định vị chính xác:** AMDE không phải là một công cụ AutoML thương mại tuyên bố đánh bại Random Search hay TPE. AMDE là một hệ thống nghiên cứu độ tin cậy phần mềm (reliability engineering), ngăn ngừa rò rỉ dữ liệu thông qua các hợp đồng tách biệt danh tính đánh giá và lưu trữ nhật ký chứng cứ gắn chuỗi mã hóa SHA-256.
- **Tái lập trong 10 giây:** `python examples/quickstart_cpu.py` chạy hoàn chỉnh trên CPU thông thường, không yêu cầu GPU hay khóa API.

---

## B. Thông Báo Ra Mắt Trên GitHub Discussions (Chính Thức)

**Tiêu đề:** Ra mắt AMDE 4.0.1: Nền tảng nghiên cứu AutoML dạng bảng có kiểm chứng chứng cứ và đa mục tiêu

Xin chào cộng đồng,

Tôi là maintainer của dự án **AMDE (Autonomous Machine Learning Discovery Engine)**. Hôm nay, chúng tôi phát hành phiên bản nghiên cứu chính thức **AMDE 4.0.1** tại nhánh canonical `release-amde-final` trên GitHub:  
👉 https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine

### Dự án này giải quyết vấn đề gì?
Khi xây dựng và thử nghiệm các thuật toán tìm kiếm tự động cho học máy (AutoML), chúng tôi nhận thấy các lỗi rất tinh vi thường làm sai lệch kết quả benchmark:
1. **Ô nhiễm bộ nhớ đệm (Cache Contamination):** Khi candidate hash chỉ dựa trên siêu tham số, việc thay đổi seed, split, hoặc tập dữ liệu có thể vô tình tái sử dụng kết quả cũ. AMDE giải quyết triệt để bằng cách tách rời cấu trúc candidate khỏi `EvaluationCacheKey` (ràng buộc chặt chẽ với dataset, split hash, seed và rung).
2. **Rò rỉ tiền xử lý (Preprocessing Leakage):** Mọi bộ chuẩn hóa dữ liệu (scaler, imputer) bắt buộc phải fit độc quyền trên tập train trước khi transform tập validation.
3. **Thiếu tính minh bạch:** AMDE tích hợp `EventLedger` dựa trên SQLite WAL với chuỗi băm mật mã SHA-256 (`verify_chain()`) giúp phát hiện mọi hành vi sửa đổi hoặc mất mát sự kiện.

### Kết quả định lượng & Tuyên bố khoa học trung thực
Trong đợt benchmark kiểm chứng độc lập trên 6 tập dữ liệu (90 lượt chạy, 50 lượt thử mỗi lượt):
- AMDE 4.0 đạt độ chính xác trung bình $0.9871 \pm 0.0144$ so với Random Search ($0.9832 \pm 0.0239$) và Bayesian TPE ($0.9809 \pm 0.0234$).
- **Kiểm định thống kê Wilcoxon có hiệu chỉnh Holm-Bonferroni cho thấy $p \ge 0.85$.** Chúng tôi **không phát hiện sự vượt trội có ý nghĩa thống kê** so với các baseline chuẩn. Giá trị của AMDE nằm ở độ tin cậy kiến trúc và tính toàn vẹn dữ liệu, không phải tuyên bố "vượt trội thuật toán".

### Giới hạn đã biết
- AMDE hiện chỉ hỗ trợ bài toán phân loại và hồi quy dạng bảng (tabular ML).
- Cổng sealed holdout là một ranh giới kiến trúc trong tiến trình Python, không phải hộp cát cô lập cấp hệ điều hành (OS sandbox).

### Câu hỏi cần cộng đồng phản biện
Chúng tôi rất mong nhận được phản biện về thiết kế: *Theo bạn, việc tách rời Candidate Identity khỏi Evaluation Identity có đủ để ngăn ngừa hoàn toàn cache leakage trong các hệ thống AutoML phân tán không?*

Xem hướng dẫn kiểm tra độc lập tại: [INDEPENDENT_REVIEW_GUIDE.md](docs/public/INDEPENDENT_REVIEW_GUIDE.md).

---

## C. Bản Nháp Reddit (Chỉ dùng trong Self-Promotion Thread)

**Subreddit:** r/MachineLearning (Chỉ đăng trong Monthly Self-Promotion Thread)

**Nội dung:**
> **[Project] AMDE 4.0.1 – An auditable, multi-objective AutoML research testbed for tabular ML**
> 
> Hi everyone, I maintain AMDE, an open-source research framework exploring search reliability and multi-objective optimization in tabular AutoML.
> 
> Rather than claiming another "breakthrough", our focus is software reliability engineering for ML:
> - **Zero Cache Contamination:** Decouples candidate genome hashing from evaluation cache keys (which bind dataset ID, split hash, seed, and fidelity rung).
> - **Tamper-Evident Ledger:** Append-only SQLite event ledger with an unbroken SHA-256 state chain.
> - **Multi-Objective Pareto:** Exact non-dominated frontier preservation and NSGA-II crowding distance with extreme point retention.
> 
> **Statistical Disclosure:** In our 90-run benchmark across 6 datasets, AMDE achieved comparable accuracy to Random Search and Bayesian TPE ($0.9871$ vs $0.9832$ and $0.9809$), but paired Wilcoxon tests yield $p \ge 0.28$ (Holm-Bonferroni adjusted $p \ge 0.85$). We explicitly claim parity, not superiority.
> 
> Runs in 5 seconds on CPU: `python examples/quickstart_cpu.py`
> Repo: https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine
> We would love feedback on our evaluation pipeline contracts and review guide!

---

## D. Dàn Ý Bài Viết Kỹ Thuật Chuyên Sâu (Technical Blog Outline)

**Tiêu đề đề xuất:** *Bài học từ kỹ thuật độ tin cậy AutoML: Tại sao bộ nhớ đệm và rò rỉ phân tách dữ liệu lại phá vỡ các công bố benchmark*

1. **Vấn đề đặt ra:**
   - Tại sao nhiều mô hình AutoML đạt điểm số cao bất thường trong phòng thí nghiệm nhưng lại thất bại khi đưa ra dữ liệu thực tế?
   - Phân tích hiện tượng "Split Contamination" và "Cache Invalidation Failure".
2. **Ví dụ thực tế:**
   - Điều gì xảy ra khi bộ nhớ đệm dùng `hash(hyperparameters)` mà không ràng buộc `split_hash` và `seed`.
3. **Giải pháp kiến trúc trong AMDE 4.0:**
   - Thiết kế hợp đồng bất biến: `Candidate` vs `EvaluationCacheKey`.
   - Ranh giới đánh giá 10 pha: ép buộc fit preprocessor hoàn toàn độc lập trên tập train.
   - Cơ chế kiểm tra tính toàn vẹn chuỗi sự kiện bằng băm SHA-256.
4. **Dữ liệu thực nghiệm & Bài học:**
   - Báo cáo kết quả trung thực từ 90 run thực nghiệm.
   - Tại sao việc công khai $p$-value không có ý nghĩa thống kê lại giúp tăng uy tín nghiên cứu.
5. **Mã nguồn mở và mời phản biện độc lập:**
   - Đường dẫn tái lập và hướng dẫn reviewer mở issue.

---

## E. Mẫu Email / Liên Hệ Các Nhóm Nghiên Cứu Công Khai

*(Chỉ gửi khi có kênh liên hệ công khai dành cho việc trao đổi nghiên cứu; không gửi hàng loạt)*

**Tiêu đề:** Trao đổi nghiên cứu về tính tái lập và kiến trúc kiểm chứng trong AutoML (Dự án AMDE)

Kính gửi [Tên Nhà Nghiên cứu / Nhóm Nghiên cứu],

Tôi theo dõi các công trình của quý nhóm về tính tái lập và kiểm chuẩn công bằng trong học máy tự động (AutoML). 

Tôi liên hệ để giới thiệu một dự án nghiên cứu mã nguồn mở phi thương mại mà chúng tôi vừa công bố: **AMDE 4.0.1** (https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine). 

Dự án này tập trung vào các hợp đồng kỹ thuật phần mềm nhằm loại bỏ hoàn toàn các lỗi đánh giá ngầm (cache contamination giữa các lần chạy, rò rỉ tiền xử lý qua các tập phân tách, và cơ chế ghi nhận chứng cứ bằng chuỗi băm mã hóa SHA-256).

Trong báo cáo benchmark kiểm chứng, chúng tôi công khai toàn bộ dữ liệu thô và chỉ ra rằng hệ thống đạt mức độ tương đương với Random Search và Bayesian TPE ($p \ge 0.28$), không phóng đại tính ưu việt thuật toán.

Nếu quý nhóm quan tâm đến việc kiểm tra độc lập hoặc có ý kiến đóng góp cho bản hướng dẫn phản biện kỹ thuật của chúng tôi ([INDEPENDENT_REVIEW_GUIDE.md](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/blob/release-amde-final/docs/public/INDEPENDENT_REVIEW_GUIDE.md)), chúng tôi rất vinh hạnh được lắng nghe.

Trân trọng,  
Maintainer Dự án AMDE (GitHub: @lntangu)
