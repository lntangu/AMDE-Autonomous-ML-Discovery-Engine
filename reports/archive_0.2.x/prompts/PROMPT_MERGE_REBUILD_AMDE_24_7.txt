# MASTER TASK — MERGE, HARDEN, VERIFY AND PACKAGE AMDE FOR SUPERVISED 24/7 RESEARCH

## Vai trò

Bạn là Principal ML Systems Engineer, Research Engineer, Test Engineer và Reliability Engineer. Nhiệm vụ là **thực hiện công việc trên mã nguồn thật**, không chỉ lập kế hoạch, không chỉ sinh file khung, không chỉ viết README. Phải kiểm tra, sửa, chạy test, xác minh, đóng gói và bàn giao artifact có thể dùng được.

## Mục tiêu duy nhất

Từ tất cả các phiên bản AMDE được cung cấp, tạo ra **một repository duy nhất tốt nhất theo bằng chứng**, giữ lại phần có ích và ổn định, hợp nhất lợi thế không trùng lặp, sửa lỗi, tăng khả năng tái lập, bảo vệ dữ liệu đánh giá, sử dụng CPU/GPU/RAM hợp lý và có thể chạy nghiên cứu dài hạn trên Windows với checkpoint/khôi phục và cơ chế dừng an toàn.

Mục tiêu nghiên cứu là xây dựng một hệ thống tự động khám phá, thử nghiệm, so sánh và chọn lọc các phương pháp học máy ngày càng tốt hơn. **Không được tuyên bố** rằng nó đã tìm ra thuật toán tối ưu tuyệt đối, tiến bộ vô hạn, đạt AGI, vượt trội phương pháp truyền thống hoặc khai thác GPU thật nếu chưa có phép thử chứng minh trực tiếp.

## Nguồn vào và xử lý nguồn

Có thể được cung cấp:

1. `full.zip` chứa nhiều bản triển khai độc lập/biến thể. Trong bản release này, archive gốc cũng nằm tại `reference/source_inputs/full.zip`; ưu tiên dùng đúng archive này khi SHA-256 khớp `manifests/SOURCE_MANIFEST.json`.
2. Một repository AMDE hiện tại đã hợp nhất một phần.
3. Báo cáo, log, test, checkpoint và script từ từng bản.

Tên file, tên folder, tên project hoặc tên nhãn **không phải bằng chứng về model AI đã sinh code**; nhãn có thể đã bị đổi có chủ ý. Không đoán danh tính model từ văn phong. Dùng SHA-256, dấu vết code, manifest, test và kết quả tái chạy để định danh artifact.

Trước khi sửa:
- Lập manifest nguồn, SHA-256, file count, Python line count và quan hệ trùng lặp.
- Giữ nguyên tất cả nguồn gốc ở vị trí read-only hoặc bản sao lưu; không ghi đè archive đầu vào.
- Đọc mã nguồn, config, test, log và báo cáo của từng phiên bản. Không chỉ đọc README.
- Chạy test hiện có trong môi trường cách ly để biết baseline lỗi thực tế.
- Phân biệt số liệu do report tự khai với số liệu đã tái chạy được.

## Quy tắc chọn lọc và hợp nhất

Không ghép mã bằng cách copy toàn bộ mọi file. Mỗi tính năng được phân vào một trong các nhóm:

- Giữ nguyên: có giá trị, đã kiểm thử, tương thích kiến trúc chính.
- Port có kiểm soát: ý tưởng có ích nhưng cần adapter/API mới.
- Viết lại: ý tưởng có ích nhưng implementation lỗi hoặc quá phụ thuộc nền cũ.
- Loại bỏ: trùng lặp, không chạy được, không có bằng chứng lợi ích, vi phạm ranh giới an toàn hoặc tăng độ phức tạp vô ích.

Với mỗi phần được port, ghi lại nguồn, lý do, tệp/hàm đích và test chứng minh. Ưu tiên một kiến trúc rõ ràng hơn là nhiều hệ thống bán trùng lặp. Tránh duy trì song song hai module có cùng chức năng nếu chưa chứng minh cả hai cần thiết.

Các thành phần quan trọng phải được kiểm tra:
- Genome/DSL có schema tường minh, giới hạn node/độ sâu/kích thước, không `eval`, `exec`, shell hoặc chạy code ứng viên tùy ý.
- Các chiến lược random, evolutionary, crossover, novelty, local; population/islands; Pareto và novelty archive; genealogy/parent hashes.
- Multi-fidelity screening và early stopping; không được lọc ứng viên chỉ bằng một noisy score mà loại mất mọi hướng khám phá mới.
- Promotion guard có đối chứng thật; không hạ tiêu chuẩn để ứng viên dễ thắng.
- Checkpoint atomic, checksum, backup, resume, single-instance lock, heartbeat, graceful stop, WAL/SQLite ledger.
- Tập challenge/held-out tách biệt; search code không được nhìn thấy dữ liệu hoặc per-example score của challenge.
- Resource scheduler đọc trạng thái CPU/RAM/disk/GPU/VRAM/nhiệt độ trong phạm vi công cụ thực tế có sẵn.
- Stagnation detector, nhưng mặc định **chỉ phân tích/đề xuất**. Không tự âm thầm đổi metric, bộ dữ liệu, mức khó, tiêu chí thắng hoặc protocol để làm tăng điểm.
- So sánh đối chứng công bằng với random search và hyperparameter-only search, cùng seed, cùng budget, cùng task/split/metric.
- CLI, launcher Windows, log, doctor report, status, stop và tài liệu khởi động lại.

## Cấu trúc nghiên cứu bắt buộc

### A. Candidate representation và sandbox

Chỉ cho phép ứng viên được biểu diễn bằng JSON/schema đã kiểm tra. Không cho phép genome chứa Python tùy ý hoặc thực thi shell/network. Mọi phép toán DSL phải được allowlist, có kiểm tra số hữu hạn, bounds, độ sâu, kích thước và timeout.

### B. Discovery engine

Hệ thống được phép thử nghiệm theo các cấp độ:
1. Siêu tham số, seed policy, regularization, batch size, training schedule.
2. Cấu hình optimizer, loss, initialization, preprocessing.
3. Kiến trúc mô hình được giới hạn rõ ràng.
4. Họ update rule/DSL có kiểu dữ liệu và giới hạn an toàn.
5. Chiến lược tìm kiếm, meta-policy hoặc cách phân bổ ngân sách — chỉ khi có baseline và phép thử riêng.

Không mô tả việc đổi `learning_rate` hoặc đột biến vài trường JSON là “phát minh thuật toán học mới”. Theo dõi độ phức tạp, novelty, số họ chiến lược thực sự được thử và tỷ lệ ứng viên lỗi.

### C. Bộ đánh giá độc lập

- Định nghĩa baseline trước khi mở thử nghiệm.
- Cố định metric, task split, seed và ngân sách cho mỗi lần đối chứng.
- Tách dev khỏi challenge/final holdout theo **họ nhiệm vụ**, không chỉ chia ngẫu nhiên nhiều mẫu tương tự.
- Challenge không được dùng cho tìm kiếm, chọn checkpoint, sửa code, điều chỉnh siêu tham số hoặc quyết định tiếp tục phát triển.
- Mọi báo cáo phải nêu số seed, số task, khoảng tin cậy hoặc mức bất định phù hợp, chi phí tính toán, timeout, candidate failures, số lần đánh giá và tất cả khác biệt protocol.
- Search RNG phải lấy từ experiment seed chứ không lấy từ `run_id` (run ID chỉ là nhãn thư mục). Thêm test tái lập: cùng cấu hình/protocol/seed nhưng khác run ID phải cho cùng proposal sequence; seed RNG protocol phải có version/hash để không tái dùng cache/checkpoint theo ngữ nghĩa cũ.
- Một thí nghiệm ngắn chỉ là smoke test/engineering check; không dùng làm bằng chứng tổng quát hóa.

### D. CPU/GPU/RAM

- Dò CPU, RAM, disk và NVIDIA GPU; ghi rõ nguồn thông tin và thời điểm đo.
- CUDA chỉ được đánh dấu active sau khi PyTorch import được, `torch.cuda.is_available()` true và một tensor operation thực sự trên GPU chạy thành công.
- Kiểm tra free VRAM, compute capability, driver/runtime compatibility và thermal limit. Nếu không an toàn hoặc không tương thích, dùng CPU và ghi rõ lý do.
- Với GPU bộ nhớ nhỏ, giới hạn đồng thời GPU job ở 1 theo mặc định; không cho từng process chiếm VRAM không kiểm soát.
- CPU workers phải bị giới hạn theo logical/physical core và RAM khả dụng; tránh oversubscription do OMP/MKL/BLAS lồng nhau.
- Dùng resource monitor để pause hoặc giảm concurrency khi vượt reserve, thiếu disk, thiếu VRAM hoặc quá nhiệt; không ép CPU/GPU 100% bất kể nhiệt độ.
- Không cài tự động một bản Torch/CUDA chỉ vì “mới nhất”. Chọn theo compatibility chính thức, cho phép CPU fallback, không phá môi trường hiện có.
- Thử so sánh output CPU/tensor-backend trên cùng dữ liệu và seed trước khi bật GPU execution. Sai khác số học phải được đánh giá bằng tolerance định trước.
- Không tuyên bố GPU được sử dụng nếu chỉ có `nvidia-smi` thấy card.

### E. Chạy dài hạn và khôi phục

Hệ thống phải có:
- Checkpoint theo generation và checkpoint recovery fallback có checksum.
- Resume chỉ khi config/protocol/data hash tương thích; thay đổi thiết bị hoặc metric phải tạo protocol hash mới/không tái dùng cache sai.
- SQLite transaction, immutable trial history và artifact budgets.
- Heartbeat có timestamp, pid, generation, trial count, backend thực tế, RAM/CPU/disk/GPU/VRAM khi có thể.
- Watchdog Windows có restart backoff, log riêng cho từng lần chạy, circuit breaker với lỗi cấu hình/lỗi lặp lại, không tạo vòng lặp restart vô hạn khi code lỗi.
- Dừng qua stop file, kết thúc safe step, ghi checkpoint và summary.
- Không tự xóa stop file, checkpoint, ledger, report hoặc archive cũ khi khởi động lại.
- Có `doctor`, `smoke`, run có giới hạn, `continuous`, `status`, `stop`, `report` và control experiment.

Chạy vô hạn là một chế độ vận hành, không phải bằng chứng rằng chất lượng tăng vô hạn. Báo cáo phải phát hiện đình trệ và đề xuất thí nghiệm đối chứng, không chỉ sinh thêm thế hệ vô nghĩa.

## Sửa lỗi và chiến lược kiểm thử

1. Lập danh sách test thất bại ban đầu.
2. Sửa lỗi gốc, không chỉ sửa assertion để test pass.
3. Thêm regression test cho từng lỗi đã sửa.
4. Chạy `compileall`/static import checks.
5. Chạy toàn bộ unit suite.
6. Chạy integration suite, bao gồm CLI once/status/report/inspect, graceful stop/resume, checkpoint corruption recovery, challenge leakage boundary, cache-key device separation, resource pressure và checkpoint hash compatibility.
7. Nếu có backend tensor: parity test với CPU reference trên linear, MLP, mọi optimizer/update-rule được hỗ trợ, classification/regression và nhiều seeds; nếu không có GPU thật thì phải ghi là “CUDA chưa được xác minh trên phần cứng”, không giả vờ đã pass GPU test.
7a. Kiểm tra tính tái lập search: cùng experiment seed/config phải không đổi proposal stream vì tên run ID; xác minh bằng unit test và ít nhất hai run nhỏ có ID khác nhau.
8. Chạy `doctor`, end-to-end smoke và một run nhỏ có giới hạn thời gian/generation. Đọc summary/ledger mới sinh để xác minh thực tế; không chỉ xem exit code.
9. Chạy control experiment nhỏ: portfolio so với random-only và HPO-only, cùng seed/budget/data. Báo số liệu như thử nghiệm sơ bộ, không dùng để tự động sửa design.
10. Kiểm tra không có test phụ thuộc thứ tự, không sử dụng challenge trong dev worker, không có cache collision giữa CPU và CUDA, không có test pass giả do skip vô ý.
11. Ghi rõ test nào chạy thành công, test nào bỏ qua, test nào bị timeout và vì sao. Không nói “all tests pass” nếu chưa chạy toàn bộ.

## Tối ưu hóa yêu cầu

Chỉ tối ưu sau khi có baseline và profiling. Ưu tiên:
- Không đánh giá trùng candidate/genome trên cùng protocol.
- Cache key đầy đủ: genome, config/protocol, data manifest, seed, stage, epochs scale, metric, engine version và effective device.
- Multi-fidelity dùng ngân sách thấp ở vòng screening và confirmation đủ mạnh cho ứng viên quan trọng.
- Hạn chế I/O và log tăng trưởng không kiểm soát.
- Dùng batch tensor operations khi nó thật sự tăng tốc trong workload này.
- Resource-aware scheduling dựa trên hiệu năng/giờ và chi phí phép tính, không tối đa hóa utilization vô nghĩa.
- Checkpoint theo mốc ổn định và throttling khi tài nguyên bị áp lực.
- Không thêm dependency lớn nếu standard library hoặc module đã có giải quyết được.

## Điều kiện nghiệm thu trước khi đóng gói

Không bàn giao cho tới khi:
- Repository có cấu trúc hoàn chỉnh, không phải file khung/placeholder.
- Các regression test mới đều pass.
- Unit + integration suites đã chạy; kết quả và commands được ghi vào report.
- `doctor` thể hiện trung thực backend thực tế.
- Smoke run có summary, ledger và checkpoint; resume cùng protocol hoạt động.
- Thử dừng an toàn không làm mất dữ liệu.
- Watchdog và launcher đã được rà soát syntax; nếu chưa có Windows PowerShell thật để chạy thì phải ghi “reviewed but not runtime-tested on Windows”.
- Control experiment có output schema rõ và disclaimer về dev metrics.
- README có hướng dẫn cài đặt, kiểm tra môi trường, chạy bounded test, bắt đầu 24/7, kiểm tra trạng thái, dừng và khôi phục.
- Có `MERGE_AUDIT.md`, `VERIFICATION_RESULTS.md`, `SOURCE_MANIFEST.json` gồm hash, `SHA256SUMS.txt`, và prompt này.
- Không đóng gói `.venv`, cache, bytecode, các database run thử quá lớn hoặc checkpoint giả. Chỉ giữ test/report/artifacts nhỏ cần thiết.
- Tạo ZIP sạch, kiểm tra unzip được, chạy test smoke từ thư mục giải nén mới nếu thời gian cho phép, và xác minh hash archive.

## Quy trình thực hiện bắt buộc

### Giai đoạn 1 — Inventory
Đọc toàn bộ nguồn vào; tạo `SOURCE_MANIFEST.json` và `MERGE_AUDIT.md`. Định danh các module trùng lặp bằng hash, không dựa trên tên model/nhãn.

### Giai đoạn 2 — Chọn nền và hợp nhất
Chọn codebase có test/recovery/ledger/safety tốt nhất làm nền. Port đúng những tính năng khác biệt thật sự hữu ích từ các bản còn lại, ghi traceability. Không copy bug hoặc báo cáo tự khai làm bằng chứng.

### Giai đoạn 3 — Sửa lỗi
Dựa trên test thất bại tái lập, sửa code gốc, thêm regression tests. Giữ nguyên các giới hạn bảo mật và challenge boundary.

### Giai đoạn 4 — Nâng cấp có kiểm soát
Thêm hoặc cải tiến GPU backend, resource policy, stagnation reports, control experiments, checkpoint/watchdog chỉ khi tích hợp được vào kiến trúc và có test. Không thêm tính năng để đạt số lượng.

### Giai đoạn 5 — Xác minh
Chạy compile, unit, integration, doctor, smoke, stop/resume, control experiment nhỏ. Lưu lệnh, exit code, thời lượng, số test pass/fail/skip, GPU availability và mọi giới hạn thực thi trong `VERIFICATION_RESULTS.md`.

### Giai đoạn 6 — Đóng gói
Tạo repository cuối cùng có phiên bản rõ ràng, tạo `SHA256SUMS.txt`, tạo ZIP, verify ZIP integrity và đưa link tới:
1. ZIP chính.
2. Prompt dùng để lặp lại nhiệm vụ.
3. Audit report.
4. Verification report.

## Đầu ra bắt buộc

- Một repository AMDE thống nhất đã được sửa và kiểm thử.
- Một ZIP sạch có thể giải nén và chạy theo README.
- Một prompt hoàn chỉnh (giữ nhiệm vụ này) để agent có thể lặp lại merge/audit/test/package trên các input mới.
- Báo cáo ghi rõ các vấn đề còn chưa giải quyết.
- Lệnh thực tế để chuẩn bị `.venv`, chạy `doctor`, `smoke`, control experiment, bắt đầu `watchdog_24x7`, xem trạng thái và graceful stop.

## Điều tuyệt đối không được làm

- Không giả định 24 giờ chạy đồng nghĩa tốt hơn.
- Không tuyên bố “hoàn hảo”, “vô hạn tiến bộ”, “tối ưu nhất thế giới”, “vượt truyền thống” khi chưa có số liệu đối chứng hợp lệ.
- Không dùng benchmark public/holdout làm tín hiệu để phát triển candidate liên tục.
- Không thêm primitive/quy tắc chỉ vì một bài dev thất bại nếu chưa đánh giá rủi ro overfitting.
- Không tự đổi metric, data split, seed, baseline hoặc promotion guard để làm điểm cao hơn.
- Không che giấu test fail/timeout hoặc tự khai test đã pass.
- Không để watchdog lặp lỗi cấu hình mãi hoặc xóa dữ liệu khi recovery.
- Không dừng sau khi chỉ viết kế hoạch. Phải hoàn thành implementation, verification, report và ZIP trong phiên làm việc hiện tại.

## Kết luận bàn giao

Bàn giao đúng artifact đã thực sự được tạo trong môi trường làm việc. Nêu rành mạch phần nào đã được chạy trực tiếp, phần nào chỉ được kiểm tra bằng code/tests, phần nào còn cần xác minh trên Windows/NVIDIA GPU của người dùng. Ưu tiên sự thật có thể tái lập hơn mọi tuyên bố lớn.
