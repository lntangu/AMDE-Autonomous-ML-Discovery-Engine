# AMDE — Kiểm toán độc lập 6 bản triển khai ẩn danh

**Ngày kiểm toán:** 2026-10-29  
**Phạm vi:** nội dung người dùng tải lên trong `full.zip`; kiểm tra cấu trúc, đối chiếu mã nguồn trùng lặp, đọc log/báo cáo JSON, chạy lại test suite trên môi trường sandbox hiện tại.  
**Mục tiêu chấm:** chất lượng của *artifact code* như nền tảng cho Autonomous ML Discovery Engine, không phải đánh giá năng lực tổng quát của nhà cung cấp/LLM đã sinh code.

## 0. Giới hạn nhận dạng

Tên các archive trong `full.zip` được dùng **chỉ làm dấu định vị**, không dùng để suy đoán AI nào đã viết chúng. Nội dung không chứa metadata đáng tin cậy để khôi phục mapping từ artifact về model gốc. Vì vậy báo cáo dùng ID A–F gắn với SHA-256 của từng member; nếu tên đã bị tráo, hãy đối chiếu theo hash, không theo filename.

| ID | Member trong `full.zip` (chỉ để tìm file) | SHA-256 đầy đủ của member | Nội dung chính |
|---|---|---|---|
| A | `deepflash.zip` | `490bfcfb0a715e020f009f764727693105b118a04520b4b24c441af5fea6c3c5` | AMDE modular, artifacts cho 6 thế hệ, ledger, checkpoint, báo cáo audit |
| B | `wen.tar` | `4ec885255f29c63bdbfcf30914edb7a22669beaa84d7f34f1088ad9d8b27ef32` | 9 module Python rời; không phải project hoàn chỉnh |
| C | `clau.zip` | `3332bb9d5ac7ee44d58a5501b82d778be21a63da8f284889005bfceb817948e2` | AMDE với control experiment, search portfolio và audit khá chi tiết |
| D | `chat.zip` | `55776a3edba56e39584232431d9aba11c43319e95fe8f54936d9d873ec33b015` | Bản nhỏ, có loop chạy được và session report |
| E | `omni.zip` | `3ea8629791b8d59b8ff121144763fa3a2fa74e31739f784e9188a8d5642077df` | Bản gần trùng với A nhưng có regression/failure trong test và run artifact |
| F | `antigrav.zip` | `d3f6a6cb32db31b3de3097006e207f59177da92f10dc4c4a861d3ed5f70a25e3` | AMDE modular độc lập, 34 test, báo cáo smoke/once |

## 1. Phán quyết tổng thể

| Xếp hạng | ID | Điểm tổng /10 | Điểm mạnh nhất | Trở ngại lớn nhất | Quyết định |
|---:|---|---:|---|---|---|
| 1 | **A** | **8.1** | Nền tảng kỹ thuật đầy đủ nhất; test unit nhiều; có run dài nhất được ghi nhận ở đây | GPU không thực sự dùng; ARC chưa có learner; bằng chứng chỉ ở synthetic suite nhỏ | **Chọn làm codebase nền**, nhưng chưa treo chạy dài ngay |
| 2 | **C** | **7.9** | Thực nghiệm đối chứng trực tiếp nhất giữa HPO, random search và portfolio | Full test suite chưa tái xác minh hết ở sandbox hiện tại; 3 seed, dev được tái sử dụng; CPU-only | **Giữ cách thiết kế thí nghiệm**, cân nhắc port phần so sánh sang A |
| 3 | **F** | **6.7** | Modular, có test bao phủ lifecycle/DSL/checkpoint/anti-leakage | Run report thiếu nhất quán; thực nghiệm chỉ ngắn; GPU/ARC không chạy | Giữ làm tài liệu tham khảo, không chọn làm nền chính |
| 4 | **D** | **6.4** | Đơn giản, 17/17 test pass, kết quả ghi rõ giới hạn | Chỉ 2 họ task synthetic; leader chưa thử challenge; engine tập trung trong file lớn | Dùng làm prototype tham khảo, không dùng làm lõi cuối |
| 5 | **E** | **5.8** | Nhiều tính năng và artifacts; có một run synthetic hoàn tất | 5 test fail, hai run fail; gần như là nhánh code của A nhưng kém ổn định | **Không chọn riêng**, chỉ tham khảo phần khác biệt nếu có lý do cụ thể |
| 6 | **B** | **1.5** | Có một số module nền tảng về genome/grammar/tensor/util | Không có entry point, config, evaluator, search loop, tests hay run evidence | Chưa phải ứng viên triển khai hoàn chỉnh |

**Cách tính điểm:** điểm chuyên gia theo 5 trục: độ hoàn chỉnh kiến trúc (25%), correctness/test evidence (25%), breadth của search (20%), độ nghiêm túc thực nghiệm (20%), mức đáp ứng mục tiêu AMDE (10%). Đây là rubric đánh giá artifact, không phải benchmark khoa học chuẩn hóa. Các điểm cách nhau ít không nên xem như độ chính xác tuyệt đối.

## 2. Những phát hiện chung ảnh hưởng đến cả sáu bản

### 2.1 Không bản nào chứng minh được “tự tìm ra phương pháp học máy tốt hơn cách truyền thống” ở mức rộng

Các bản hoàn chỉnh đang tối ưu trong không gian định nghĩa trước: kiểu mô hình linear/tiny MLP, optimizer/recipe/hyperparameter và một DSL giới hạn cho update rule. Đây là **AutoML / tìm kiếm trong không gian ứng viên được thiết kế thủ công**, chưa phải bộ máy có thể phát minh mọi thuật toán học máy không giới hạn.

Ở run A, ứng viên tốt nhất là level C, có `dsl.w_update`, `dsl.b_update`, `dsl.m_update`, `dsl.v_update` đều `null`; cải thiện được báo cáo có thể đến từ recipe/optimizer/kiểu mô hình, không phải quy tắc cập nhật mới. Ở run E, best candidate level B với optimizer momentum và DSL cũng rỗng. Ở run D, champion thế hệ 2 đến từ chiến lược `random`, không phải `evolutionary`. Các kết quả đó chưa chứng minh rằng tiến hóa/DSL vượt random search hoặc HPO.

### 2.2 GPU chỉ được phát hiện hoặc dự trữ, không thực sự dùng huấn luyện

- A: báo cáo xác nhận `gpu_effective=false`, `gpu_slots=0`; trainer dùng CPU.
- C: README/audit nói learner NumPy/CPU, `--device cuda` không kích hoạt backend CUDA.
- D: `candidate_backend=numpy-cpu`, `cuda_available=false` trong artifact đang có.
- E: resource plan và limitation nói trainer CPU; một vài summary có cờ `gpu_effective` không nhất quán với `device_effective=cpu`/`gpu_slots=0`.
- F: PyTorch được ghi là bản CPU (`2.14.1+cpu`), `preferred_device=cpu`.
- B: không có engine/runtime để đánh giá việc dùng GPU.

Do đó, yêu cầu “tận dụng GPU + CPU + RAM” **chưa được bản nào đáp ứng đầy đủ**. Detect GPU không đồng nghĩa có CUDA training.

### 2.3 ARC chưa được thực thi

A, C, D và F đều ghi ARC là `BLOCKED` vì thiếu task data cục bộ; ít nhất A/C còn ghi rõ learner chưa áp genome trực tiếp lên grid ARC. Vì vậy không dùng điểm synthetic để suy luận về ARC-AGI hoặc trí tuệ tổng quát.

### 2.4 Chạy liên tục không đồng nghĩa tiến bộ vô hạn

Các bản tốt có checkpoint, ledger, stop flag, giới hạn tài nguyên và search loop. Chúng chứng minh được khả năng chạy/thử nghiệm hữu hạn và phục hồi trạng thái; **không** chứng minh rằng để máy chạy nhiều tháng sẽ sinh ra cải tiến gấp bội. Cần theo dõi plateau, novelty thực, so sánh với random-search-only và đánh giá trên họ task chưa dùng để chọn ứng viên.

### 2.5 A và E không phải hai thiết kế độc lập

Trong các đường dẫn Python trùng nhau giữa hai archive, **46/57 tệp giống hệt byte-for-byte (~81%)**. Đây là dấu hiệu hai artifact cùng code lineage hoặc được dẫn xuất rất gần nhau. Không nên tính chúng như hai bằng chứng độc lập ủng hộ một phương pháp.

## 3. Đánh giá từng ứng viên

## A — SHA `490bfcfb` — điểm 8.1/10

**Bằng chứng có trong gói**
- Khoảng 9,684 dòng Python nguồn, 41 file Python nguồn và 109 hàm test.
- Báo cáo `reports/IMPLEMENTATION_AUDIT.md` nói đã chạy 109 tests trên Windows 11; báo cáo ghi rõ ARC và GPU training chưa được triển khai.
- `runs/once_final/summary.json`: `completed`, 6 generations, 48 trials, khoảng 17.34 giây; 36 quick + 12 confirm; best confirm score `0.499954` so baseline `mlp_small=0.445461`.
- Challenge aggregate được ghi là candidate `0.478994`, baseline `0.439493`, delta `+0.039501` trên một cohort tổng hợp duy nhất.
- Mình chạy lại `PYTHONPATH=src python -m pytest -q tests/unit`: **105 passed, 26 subtests passed**. Full integration suite không hoàn tất trong giới hạn sandbox hiện tại, nên không tuyên bố đã tái xác minh toàn bộ 109 test tại đây.

**Điểm mạnh**
- Kiến trúc tương đối hoàn chỉnh: typed genome/DSL, nhiều chiến lược tìm kiếm, multi-fidelity, evaluator tách process, ledger SQLite, checkpoint checksum/backup, resource policy, graceful stop và báo cáo.
- Phân biệt lỗi ứng viên với lỗi hạ tầng; có challenge boundary và lưu lịch sử cha-con.
- Đây là artifact có bằng chứng vận hành rõ nhất và test unit tái chạy được nhiều nhất trong nhóm.

**Điểm yếu / rủi ro**
- Không có CUDA backend; GPU P2000 không huấn luyện model.
- ARC adapter hiện baseline/read-only, không có learner ánh xạ genome sang phép biến đổi grid ARC.
- Run chỉ 6 thế hệ/48 trials; chỉ có một challenge opening. Candidate challenge có `seed_policy="constant"` và `std_seeds` gần 0, nên không được diễn giải 3 seed là bằng chứng ổn định mạnh khi chưa kiểm tra seed semantics.
- Candidate thắng là level C với DSL update rỗng; kết quả phù hợp với cải thiện recipe/architecture search hơn là phát minh thuật toán học mới.

**Đánh giá:** lựa chọn tốt nhất làm **nền kỹ thuật**. Không nên tuyên bố là phương pháp học vượt trội, và chưa nên treo continuous mode dài hạn trước khi nghiệm thu trên Windows thật, đo tài nguyên và sửa protocol seed/baseline.

## B — SHA `4ec88525` — điểm 1.5/10

**Bằng chứng có trong gói**
- Chỉ có 9 file Python, khoảng 1,163 dòng; các file là `util/rng.py`, logging/hash/stats, `discovery/grammar.py`, `discovery/genome.py`, `tensor.py` và `__init__.py`.
- Không có `pyproject.toml`/requirements hoàn chỉnh, CLI/entry point, runtime/orchestrator, evaluator, baselines, tests, launcher, ledger/checkpoint hay report/run artifacts.

**Điểm mạnh**
- Có thể chứa ý tưởng hữu ích ở grammar/genome/tensor và utility.

**Điểm yếu**
- Không thể xem là một hệ AMDE hoàn chỉnh, chưa thể kiểm tra end-to-end, chưa có cơ sở chấm hiệu năng hay khả năng chạy liên tục.

**Đánh giá:** chỉ dùng như một túi module tham khảo sau khi đọc từng file. Không chọn làm codebase và không so sánh điểm của nó với các project có runner.

## C — SHA `3332bb9d` — điểm 7.9/10

**Bằng chứng có trong gói**
- Khoảng 3,937 dòng Python nguồn; 78 test functions.
- Unit suite tái chạy được: **54 passed** trong 11.34 giây.
- Trong integration suite, 4 test đầu pass; lần chạy hiện tại không hoàn tất test tiếp theo trong giới hạn 120 giây, nên chưa xác minh độc lập toàn bộ suite. Báo cáo kèm theo ghi 78/78 pass ở môi trường Python 3.12.3 khác với sandbox hiện tại.
- `reports/control_experiment.json` có so sánh trực tiếp 3 cấu hình, mỗi cấu hình 3 seed: `hpo_A` dev mean `0.7231`; `random_ABD` `0.7724`; `portfolio` (random + evolution + novelty + guided) `0.8263`. Portfolio hơn HPO_A khoảng `+0.1032` và hơn random_ABD khoảng `+0.0539` trung bình; theo bảng, portfolio chỉ thắng random_ABD ở 2/3 seed.

**Điểm mạnh**
- **Thiết kế thực nghiệm tốt nhất nhóm:** có HPO-only, random search trên cùng không gian A/B/D và portfolio để phân biệt đóng góp của chiến lược tìm kiếm so với tinh chỉnh siêu tham số.
- Split theo họ tác vụ search/dev/challenge, baseline và metric version/hash, multi-fidelity, DSL an toàn, process timeout, chống leak, checkpoint và recovery.
- Audit thừa nhận thẳng rằng mức tăng lớn nhất có thể giải thích bằng HPO; chưa chứng minh portfolio/evolution/DSL thắng random search do n=3 và không có CI.

**Điểm yếu / rủi ro**
- Control experiment vẫn chọn champion dựa trên dev được dùng lặp lại; có adaptive overfitting.
- Chưa đủ seed/CI/power; các kết quả không chứng minh ưu thế tổng quát ngoài benchmark synthetic.
- Không có learner ARC, không có CUDA backend, chưa chứng minh chạy đa worker >1 trên máy mục tiêu.
- Full integration suite chậm trong môi trường sandbox này; cần kiểm tra lại bằng đúng lệnh/môi trường được ghi trong audit.

**Đánh giá:** **bản tốt nhất về cách kiểm chứng khoa học**. Nên lấy control experiment, random-search-only ablation và reporting evidence của bản này làm tiêu chuẩn nghiên cứu cho A; không cần thay toàn bộ codebase A bằng C ngay lập tức.

## D — SHA `55776a3e` — điểm 6.4/10

**Bằng chứng có trong gói**
- Khoảng 2,018 dòng Python nguồn, 17 test functions.
- Mình chạy lại `PYTHONPATH=src python -m pytest -q tests`: **17 passed**.
- `reports/latest_session.json`: generation 2; 12 evaluator calls trong phiên, 16 trials tích lũy; leader dev `0.936905`, fixed-linear reference dev `0.649675`, chênh `+0.28723`.
- Benchmark đang ghi là `synthetic-sanity-v1`, gồm hai họ: `nonlinear_regression` và `quadrant_xor`. Champion thế hệ 2 chưa được đánh giá trên challenge; challenge score `0.450299` là của ứng viên thế hệ 1, so fixed-linear challenge `0.393637`.

**Điểm mạnh**
- Project chạy được, test pass, có checkpoint, ledger, random/evolutionary proposal, paired-seed promotion, timeout và tài liệu giới hạn khá rõ.
- Có phân biệt leader theo dev với champion được xác nhận bằng seed khác.

**Điểm yếu / rủi ro**
- Engine tập trung trong `src/amde/engine.py` lớn (~51 KB), nên bảo trì và phân lập lỗi khó hơn kiến trúc chia module.
- Hai họ task là quá ít; điểm dev cao chỉ cho biết thích nghi tốt với bài synthetic đang dùng.
- Candidate tốt nhất được sinh bởi `random`, chỉ 2 thế hệ và 12 evaluator calls: chưa chứng minh tiến hóa tốt hơn random search.
- CPU-only, ARC `BLOCKED`, không có bằng chứng dài hạn.

**Đánh giá:** prototype nhỏ, dễ hiểu và khá trung thực về giới hạn. Dùng làm mẫu cho checkpoint/promotion hoặc bản chạy thử đơn giản, không dùng điểm `0.9369` để xếp hạng trên các bản khác vì benchmark khác hoàn toàn.

## E — SHA `3ea86297` — điểm 5.8/10

**Bằng chứng có trong gói**
- Khoảng 9,594 dòng Python nguồn, 83 test functions.
- Mình chạy lại unit suite: **78 passed, 5 failed** (cộng 6 subtests passed).
- Các lỗi quan sát được:
  1. `test_default_config_loads_and_hashes_are_stable`: số `task_specs()` là 2, trong khi test đòi tối thiểu 9.
  2. DSL `clip` trả `2.0` cho test mong `-2.0` khi bounds bị đảo.
  3. API ledger `latest_challenge_opening` lộ `candidate_aggregate_json` chứa `per_task`, trái với kỳ vọng giao diện aggregate-only của test.
  4. Portfolio không thử hết các arm; proposer không tạo operator `crossover` như contract test mong đợi.
  5. Các contract test về chiến lược chọn/đề xuất không khớp hành vi hiện tại.
- `runs/smoke_test_02/summary.json` fail với `ValidationError: DSL optimizer needs w_update and b_update`.
- `runs/smoke_test_04/summary.json` fail với `AttributeError: 'Genome' object has no attribute 'genome'`.
- `runs/once_test_01/summary.json` có một run 6 generations/48 trials hoàn tất và confirm score `0.524485` so baseline `0.445461`, nhưng summary không gắn source hash để chứng minh kết quả đó khớp tuyệt đối với snapshot code đang được đóng gói.

**Điểm mạnh**
- Có nhiều cơ chế như A: multi-fidelity, portfolio, ledger, process isolation, sealed evaluator và checkpoints.
- Một run synthetic hoàn tất với score cao hơn baseline, nhưng không so sánh trực tiếp được với các bản khác.

**Điểm yếu / rủi ro**
- 5 unit test fail và 2 run fail là tín hiệu độ ổn định thấp hơn A.
- Có 46/57 Python files chung với A giống hệt; đây không phải bằng chứng độc lập thứ hai cho chất lượng kiến trúc.
- Có dấu hiệu regression hoặc tests/artifacts không đồng bộ. Các trường hợp lộ aggregate JSON ở API là cần xử lý trước khi dùng challenge boundary.

**Đánh giá:** không chọn E làm nhánh chính khi A đã có code lineage gần giống nhưng test evidence tốt hơn. Chỉ nên cherry-pick phần khác biệt sau khi chỉ rõ nó sửa vấn đề nào và chạy regression tests.

## F — SHA `d3f6a6cb` — điểm 6.7/10

**Bằng chứng có trong gói**
- Khoảng 4,959 dòng Python nguồn; 34 test functions.
- Mình chạy test nguyên trạng: **33 passed, 1 failed** vì CLI subprocess không tìm thấy package `amde` trước khi cài project/editable path.
- Chạy lại với `PYTHONPATH=src`: **34 passed, 2 warnings** trong 9.76 giây.
- Có báo cáo smoke/once, mỗi báo cáo nói 3 trials. Candidate đứng đầu có cùng genome hash và same trial IDs qua các báo cáo, score `0.75`; nhưng `baseline_score` trong smoke là `0.7` trong khi `once` là `0.1`, không có benchmark-suite ID đủ rõ trong các summary để giải thích chênh lệch. Báo cáo once ghi `elapsed_seconds=0.16` dù các `wall_clock_seconds` của 3 top trials lần lượt khoảng 0.370, 0.159, 0.099 giây; có thể là run dùng cache, nhưng provenance không đủ để so trực tiếp.
- Hardware report ghi PyTorch `2.14.1+cpu`, `preferred_device=cpu`; ARC suite `BLOCKED`.

**Điểm mạnh**
- Modular, có genome/grammar/mutation/crossover, nhiều chiến lược search, Pareto selection, sealed protocol, checkpoint, ledger và resource manager.
- Test coverage có kiểm tra anti-leakage, chia split, resume, SQLite và điều phối tài nguyên.

**Điểm yếu / rủi ro**
- Chất lượng provenance của run summary yếu: baseline/elapsed không đủ minh bạch để xác nhận các số so sánh.
- Thí nghiệm ngắn; artifacts không cho thấy kiểm tra dài hạn hay đối chứng đủ mạnh giữa search strategy.
- GPU không thực sự được dùng; ARC chưa chạy.

**Đánh giá:** nền kiến trúc có tiềm năng, nhưng báo cáo kết quả cần sửa trước khi tin vào score và dùng nó làm cơ chế tự chọn model.

## 4. Vì sao điểm tổng không phải bảng xếp hạng “AI nào thông minh nhất”

Một codebase lớn hơn không tự động tốt hơn; số test nhiều không bảo đảm test đúng; một score cao hơn không so được nếu task suite/metric/budget khác. Báo cáo đánh giá **sản phẩm code hiện có**, không thể suy ra model gốc nào viết code này hoặc model gốc nào nói chung mạnh hơn.

Đặc biệt, tất cả các số điểm được báo cáo là trên benchmark synthetic khác nhau hoặc có thiết lập khác nhau. Không lấy `0.9369` của D, `0.8263` của C, `0.49995` của A hay `0.75` của F để xếp hạng thuật toán trực tiếp.

## 5. Khuyến nghị quyết định

1. **Lấy A làm codebase nền** vì có lịch sử vận hành/test và artifact kỹ thuật phong phú nhất trong ZIP.
2. **Lấy C làm chuẩn phương pháp nghiên cứu**: yêu cầu baseline mặc định, HPO-only, random-search-only, portfolio/evolution cùng ngân sách; tối thiểu nhiều seed hơn 3; báo cáo paired effect, CI và worst-family score.
3. **Không dùng E làm bản độc lập**: 81% mã Python chung với A và hiện có các regressions được test phát hiện.
4. **Giữ F như nguồn tham khảo**, nhưng trước khi tái sử dụng phải làm rõ benchmark ID và sửa provenance trong report exporter.
5. **D và B không làm nền chính**: D quá hẹp về benchmark; B không có executable engine.
6. Trước khi treo dài hạn: phải có thử nghiệm smoke/resume/stop trên chính Windows của m; resource/thermal safety phải hoạt động; ledger ghi lại mã nguồn/config/data hashes; GPU chỉ được đánh dấu active khi có tensor training thật; tự động dừng việc promote khi score tăng ở dev mà không được lặp lại trên họ task giữ kín.
7. Trước khi tuyên bố “tìm được phương pháp học tốt hơn truyền thống”: tối thiểu phải có đối chứng cùng task và budget với baseline đã tune, HPO/random-only, nhiều seed, họ task giữ kín và nguồn sinh tác vụ độc lập. Không dùng ARC public evaluation cho quá trình chọn ứng viên.

## 6. Lệnh tái kiểm tra đã thực hiện trong sandbox

- F: `PYTHONPATH=src python -m pytest -q tests` → 34 passed, 2 warnings. Chạy không đặt `PYTHONPATH` có 1 lỗi CLI import.
- D: `PYTHONPATH=src python -m pytest -q tests` → 17 passed.
- C: `PYTHONPATH=src python -m pytest -q tests/unit` → 54 passed; integration suite không hoàn tất trong thời hạn 120 giây ở sandbox hiện tại.
- A: `PYTHONPATH=src python -m pytest -q tests/unit` → 105 passed, 26 subtests passed; integration suite không hoàn tất trong thời hạn sandbox hiện tại.
- E: `PYTHONPATH=src python -m pytest -q tests/unit` → 78 passed, 5 failed, 6 subtests passed.
- B: không thể chạy project end-to-end vì các phần bắt buộc không có trong archive.

**Môi trường sandbox:** Linux x86_64, Python 3.13.5, pytest 9.0.2. Đây không phải chính Windows PC của người dùng; test chạy lại có thể khác với kết quả trong các báo cáo gốc. Báo cáo này không sửa bất kỳ source code nào trong các archive.
