> **Historical record:** this report documents the original 0.2.1 merge/audit state. It is retained for provenance and is superseded for current release status by `INDEPENDENT_AUDIT_AND_RELEASE_QUALIFICATION.md`, `TEST_MATRIX.md`, and `VERIFICATION_RESULTS.md` in AMDE 0.2.6. Do not treat historical test totals or qualification claims here as results for 0.2.3.

# AUDIT BEFORE & AFTER — AMDE 0.2.1

**Ngày kiểm toán:** 2026-10-29  
**Đối tượng kiểm toán:**
1. Gói đầu vào đã hợp nhất một phần: `AMDE_Integrated_24x7_Release.zip` (SHA-256: `a3d5ad71b6cf710dab7d13829bdb19418182ecc6368eb20b3034468f3d54df89`).
2. Gói nguồn tham khảo 6 biến thể: `full.zip` (SHA-256: `6b57eeb46aa98163945f00dcb73b4bba505514eb84d3463d027a3d4e1c5c5215`).

## 1. Kiến trúc thực thi tổng thể (End-to-End Execution Flow)

```text
[CLI / Windows Launchers: run_amde.py, scripts/*.bat, scripts/*.ps1]
       │
       ▼
[Config & ResourcePolicy: src/amde/config.py, src/amde/runtime/resources.py]
  - Kiểm tra schema config, task_specs, holdout_only boundary, safety limits
  - Dò CPU, RAM (psutil -> GlobalMemoryStatusEx -> /proc/meminfo), Disk, NVIDIA GPU
  - Chỉ bật CUDA nếu torch.cuda.is_available() == True VÀ CUDA tensor smoke test PASS
  - Gắn effective_device + search_rng_protocol vào protocol_hash
       │
       ▼
[Orchestrator & SingleInstanceLock: src/amde/orchestrator.py, src/amde/runtime/lock.py]
  - Khóa đơn phiên (.amde.lock) có phát hiện stale PID/heartbeat
  - Khởi tạo SQLite Ledger (WAL mode hoặc MEMORY fallback trên FS không hỗ trợ fsync)
  - Tạo data_manifest.json & split_manifest.json (xóa ngay mẫu challenge khỏi bộ nhớ)
       │
       ▼
[Baselines: src/amde/baselines/registry.py]
  - Huấn luyện và chấm điểm thật 3 baseline (majority_mean, linear_sgd, mlp_small)
  - Chọn strongest baseline trên tập dev mở (open dev tasks)
       │
       ▼
[Discovery & Search Loop: src/amde/search/*, src/amde/discovery/*]
  - RNG khởi tạo từ search_seed_for(experiment_seed), độc lập hoàn toàn với run_id
  - Portfolio UCB1 + Proposer (random, evolutionary, crossover, novelty, local)
  - Genome JSON kiểm tra chặt theo cấp độ A/B/C/D + DSL allowlist (không eval/exec)
       │
       ├──► [Multi-Fidelity Quick Screening: src/amde/evaluation/evaluator.py]
       │      - Chạy trong tiến trình con độc lập (multiprocessing spawn)
       │      - 3 quick tasks, 1 seed, epochs_scale=0.35, timeout + RSS cap
       │      - Cập nhật Population (2 islands), ParetoArchive, NoveltyArchive (MAP-Elites)
       │
       ├──► [Full Confirmation & Promotion Guard: src/amde/search/selection.py]
       │      - Chọn ứng viên đứng đầu mỗi level để chạy confirm (9 tasks, 3 seeds)
       │      - Promotion Guard: kiểm tra min_effect (0.01), min_families_won (2),
       │        worst_family_delta >= -0.03, và khoảng tin cậy (CI low)
       │
       ├──► [Passive Stagnation Diagnostics: src/amde/search/stagnation.py]
       │      - Tính phenotype diversity, invalid rate, plateau generations
       │      - Chỉ ghi khuyến nghị vào ledger/report; không tự ý sửa metric hay split
       │
       ├──► [Atomic Checkpointing & Heartbeat: src/amde/storage/checkpoint.py]
       │      - Ghi atomic (temp + safe_fsync + os.replace), kiểm tra SHA-256 ngay sau ghi
       │      - Lưu backup theo generation; tự động khôi phục từ backup nếu latest bị hỏng
       │
       └──► [Sealed Challenge Boundary: src/amde/evaluation/sealed.py]
              - Chỉ mở khi đủ số generation (mặc định >= 5) và có incumbent
              - Tiến trình con sealed_worker tự sinh dữ liệu challenge (kể cả shifted tasks)
              - Chỉ trả về thống kê tổng hợp (mean, median, CI, family_scores); xóa mọi per-example/per-seed
```

## 2. Nguồn gốc & Quyết định hợp nhất từ `full.zip`

| Thành viên trong `full.zip` | SHA-256 (12 ký tự đầu) | Quyết định | Lý do kỹ thuật & Bằng chứng |
|---|---|---|---|
| `deepflash.zip` | `490bfcfb0a71` | **Giữ làm nền chính (Base)** | Kiến trúc modular đầy đủ nhất (41 tệp nguồn, 109 test gốc), có process isolation, SQLite ledger, atomic checkpoint, sealed challenge boundary và Windows launchers. |
| `clau.zip` | `3332bb9d5ac7` | **Port có kiểm soát** | Port thiết kế thí nghiệm đối chứng (`scripts/control_experiment.py`) và cơ chế tách họ nhiệm vụ giữ kín (`holdout_only` task family separation trong `config.py` & `orchestrator.py`). Không gộp nguyên khối vì `clau.zip` dùng `inproc` mặc định và thiếu GPU tensor trainer. |
| `antigrav.zip` | `d3f6a6cb32db` | **Port / Viết lại có chọn lọc** | Port ý tưởng chẩn đoán đình trệ thụ động (`src/amde/search/stagnation.py`), bổ sung thông tin tài nguyên vào `heartbeat.json` và báo cáo Markdown. Không lấy phần ledger/runner của `antigrav.zip` vì báo cáo gốc có mâu thuẫn provenance (`baseline_score` nhảy giữa 0.7 và 0.1). |
| `chat.zip` | `55776a3edba5` | **Loại bỏ mã nguồn / Chỉ đối chiếu test** | Toàn bộ engine nằm trong một file `engine.py` lớn (~51 KB) với chỉ 2 họ bài toán synthetic; hẹp hơn nền chính. |
| `omni.zip` | `3ea8629791b8` | **Loại bỏ (Exclude)** | 46/57 tệp Python trùng khít byte-for-byte với `deepflash.zip` (~80.7%), nhưng có 5 unit test fail (vỡ `clip` DSL, lộ `per_task` ở challenge opening, sai số lượng task mặc định) và 2 run artifact lỗi. |
| `wen.tar` | `4ec885255f29` | **Viết lại khái niệm** | Chỉ có 9 module rời rạc, thiếu CLI, orchestrator, evaluator và test. Ý tưởng tính toán tensor cho DSL update rule được viết lại hoàn chỉnh trong `src/amde/models/torch_trainer.py` có kiểm thử parity với CPU reference. |

## 3. Danh mục lỗi & rủi ro đã phát hiện và xử lý (P0 → P3)

| Mức độ | Mã lỗi / Hạng mục | Hiện trạng trước khi sửa (Before) | Nguyên nhân gốc (Root Cause) | Giải pháp đã triển khai (After) | Tệp / Hàm thay đổi | Bằng chứng Test xác nhận |
|---|---|---|---|---|---|---|
| **P0** | Lộ mẫu dữ liệu Challenge trong bộ nhớ tiến trình chính | `build_manifests()` sinh dữ liệu challenge để băm SHA-256 nhưng giữ `task.challenge` trong danh sách `tasks` trả về trước khi bị xóa. | `generate_tasks(config, include_challenge=True)` giữ list `Example` trên từng `SplitData`. | Xóa tường minh `task.challenge = None` ngay sau khi tính hash trong `build_manifests()` và `manifests.pop("tasks", None)` trong `_prepare_protocol()`. | `src/amde/benchmarks/registry.py` (`build_manifests`), `src/amde/orchestrator.py` | `test_manifest_contains_hashes_not_challenge_examples`, `test_explicit_challenge_is_aggregate_only_and_single_use` (**PASS**) |
| **P0** | Tách biệt họ nhiệm vụ giữ kín (Task-family Holdout) | Tất cả `task_specs` trong `config/default.json` đều được đưa vào dev/confirm; chưa có cờ cấu hình để giữ kín một họ nhiệm vụ chỉ dành riêng cho `challenge`. | `Config` chỉ có `task_specs()` chung cho cả dev và challenge. | Thêm `Config.dev_task_specs()` và `Config.challenge_task_specs()`; hỗ trợ `"holdout_only": true` trên từng task spec và cấm đưa task `holdout_only` vào `quick_task_ids`. | `src/amde/config.py`, `src/amde/orchestrator.py` | `test_holdout_only_tasks_are_excluded_from_dev_and_quick` (**PASS**) |
| **P1** | Search RNG phụ thuộc vào `run_id` làm hỏng tính tái lập | Search RNG được seed bằng `seed_from(seed, run_id, "search")`. Hai lần chạy có cùng seed/config nhưng khác tên thư mục (`run_id`) sinh ra chuỗi ứng viên khác nhau. | Trộn định danh thư mục xuất (`run_id`) vào seed toán học của bộ sinh đề xuất. | Tách `search_seed_for(experiment_seed)` chỉ phụ thuộc vào `seed`; tăng `SEARCH_RNG_PROTOCOL_VERSION = "amde-search-rng-v2-seed-only"` trong `protocol_hash`. | `src/amde/util.py`, `src/amde/config.py`, `src/amde/orchestrator.py` | `test_search_seed_depends_on_experiment_seed_not_run_id`, `test_search_reproducibility_is_independent_of_run_id_and_guards_protocol_hash` (**PASS**) |
| **P1** | Crash `OSError: [Errno 95] Operation not supported` và `sqlite3.OperationalError: disk I/O error` trên Virtual/FUSE Filesystem | Khi chạy `smoke`/`once` trên filesystem ảo hóa không hỗ trợ `os.fsync`, `SingleInstanceLock.acquire()`, `atomic_write_bytes()` và SQLite `WAL` commit bị crash ngay lập tức. | Gọi trực tiếp `os.fsync(fd)` không bắt `ENOTSUP`/`EINVAL`, và ép `PRAGMA journal_mode=WAL` trên mount không hỗ trợ đồng bộ descriptor. | Thêm `safe_fsync(fd)` trong `util.py`; thêm `_directory_supports_fsync()` trong `Ledger` để dùng `WAL`/`NORMAL` trên ổ đĩa thật (NTFS/ext4) và tự động fallback sang `MEMORY`/`OFF` trên mount ảo; bảo vệ `ROLLBACK` bằng `in_transaction`. | `src/amde/util.py`, `src/amde/runtime/lock.py`, `src/amde/storage/ledger.py` | Chạy trực tiếp `smoke` và `control_experiment.py` trên `/working_dir` (FUSE) và `/tmp` (ext4) đều **PASS** |
| **P1** | Cache collision giữa CPU và CUDA | Cache key của trial không chứa thiết bị thực thi (`effective_device`), dẫn tới nguy cơ tái sử dụng kết quả CPU khi chạy ở chế độ CUDA hoặc ngược lại. | `evaluation_cache_key()` và `Config._protocol_material()` thiếu trường `training_device`. | Đưa `training_device` vào cả `evaluation_cache_key()` và `Config._protocol_material()`. | `src/amde/evaluation/evaluator.py`, `src/amde/config.py` | `test_cache_key_is_stable_and_protocol_sensitive` (**PASS**) |
| **P2** | Không đọc được RAM trên Linux khi thiếu `psutil` & Heartbeat thiếu thông tin thiết bị/tài nguyên | Khi không cài `psutil` trên Linux, `_mem_info()` trả về `0.0 MB` RAM; `heartbeat.json` chỉ có `status`, `generation`, `trials`. | Chỉ có nhánh `psutil` và `GlobalMemoryStatusEx` (Windows), thiếu `/proc/meminfo`. | Thêm `_linux_proc_meminfo()` (`memory_source="proc-meminfo"`); bổ sung `device_effective`, `gpu_effective`, `cpu_workers` và `resources` snapshot vào `heartbeat.json`. | `src/amde/runtime/resources.py`, `src/amde/orchestrator.py` | `test_hardware_and_resource_sampler_report_memory_and_pressure`, `test_cli_once_status_report_and_inspect` (**PASS**) |
| **P2** | Log phình không giới hạn khi chạy 24/7 | `FileHandler` ghi nối tiếp vô hạn vào `engine.log`; watchdog sinh vô số file log theo thời gian. | Thiếu cơ chế xoay vòng (rotation) và giới hạn số file log. | Dùng `RotatingFileHandler` (`maxBytes=5MB`, `backupCount=5`) cho `engine.log` và giới hạn 40 log gần nhất trong `watchdog_24x7.ps1`. | `src/amde/orchestrator.py`, `scripts/watchdog_24x7.ps1` | `test_engine_log_rotation_bounds_long_run_logs` (**PASS**) |
| **P3** | Báo cáo Markdown thiếu chẩn đoán đình trệ & Thiếu `scripts/report.ps1` | `generate_report()` chưa xuất thông tin chẩn đoán đình trệ (`stagnation`) ra báo cáo Markdown; `README.md` còn một vài dòng mô tả cũ từ bản `0.1.0`; thiếu `scripts/report.ps1`. | Chưa nối dữ liệu `stagnation` từ `summary_json` của generation vào `generate_report()`. | Thêm mục `## Search Diagnostics & Stagnation` vào `generate_report()`, thêm `scripts/report.ps1`, và đồng bộ toàn bộ tài liệu `README.md`. | `src/amde/orchestrator.py`, `scripts/report.ps1`, `README.md` | `test_cli_once_status_report_and_inspect` (**PASS**) |
