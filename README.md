# AMDE — Autonomous Machine Learning Discovery Engine

[![Release](https://img.shields.io/badge/release-v2.0.3-blue.svg)](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Experimental Research](https://img.shields.io/badge/Status-Experimental%20Research-orange.svg)](#scientific-claims--limitations)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)

**AMDE (Autonomous Machine Learning Discovery Engine)** is an experimental research platform for investigating autonomous model candidate generation, search intelligence, and hybrid classical/quantum simulation workflows.

> **Research Status**: AMDE is an experimental software platform published for open research, reproduction, and community evaluation. It is not an enterprise AutoML product and does not demonstrate quantum advantage. See [Scientific Claims & Limitations](#scientific-claims--limitations).

---

## Architecture Overview

AMDE employs an out-of-band supervision topology designed for native Windows 11 and Linux environments:

```
+--------------------------------------------------------------------------+
| Guardian Process (Windows Job Object Supervisor)                         |
|   |-- Sub-15ms Hard Deadline Enforcement & Thermal / VRAM Watchdog       |
|   +-- Manager Process (AMDE 2.0 Campaign Orchestrator)                   |
|         |-- Relay Workers & Windows Job Kernel Containment               |
|         |-- Classical Worker (Multi-Island Evolutionary Search)          |
|         |     |-- Level A-C (Hyperparameters & Neural Architectures)     |
|         |     +-- Level D (Symbolic AST Grammar Optimization)            |
|         +-- Quantum Worker (PyTorch Statevector Simulation)              |
|               |-- Variational Quantum Classifiers (4, 6, 8 Qubits)       |
|               +-- Quantum Fidelity Kernel Ridge Regression               |
+--------------------------------------------------------------------------+
```

### Key Engineering Capabilities
* **Strict Hard Deadline Enforcement**: Guaranteed sub-15 ms trigger latency and sub-120 ms process termination overshoot.
* **Process Containment**: 100% elimination of orphaned PIDs and child processes via Windows Job Object kernel limits (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
* **Crash-Resilient State**: Dual SQLite ledgers operating in Write-Ahead Logging (WAL) mode with verified ACID transaction integrity across $> 69,000$ trials.
* **Full-Fidelity Backup & Restore**: Automated ZIP packaging, SHA-256 verification, and sandbox restoration testing.

---

## System Requirements

* **Operating System**: Native Windows 11 (recommended for Job Object containment) or Linux (Ubuntu 22.04+).
* **Python**: Python 3.10, 3.11, 3.12, or 3.14.
* **Memory**: Minimum 8 GB RAM (16 GB recommended).
* **GPU (Optional)**: NVIDIA GPU with CUDA support for accelerated PyTorch statevector simulation.

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux:
source .venv/bin/activate

# 3. Install core dependencies
pip install .

# 4. (Optional) Install quantum simulation & tabular benchmark dependencies
pip install ".[quantum,tabular]"
```

Verify your installation:
```bash
python scripts/amde2.py doctor
```

---

## Quickstart: Running a Discovery Experiment

Run a 2-round classical discovery session on the synthetic XOR benchmark:

```bash
python scripts/research_campaign.py \
    --campaign-id demo_run \
    --rounds 2 \
    --generations 3 \
    --quick-budget 8 \
    --confirm-budget 4 \
    --tasks xor_classification
```

Inspect discovered candidates in the SQLite archive:
```bash
python -c "
import sqlite3
conn = sqlite3.connect('storage/discovery_archive.sqlite3')
cur = conn.cursor()
for row in cur.execute('SELECT candidate_hash, level, best_confirm_score FROM discovery_candidates ORDER BY best_confirm_score DESC LIMIT 5'):
    print(row)
conn.close()
"
```

To run an autonomous campaign with hard deadline enforcement:
```bash
python scripts/amde2.py run-campaign \
    --campaign-id timed_campaign \
    --duration-hours 0.25 \
    --deadline-policy hard \
    --mode classical
```

---

## Running Tests

AMDE uses Python's standard `unittest` framework:

```bash
# Run unit test suite
python -m unittest discover tests/unit

# Run individual test modules
python -m unittest tests/unit/test_config.py
python -m unittest tests/unit/test_amde2_runtime.py
```

---

## Scientific Claims & Limitations

In accordance with open science principles, all operational achievements and research limitations are explicitly disclosed:

### Verified Operational Achievements
1. **Hard Deadline Reliability**: Rigorously qualified across 10-run native-Windows integration suites with zero surviving PIDs.
2. **Endurance Execution**: Completed a continuous 12-hour hybrid campaign (`amde2_12h_endure_20261031`).
3. **Preservation**: The 100-hour campaign was safely halted at T+23.98 hours, completely cleaned up, and archived as `STOPPED_EARLY_BY_OPERATOR`.

### Known Scientific Limitations
1. **Generational Degradation**: In AMDE 2.0.3, fitness peaks at Generation 0 (mean 0.7860) and drops over subsequent generations (0.4073 at Generation 4) due to high unguided mutation rates (80% on learning rate).
2. **Candidate Recycling**: Historical candidate `d3c33842...` was re-injected as a static seed across 25 consecutive rounds without generational improvement. Variations in reported score were due to split variance.
3. **Quantum Configuration Freeze**: Due to an off-by-one boundary condition in config generation, only 3 preset VQC circuits were evaluated during historical runs.
4. **Classical Simulation Backend**: All quantum components run on classical PyTorch tensor statevectors; no physical quantum hardware is utilized.
5. **Benchmark Status**: Multi-task baseline comparison against Bayesian Optimization (TPE) is in progress: `SCIENTIFIC BASELINE VALIDATION IN PROGRESS`.

Detailed analysis: [docs/SCIENTIFIC_LIMITATIONS.md](docs/SCIENTIFIC_LIMITATIONS.md)

---

## Documentation

* [Architecture Overview](docs/ARCHITECTURE.md)
* [Installation Guide](docs/INSTALLATION.md)
* [Quickstart Guide](docs/QUICKSTART.md)
* [CLI Reference](docs/CLI_REFERENCE.md)
* [Hard Deadline & Process Containment](docs/DEADLINE_AND_PROCESS_MANAGEMENT.md)
* [Data Storage & WAL Mode](docs/DATA_STORAGE.md)
* [Scientific Limitations Statement](docs/SCIENTIFIC_LIMITATIONS.md)
* [Security & Isolation Model](docs/SECURITY_MODEL.md)
* [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## Contributing

We welcome contributions from researchers and engineers. Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

For security reports, please refer to [SECURITY.md](SECURITY.md).

---

## Citation

If you use AMDE in your research, please cite it using [CITATION.cff](CITATION.cff):

```bibtex
@software{amde_2026,
  author = {AMDE Project Maintainers},
  title = {AMDE: Autonomous Machine Learning Discovery Engine},
  version = {2.0.3},
  year = {2026},
  url = {https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine}
}
```

---

## License

AMDE is open-source software licensed under the [MIT License](LICENSE). Third-party dependencies are acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Hướng dẫn Tiếng Việt (Vietnamese Guide)

### Giới thiệu
AMDE (Autonomous Machine Learning Discovery Engine) là nền tảng nghiên cứu thực nghiệm dành cho việc tìm kiếm mô hình học máy tự động, đánh giá thuật toán tiến hóa (genetic algorithms) và mô phỏng mạch lượng tử biến thiên (VQC) trên nền tảng PyTorch GPU.

### Điểm nổi bật về mặt kỹ thuật
1. **Kiểm soát tiến trình nghiêm ngặt**: Sử dụng Windows Job Object của kernel Windows NT để đảm bảo khi kết thúc chiến dịch, 100% các tiến trình con (child, grandchild PIDs) đều bị dừng hoàn toàn, không để lại tiến trình rác.
2. **Thực thi đúng hạn (Hard Deadline)**: Độ trễ kích hoạt dừng dưới 15ms khi chạm mốc thời gian quy định.
3. **Minh bạch khoa học**: Công khai đầy đủ các giới hạn hiện tại của thuật toán tìm kiếm (suy giảm độ thích nghi qua các thế hệ do đột biến ngẫu nhiên quá cao) để làm tiền đề nâng cấp cho phiên bản tiếp theo.

Xem chi tiết hướng dẫn cài đặt và sử dụng tại [docs/QUICKSTART.md](docs/QUICKSTART.md).
