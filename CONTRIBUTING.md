# Contributing to AMDE

Thank you for your interest in contributing to the Autonomous Machine Learning Discovery Engine (AMDE)!

AMDE is an experimental research platform. We welcome contributions that improve algorithmic search intelligence, process reliability, cross-platform compatibility, test coverage, and scientific documentation.

---

## 1. Development Principles

1. **Evidence-Based Engineering**: Any claim of algorithmic improvement must be supported by empirical comparisons using identical random seeds, dataset splits, and evaluation budgets.
2. **Strict Reliability & Containment**: Changes to execution or process management must preserve process termination guarantees (no orphaned PIDs) and SQLite transaction integrity.
3. **Transparency & Honesty**: Clearly distinguish classical simulations from physical quantum hardware, and validation performance from true generalizability.

---

## 2. Setting Up Local Development

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Native Windows 11 or Linux environment
- Optional: CUDA 12+ for GPU statevector acceleration

### Installation
```bash
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## 3. Running Tests

AMDE uses Python's standard `unittest` framework to avoid external test harness lock-in.

```bash
# Run all unit tests
python -m unittest discover tests/unit

# Run specific test modules
python -m unittest tests/unit/test_config.py
python -m unittest tests/unit/test_amde2_runtime.py
```

---

## 4. Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, modular Python following PEP 8.
3. Include unit tests covering new logic or bug fixes.
4. Verify that all unit tests pass before opening a Pull Request.
5. Provide a clear summary of changes, motivation, and verification steps in your PR description.
