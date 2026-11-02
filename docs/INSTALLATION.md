# AMDE Installation Guide

AMDE is designed to run on native Windows 11 and Linux (Ubuntu 22.04+).

---

## 1. System Requirements

* **Operating System**: Native Windows 11 (recommended for Job Object process containment) or Linux.
* **Python**: Python 3.10, 3.11, 3.12, or 3.14.
* **Memory**: Minimum 8 GB RAM (16 GB+ recommended for long endurance runs).
* **Storage**: 2 GB free disk space.
* **GPU (Optional)**: NVIDIA CUDA-capable GPU (tested on NVIDIA Quadro P2000 Mobile, RTX 30/40 series) for accelerated PyTorch statevector simulation.

---

## 2. Standard Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine
```

### Step 2: Create a Virtual Environment
```bash
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Package
```bash
# Minimal core installation (CPU only, standard library + psutil)
pip install .

# Full installation with PyTorch CUDA and Scikit-learn tabular benchmarks
pip install ".[quantum,tabular]"
```

---

## 3. Verifying Installation

Run the environment verification doctor check:
```bash
# Using the amde CLI entrypoint
amde doctor

# Or using the amde2 operational runner
python scripts/amde2.py doctor
```

Expected output:
```
[doctor] platform: Windows-11 / Linux
[doctor] python: 3.x
[doctor] psutil: available
[doctor] job objects: supported
[doctor] cuda: available (or cpu-only mode)
[doctor] status: OK
```
