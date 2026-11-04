# AMDE 4.0 — INSTALLATION & DEPENDENCY GUIDE
**Distribution Version:** 4.0.2  
**Python Compatibility:** Python 3.10, 3.11, 3.12, 3.13, 3.14  

---

## 1. Prerequisites

AMDE 4.0 requires standard scientific Python libraries:
- `python >= 3.10`
- `numpy >= 1.24`
- `scipy >= 1.10`
- `scikit-learn >= 1.3`
- `psutil >= 5.9`

---

## 2. Standard Installation

### 2.1 From Pre-Built Wheel (Recommended)
Download `amde-4.0.2-py3-none-any.whl` from the GitHub Release assets:
```bash
pip install amde-4.0.2-py3-none-any.whl
```

### 2.2 From Source Distribution
```bash
pip install amde-4.0.2.tar.gz
```

### 2.3 From Cloned Source Repository
```bash
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine
pip install .
```

---

## 3. Verifying Installation

Verify that AMDE 4.0.2 is correctly installed:
```bash
python3 -c "import amde; print('AMDE Version:', amde.__version__)"
# Output: AMDE Version: 4.0.2
```

Run the built-in smoke verification:
```bash
python3 -c "
import amde
from amde import AMDE4Engine
from pathlib import Path
import numpy as np

X = np.random.randn(50, 4)
y = (X[:, 0] > 0).astype(int)

engine = AMDE4Engine(n_features=4, seed=42)
res = engine.search(X[:30], y[:30], X[30:], y[30:], dataset_name='verify', budget=3)
print('Incumbent Accuracy:', res['incumbent_score'])
print('Pareto Frontier Size:', res['pareto_size'])
"
```
