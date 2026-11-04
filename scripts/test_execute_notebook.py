"""Headless validation test runner for AMDE_Real_World_Demo.ipynb."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
plt.ioff()

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "AMDE_Real_World_Demo.ipynb"

def test_notebook():
    print(f"Validating notebook format: {NOTEBOOK_PATH}")
    nb = nbf.read(str(NOTEBOOK_PATH), as_version=4)
    print(f"Notebook loaded successfully! Total cells: {len(nb.cells)}")

    # Isolate environment namespace
    global_ns = {"__name__": "__main__"}

    # Execute code cells in sequence
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"Total code cells to execute: {len(code_cells)}")

    for idx, cell in enumerate(code_cells, 1):
        source = cell.source
        # Skip colab pip installation line in local runner and mock display
        lines = [line for line in source.splitlines() if not line.strip().startswith("!") and not line.strip().startswith("display(")]
        cleaned_source = "\n".join(lines)
        print(f"Executing Code Cell [{idx}/{len(code_cells)}] ...", end=" ", flush=True)
        try:
            exec(cleaned_source, global_ns)
            print("OK")
        except Exception as exc:
            print(f"FAILED!\nError in Cell {idx}: {exc}")
            raise exc

    print("\nALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY WITHOUT ERROR!")

if __name__ == "__main__":
    test_notebook()
