"""Quantum and quantum-inspired research tools for AMDE.

The included implementation is a classical statevector simulation layer. It
can run on PyTorch CPU or CUDA but does not connect to physical QPU hardware.
"""

QUANTUM_SIMULATOR_VERSION = "amde-statevector-v1"
VQC_PROTOCOL_VERSION = "amde-vqc-angle-ansatz-v1"
MAZE_PROTOCOL_VERSION = "amde-grover-maze-query-model-v1"

__all__ = ["QUANTUM_SIMULATOR_VERSION", "VQC_PROTOCOL_VERSION", "MAZE_PROTOCOL_VERSION"]
