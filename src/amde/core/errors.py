"""AMDE 4.0 Error Taxonomy and Typed Exception Hierarchy."""

from __future__ import annotations


class AMDEError(Exception):
    """Base exception for all AMDE 4.0 errors."""
    pass


class ValidationError(AMDEError):
    """Raised when an experiment spec, configuration, or parameter schema is invalid."""
    pass


class SearchSpaceError(AMDEError):
    """Raised when candidate genome violates constraints or cannot be constructed."""
    pass


class EvaluationError(AMDEError):
    """Raised when candidate training or evaluation fails unexpectedly."""
    pass


class EvaluationTimeoutError(EvaluationError):
    """Raised when an evaluation trial exceeds its allocated wall-clock budget."""
    pass


class ResourceExhaustedError(AMDEError):
    """Raised when an experiment exhausts its allocated resource quota."""
    pass


class LedgerError(AMDEError):
    """Raised when the evidence ledger fails to persist, recover, or verify an event."""
    pass


class IncompatibleCacheError(AMDEError):
    """Raised when an evaluation cache lookup encounters mismatched splits, seeds, or fidelity."""
    pass
