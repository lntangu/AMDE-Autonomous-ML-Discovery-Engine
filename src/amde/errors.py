"""Exception taxonomy used across AMDE.

Errors are classified so that candidate failures are never accidentally
counted as evaluator failures, and infrastructure failures are never counted
as a low candidate score.
"""

from __future__ import annotations


class AMDEBaseError(Exception):
    """Base class for all AMDE errors."""


class ConfigError(AMDEBaseError):
    """Invalid or incompatible configuration."""


class ValidationError(AMDEBaseError):
    """A candidate, artifact, or protocol failed validation."""

    def __init__(self, message: str, *, code: str = "validation_error") -> None:
        super().__init__(message)
        self.code = code


class DSLValidationError(ValidationError):
    """A candidate DSL program is unsafe, malformed, or too large."""


class CandidateNumericError(AMDEBaseError):
    """A candidate generated NaN/Inf or otherwise invalid numeric output."""


class CandidateTimeoutError(AMDEBaseError):
    """A candidate exceeded its evaluation budget."""


class CandidateMemoryError(AMDEBaseError):
    """A candidate exceeded the per-trial memory budget."""


class EvaluatorProtocolError(AMDEBaseError):
    """An evaluator returned a malformed or inconsistent result."""


class InfrastructureError(AMDEBaseError):
    """The evaluator or another infrastructure component failed."""


class SealedProtocolError(AMDEBaseError):
    """The sealed challenge protocol was violated or exhausted."""


class LockError(AMDEBaseError):
    """Single-instance lock could not be acquired or verified."""


class CheckpointError(AMDEBaseError):
    """Checkpoint is corrupt, incompatible, or cannot be loaded."""


class ResourcePressure(AMDEBaseError):
    """A configured resource reserve would be violated."""


class StopRequested(AMDEBaseError):
    """A graceful stop was requested."""
