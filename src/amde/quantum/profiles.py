"""Adaptive compute profiles and hardware-aware resource governor for AMDE Quantum & Hybrid engines.

Implements benchmarkable compute profiles (BASELINE, MEDIUM, HIGH, ADAPTIVE) with
strict provenance tracking, thermal/memory hysteresis, and hardware-specific safety limits.
"""
from __future__ import annotations

import copy
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .vqc import VQCConfig
from ..util import canonical_json, hash_json, utc_now


@dataclass(frozen=True)
class ComputeProfile:
    """Immutable workload definition for Quantum/Hybrid ML experimentation."""
    name: str
    protocol_id: str
    n_qubits: int
    n_layers: int
    batch_size: int
    epochs: int
    train_size: int
    val_size: int
    test_size: int
    max_configs: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol_id": self.protocol_id,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "max_configs": self.max_configs,
            "description": self.description,
            "profile_hash": hash_json({
                "name": self.name,
                "protocol_id": self.protocol_id,
                "n_qubits": self.n_qubits,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "train_size": self.train_size,
                "val_size": self.val_size,
                "test_size": self.test_size,
            }),
        }


# Canonical Compute Profiles
PROFILE_BASELINE = ComputeProfile(
    name="BASELINE",
    protocol_id="amde-quantum-baseline-v1",
    n_qubits=8,
    n_layers=2,
    batch_size=16,
    epochs=6,
    train_size=64,
    val_size=32,
    test_size=64,
    max_configs=3,
    description="Exact original baseline configuration; low vector saturation on Quadro P2000.",
)

PROFILE_MEDIUM = ComputeProfile(
    name="MEDIUM",
    protocol_id="amde-quantum-medium-v1",
    n_qubits=8,
    n_layers=2,
    batch_size=64,
    epochs=12,
    train_size=256,
    val_size=64,
    test_size=128,
    max_configs=6,
    description="Balanced profile with 4x batch size and 4x train size for increased vector saturation.",
)

PROFILE_HIGH = ComputeProfile(
    name="HIGH",
    protocol_id="amde-quantum-high-v1",
    n_qubits=10,
    n_layers=2,
    batch_size=128,
    epochs=16,
    train_size=512,
    val_size=128,
    test_size=256,
    max_configs=10,
    description="Maximum throughput profile utilizing up to 1024 statevector dimensions and 128 batch size.",
)

PROFILE_CATALOG: dict[str, ComputeProfile] = {
    "BASELINE": PROFILE_BASELINE,
    "MEDIUM": PROFILE_MEDIUM,
    "HIGH": PROFILE_HIGH,
}


@dataclass
class GovernorLimits:
    """Configurable safety thresholds for system resources."""
    max_ram_percent: float = 85.0
    min_free_ram_gb: float = 4.0
    max_vram_mb: float = 3500.0
    min_free_vram_mb: float = 1000.0
    max_gpu_temperature_c: float = 78.0
    temp_hysteresis_c: float = 5.0
    vram_hysteresis_mb: float = 500.0


class AdaptiveComputeGovernor:
    """Monitors live hardware telemetry and selects optimal workload parameters with hysteresis."""

    def __init__(self, limits: GovernorLimits | None = None) -> None:
        self.limits = limits or GovernorLimits()
        self.is_throttled = False
        self.throttle_reason: str | None = None
        self.history: list[dict[str, Any]] = []

    def evaluate_telemetry(
        self,
        *,
        gpu_temp_c: float | None = None,
        free_vram_mb: float | None = None,
        ram_available_gb: float | None = None,
        ram_percent: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate resource metrics and determine throttle or normal state using hysteresis."""
        state_changed = False

        # Check conditions triggering throttle
        trigger_reasons = []
        if gpu_temp_c is not None and gpu_temp_c >= self.limits.max_gpu_temperature_c:
            trigger_reasons.append(f"GPU temperature ({gpu_temp_c:.1f}C) >= limit ({self.limits.max_gpu_temperature_c:.1f}C)")
        if free_vram_mb is not None and free_vram_mb < self.limits.min_free_vram_mb:
            trigger_reasons.append(f"Free VRAM ({free_vram_mb:.0f}MB) < limit ({self.limits.min_free_vram_mb:.0f}MB)")
        if ram_available_gb is not None and ram_available_gb < self.limits.min_free_ram_gb:
            trigger_reasons.append(f"Available RAM ({ram_available_gb:.1f}GB) < limit ({self.limits.min_free_ram_gb:.1f}GB)")
        if ram_percent is not None and ram_percent >= self.limits.max_ram_percent:
            trigger_reasons.append(f"RAM usage ({ram_percent:.1f}%) >= limit ({self.limits.max_ram_percent:.1f}%)")

        if trigger_reasons:
            if not self.is_throttled:
                self.is_throttled = True
                state_changed = True
            self.throttle_reason = "; ".join(trigger_reasons)
        elif self.is_throttled:
            # Check recovery conditions with hysteresis
            can_recover = True
            if gpu_temp_c is not None and gpu_temp_c > (self.limits.max_gpu_temperature_c - self.limits.temp_hysteresis_c):
                can_recover = False
            if free_vram_mb is not None and free_vram_mb < (self.limits.min_free_vram_mb + self.limits.vram_hysteresis_mb):
                can_recover = False
            if ram_available_gb is not None and ram_available_gb < (self.limits.min_free_ram_gb + 1.0):
                can_recover = False
            if can_recover:
                self.is_throttled = False
                self.throttle_reason = None
                state_changed = True

        status = {
            "at": utc_now(),
            "is_throttled": self.is_throttled,
            "state_changed": state_changed,
            "reason": self.throttle_reason,
            "gpu_temperature_c": gpu_temp_c,
            "free_vram_mb": free_vram_mb,
            "ram_available_gb": ram_available_gb,
        }
        self.history.append(status)
        if len(self.history) > 100:
            self.history.pop(0)
        return status

    def select_adaptive_profile(
        self,
        *,
        gpu_temp_c: float | None = None,
        free_vram_mb: float | None = None,
        ram_available_gb: float | None = None,
    ) -> ComputeProfile:
        """Select the appropriate compute profile given live hardware state."""
        self.evaluate_telemetry(
            gpu_temp_c=gpu_temp_c,
            free_vram_mb=free_vram_mb,
            ram_available_gb=ram_available_gb,
        )
        if self.is_throttled:
            # Fall back to minimal baseline to shed thermal and memory load
            return PROFILE_BASELINE

        # Normal operation: Select based on available capacity
        free_vram = free_vram_mb if free_vram_mb is not None else 3500.0
        temp = gpu_temp_c if gpu_temp_c is not None else 50.0

        if free_vram >= 2500.0 and temp < 70.0:
            return PROFILE_HIGH
        elif free_vram >= 1500.0 and temp < 74.0:
            return PROFILE_MEDIUM
        return PROFILE_BASELINE
