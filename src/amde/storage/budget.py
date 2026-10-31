"""Bounded storage accounting for long-running AMDE runs.

The guard never deletes experiment evidence. It requests a resumable pause while
there is still headroom for a final checkpoint and diagnostic records.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable


def tree_size_bytes(path: Path) -> int:
    """Return the size of regular files below *path*, without following symlinks."""
    root = Path(path)
    if not root.exists() or root.is_symlink():
        return 0
    total = 0
    if root.is_file():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    try:
        for item in root.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            try:
                total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


class RunStorageBudget:
    """Measure the combined run/artefact footprint and signal when to pause.

    The check is deliberately non-destructive. ``pause_fraction`` leaves room
    for atomic checkpoint replacement, SQLite WAL cleanup and final reports.
    ``check_interval_seconds`` limits recursive scans during long experiments;
    callers can force a scan at generation boundaries.
    """

    def __init__(
        self, roots: Iterable[Path], *, max_total_bytes: int,
        pause_fraction: float = 0.80, check_interval_seconds: float = 10.0,
    ) -> None:
        self.roots = self._normalize_roots(roots)
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.pause_fraction = min(0.95, max(0.50, float(pause_fraction)))
        self.check_interval_seconds = max(0.0, float(check_interval_seconds))
        self._last_check = 0.0
        self._last_result: dict[str, int | bool | float] | None = None

    @staticmethod
    def _normalize_roots(roots: Iterable[Path]) -> list[Path]:
        resolved = sorted({Path(p).resolve() for p in roots}, key=lambda p: len(p.parts))
        kept: list[Path] = []
        for candidate in resolved:
            if any(candidate == existing or existing in candidate.parents for existing in kept):
                continue
            kept.append(candidate)
        return kept

    @property
    def pause_at_bytes(self) -> int:
        return int(self.max_total_bytes * self.pause_fraction)

    def check(self, *, force: bool = False) -> dict[str, int | bool | float]:
        now = time.monotonic()
        if (not force and self._last_result is not None and
                now - self._last_check < self.check_interval_seconds):
            return dict(self._last_result)
        total = sum(tree_size_bytes(root) for root in self.roots)
        result: dict[str, int | bool | float] = {
            "bytes": total,
            "max_bytes": self.max_total_bytes,
            "pause_at_bytes": self.pause_at_bytes,
            "fraction": total / self.max_total_bytes,
            "over_budget": total >= self.pause_at_bytes,
        }
        self._last_check = now
        self._last_result = dict(result)
        return result
