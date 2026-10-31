"""Atomic, checksummed checkpoint state with verified backup recovery."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ..errors import CheckpointError
from ..util import atomic_write_json, ensure_dir, hash_json, read_json, utc_now

CHECKPOINT_SCHEMA = "1.0.0"


class CheckpointManager:
    def __init__(
        self, run_dir: Path, ledger: Any | None = None, run_id: str | None = None,
        *, max_backups: int = 5,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.max_backups = max(1, int(max_backups))
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self.backup_dir = ensure_dir(self.checkpoint_dir / "backups")
        self.latest = self.checkpoint_dir / "latest.json"
        self.ledger = ledger
        self.run_id = run_id

    def save(self, state: dict[str, Any], *, generation: int, status: str = "saved") -> dict[str, Any]:
        # Freeze the exact object before hashing. Hashing the caller's live
        # mutable state and deep-copying it afterwards leaves a race window:
        # concurrent bookkeeping can mutate a nested value between those two
        # operations, producing a checkpoint whose checksum cannot verify.
        state_snapshot = copy.deepcopy(state)
        wrapped = {
            "schema_version": CHECKPOINT_SCHEMA,
            "created_at": utc_now(),
            "generation": int(generation),
            "checksum": hash_json(state_snapshot),
            "state": state_snapshot,
        }
        if self.latest.exists():
            try:
                previous = read_json(self.latest)
                prev_gen = int(previous.get("generation", 0))
                backup = self.backup_dir / f"gen_{prev_gen:08d}_{int(time.time() * 1000)}.json"
                if not backup.exists():
                    atomic_write_json(backup, previous)
            except Exception:
                # Keep running: the new checkpoint is still valid, and corrupt
                # old files are not silently deleted.
                pass
        atomic_write_json(self.latest, wrapped)
        # Verify immediately after the atomic rename.
        written = read_json(self.latest)
        if written.get("checksum") != hash_json(written.get("state", {})):
            raise CheckpointError("checkpoint checksum verification failed immediately after write")
        self._prune_backups()
        if self.ledger is not None and self.run_id is not None:
            self.ledger.save_checkpoint_metadata(
                run_id=self.run_id,
                generation=int(generation),
                path=str(self.latest),
                checksum=written["checksum"],
                status=status,
            )
        return {
            "path": str(self.latest),
            "checksum": written["checksum"],
            "generation": int(generation),
            "status": status,
        }

    def _prune_backups(self) -> None:
        """Keep a bounded tail of backup checkpoints; latest.json stays untouched."""
        backups = []
        for path in self.backup_dir.glob("gen_*.json"):
            try:
                if path.is_file() and not path.is_symlink():
                    backups.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
        backups.sort(key=lambda item: item[0], reverse=True)
        for _, path in backups[self.max_backups:]:
            try:
                path.unlink()
            except OSError:
                # Retention must not invalidate a checkpoint that was just saved.
                continue

    def load(self) -> dict[str, Any] | None:
        candidates: list[Path] = []
        if self.latest.exists():
            candidates.append(self.latest)
        if self.backup_dir.exists():
            candidates.extend(
                sorted(self.backup_dir.glob("gen_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
            )
        errors: list[str] = []
        for path in candidates:
            try:
                wrapped = read_json(path)
                if not isinstance(wrapped, dict) or "state" not in wrapped or "checksum" not in wrapped:
                    raise CheckpointError(f"checkpoint {path} has an invalid envelope")
                actual = hash_json(wrapped["state"])
                if actual != wrapped["checksum"]:
                    raise CheckpointError(f"checkpoint {path} checksum mismatch")
                if wrapped.get("schema_version") != CHECKPOINT_SCHEMA:
                    raise CheckpointError(f"checkpoint {path} has unsupported schema")
                state = copy.deepcopy(wrapped["state"])
                state["_checkpoint_meta"] = {
                    "path": str(path),
                    "checksum": wrapped["checksum"],
                    "created_at": wrapped.get("created_at"),
                    "generation": wrapped.get("generation"),
                    "recovered_from_backup": path != self.latest,
                }
                return state
            except Exception as exc:
                errors.append(str(exc))
                continue
        if errors:
            raise CheckpointError("no valid checkpoint could be loaded; " + "; ".join(errors[:3]))
        return None

    def list_checksums(self) -> list[dict[str, Any]]:
        result = []
        for path in [self.latest, *sorted(self.backup_dir.glob("gen_*.json"))]:
            if not path.exists():
                continue
            try:
                wrapped = read_json(path)
                result.append(
                    {
                        "path": str(path),
                        "generation": wrapped.get("generation"),
                        "checksum": wrapped.get("checksum"),
                        "valid": wrapped.get("checksum") == hash_json(wrapped.get("state", {})),
                    }
                )
            except Exception as exc:
                result.append({"path": str(path), "valid": False, "error": str(exc)})
        return result
