"""Content-addressed artifacts with checksums and a disk budget."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import InfrastructureError
from ..util import atomic_write_bytes, canonical_json, ensure_dir
from .ledger import Ledger


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    relative_path: str
    size_bytes: int
    kind: str
    stored: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "stored": self.stored,
            "reason": self.reason,
        }


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        ledger: Ledger | None = None,
        *,
        run_id: str | None = None,
        max_total_bytes: int = 2 * 1024**3,
        max_artifact_bytes: int = 1 * 1024**2,
    ) -> None:
        self.root = Path(root)
        ensure_dir(self.root)
        self.ledger = ledger
        self.run_id = run_id
        self.max_total_bytes = int(max_total_bytes)
        self.max_artifact_bytes = int(max_artifact_bytes)

    def _total_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != ".quota":
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def put_json(self, value: Any, *, kind: str) -> ArtifactRef:
        data = (canonical_json(value) + "\n").encode("utf-8")
        return self.put_bytes(data, kind=kind, extension=".json")

    def put_bytes(self, data: bytes, *, kind: str, extension: str = ".bin") -> ArtifactRef:
        sha = hashlib.sha256(data).hexdigest()
        if len(data) > self.max_artifact_bytes:
            return ArtifactRef(
                sha256=sha,
                relative_path="",
                size_bytes=len(data),
                kind=kind,
                stored=False,
                reason=f"artifact exceeds per-file limit ({len(data)} > {self.max_artifact_bytes})",
            )
        current = self._total_bytes()
        if current + len(data) > self.max_total_bytes:
            return ArtifactRef(
                sha256=sha,
                relative_path="",
                size_bytes=len(data),
                kind=kind,
                stored=False,
                reason=f"artifact store quota would be exceeded ({current + len(data)} bytes)",
            )
        shard = sha[:2]
        relative = Path(shard) / f"{sha}{extension}"
        target = self.root / relative
        if not target.exists():
            ensure_dir(target.parent)
            atomic_write_bytes(target, data)
        else:
            # Verify an existing artifact rather than silently trusting it.
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != sha:
                raise InfrastructureError(f"artifact hash mismatch for {target}")
        if self.ledger is not None and self.run_id is not None:
            self.ledger.record_artifact(
                run_id=self.run_id,
                sha256=sha,
                relative_path=str(relative).replace("\\", "/"),
                size_bytes=len(data),
                kind=kind,
            )
        return ArtifactRef(sha, str(relative).replace("\\", "/"), len(data), kind)

    def verify(self, relative_path: str, sha256: str) -> bool:
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            return False
        if not target.exists():
            return False
        return hashlib.sha256(target.read_bytes()).hexdigest() == sha256

    def read_json(self, relative_path: str, sha256: str | None = None) -> Any:
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise InfrastructureError("artifact path escapes store") from exc
        data = target.read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise InfrastructureError(f"artifact checksum mismatch: {relative_path}")
        import json

        return json.loads(data.decode("utf-8"))

    def stats(self) -> dict[str, Any]:
        return {"bytes": self._total_bytes(), "root": str(self.root)}

    def free_bytes(self) -> int:
        usage = shutil.disk_usage(self.root)
        return int(usage.free)
