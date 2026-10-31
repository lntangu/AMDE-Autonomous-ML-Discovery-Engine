"""Small, dependency-free utilities shared by the engine."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

CANONICAL_JSON_VERSION = "amde-canonical-json-v2-string-normalized-keys"

_FSYNC_IGNORED_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.EBADF,
        getattr(errno, "ENOTSUP", 95),
        getattr(errno, "EOPNOTSUPP", 95),
        getattr(errno, "ENOSYS", 38),
    }
)


def safe_fsync(fd: int) -> bool:
    """Flush file descriptor to storage; ignore unsupported virtual/network FS errors."""
    try:
        os.fsync(fd)
        return True
    except OSError as exc:
        if exc.errno in _FSYNC_IGNORED_ERRNOS:
            return False
        raise


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json_value(value: Any) -> Any:
    """Normalize JSON-compatible values before deterministic serialization.

    JSON object keys are strings. Python's encoder sorts integer keys as
    integers before converting them to strings, while a JSON round-trip sorts
    those same keys as strings. That distinction used to make hashes of maps
    with integer keys (notably per-seed metric maps) unstable after reload.
    Normalize keys recursively before sorting and reject stringification
    collisions rather than silently overwriting data.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                text_key = key
            elif isinstance(key, bool):
                text_key = "true" if key else "false"
            elif isinstance(key, int):
                text_key = str(key)
            elif isinstance(key, float):
                if not math.isfinite(key):
                    raise ValueError("non-finite JSON object key")
                text_key = json.dumps(key, allow_nan=False, separators=(",", ":"))
            elif key is None:
                text_key = "null"
            else:
                raise TypeError(f"unsupported JSON object key type: {type(key).__name__}")
            if text_key in normalized:
                raise ValueError(f"JSON object keys collide after string normalization: {text_key!r}")
            normalized[text_key] = _canonical_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Canonical JSON used for hashes, manifests, checkpoints and seeds.

    Object keys are normalized to strings before lexicographic sorting, so
    hashes remain stable across a JSON write/read round-trip. NaN/Infinity are
    intentionally rejected because they make numerical evidence ambiguous.
    """
    normalized = _canonical_json_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def seed_from(*parts: Any) -> int:
    """Derive a stable 63-bit seed from arbitrary JSON-serializable parts."""

    material = canonical_json(list(parts))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def search_seed_for(experiment_seed: int) -> int:
    """Derive the search RNG seed from the experiment seed, never the run ID.

    A run ID identifies an output directory, not a scientific randomization
    factor. Excluding it makes repeated runs with the same configured seed
    reproducible and makes paired strategy comparisons interpretable.
    """

    return seed_from(int(experiment_seed), "search")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def robust_replace(src: Path, dst: Path, *, max_attempts: int = 25, initial_delay: float = 0.01) -> None:
    """Atomically replace a file with bounded retry on Windows lock collisions.

    On Windows (especially NTFS), os.replace requires delete access on the destination.
    If another process (e.g. status reader, antivirus, backup) has the destination
    momentarily open, os.replace raises PermissionError (WinError 5) or OSError (WinError 32).
    A bounded retry loop with exponential backoff and jitter makes the replace reliable
    under concurrent read/write patterns.
    """
    src = Path(src)
    dst = Path(dst)
    for attempt in range(max_attempts):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError):
            if attempt == max_attempts - 1:
                raise
            delay = initial_delay * (1.2 ** attempt) + random.uniform(0.002, 0.015)
            time.sleep(delay)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically using a same-directory temp file then robust rename."""

    ensure_dir(path.parent)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            safe_fsync(handle.fileno())
        robust_replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def read_json_safe(path: Path, default: Any = None, *, max_attempts: int = 5) -> Any:
    """Read a JSON file, retrying transient sharing violations on Windows before falling back to default."""
    path = Path(path)
    for attempt in range(max_attempts):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (PermissionError, OSError):
            if attempt == max_attempts - 1 or not path.exists():
                return default
            time.sleep(0.01 * (attempt + 1))
        except (ValueError, json.JSONDecodeError):
            return default
    return default


def read_json(path: Path) -> Any:
    """Read a JSON file with retry on transient Windows file sharing locks."""
    path = Path(path)
    for attempt in range(5):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (PermissionError, OSError) as exc:
            if attempt == 4 or not path.exists():
                raise
            time.sleep(0.01 * (attempt + 1))
    return None


def append_jsonl(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    line = canonical_json(value) + "\n"
    # One writer per process is enforced by the engine lock. A short retry
    # handles transient Windows sharing errors from status/log readers.
    for attempt in range(5):
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                safe_fsync(handle.fileno())
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""

    result: dict[str, Any] = {k: _copy_value(v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = _copy_value(value)
    return result


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _copy_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_value(v) for v in value]
    return value


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(sec)}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


def safe_relpath(root: Path, candidate: Path) -> Path:
    """Resolve a path and require it to remain inside root."""

    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {candidate}") from exc
    return resolved


def sanitize_id(text: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:96] or fallback


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def safe_process_start_time(pid: int) -> float | None:
    """Return process creation time when psutil is available (best effort)."""

    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        return float(proc.create_time())
    except Exception:
        return None
