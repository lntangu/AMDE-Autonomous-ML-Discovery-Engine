"""Generates SHA256SUMS.txt covering all Git-tracked repository files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "SHA256SUMS.txt"


BINARY_EXTENSIONS = {".zip", ".png", ".pdf", ".whl", ".gz", ".tar.gz", ".db", ".sqlite3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(path: Path) -> str:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return sha256(path)
    # Canonical POSIX LF line ending hash for all text assets
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def notebook_canonical_sha256(path: Path) -> str:
    """Computes a deterministic hash of an interactive Jupyter notebook's code & markdown cells.

    This ensures that dynamic execution outputs (execution counts, plots, run timestamps)
    produced during interactive Google Colab execution do not invalidate cryptographic
    manifest integrity.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get("cells", []):
            c_type = cell.get("cell_type")
            raw_source = cell.get("source", "")
            if isinstance(raw_source, list):
                source = "".join(raw_source)
            else:
                source = str(raw_source)
            source = source.replace("\r\n", "\n").strip()
            # Ignore auto-prepended colab view-in-github badge cell
            if "view-in-github" in str(cell.get("metadata", {})) or "colab-badge.svg" in source:
                continue
            cells.append({"cell_type": c_type, "source": source})
        canonical_json = json.dumps(cells, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    except Exception:
        return sha256(path)


def main() -> int:
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]

    # Exclude SHA256SUMS.txt itself
    files_to_hash = [f for f in tracked_files if f != "SHA256SUMS.txt"]
    files_to_hash.sort()

    lines: list[str] = []
    for rel in files_to_hash:
        path = ROOT / rel
        if not path.is_file():
            print(f"ERROR: Tracked file missing on disk: {rel}", file=sys.stderr)
            return 1
        if path.suffix == ".ipynb":
            digest = notebook_canonical_sha256(path)
        else:
            digest = sha256_canonical(path)
        lines.append(f"{digest}  {rel}\n")

    MANIFEST_PATH.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"Successfully generated {MANIFEST_PATH} with {len(lines)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
