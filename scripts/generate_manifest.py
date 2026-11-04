"""Generates SHA256SUMS.txt covering all Git-tracked repository files."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        digest = sha256(path)
        lines.append(f"{digest}  {rel}\n")

    MANIFEST_PATH.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"Successfully generated {MANIFEST_PATH} with {len(lines)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
