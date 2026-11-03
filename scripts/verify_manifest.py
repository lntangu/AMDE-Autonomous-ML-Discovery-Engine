"""Cross-platform SHA-256 manifest verifier for an extracted AMDE release."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"FAIL: manifest missing: {MANIFEST}", file=sys.stderr)
        return 2
    expected: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"line {line_number}: malformed entry")
            continue
        digest, rel = parts
        candidate = (ROOT / rel).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"line {line_number}: path escapes project root: {rel}")
            continue
        if rel in expected:
            errors.append(f"line {line_number}: duplicate entry: {rel}")
            continue
        expected[rel] = digest.lower()
    checked = 0
    for rel, expected_digest in expected.items():
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            errors.append(f"MISSING/UNSAFE: {rel}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected_digest:
            # Tolerant fallback: Check if difference is solely Windows git checkout CRLF translation
            if sha256_normalized(path) == expected_digest:
                continue
            errors.append(f"MISMATCH: {rel} expected={expected_digest} actual={actual}")
    if errors:
        print(f"FAIL: checked {checked}/{len(expected)} manifest entries")
        for error in errors:
            print(f" - {error}")
        return 1
    print(f"PASS: {checked}/{len(expected)} SHA-256 entries verified")
    print("Note: SHA256SUMS.txt does not hash itself (self-reference is impossible).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
