"""Cross-platform SHA-256 manifest verifier for an extracted AMDE release."""
from __future__ import annotations

import hashlib
import json
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


def notebook_canonical_sha256(path: Path) -> str:
    """Computes a deterministic hash of an interactive Jupyter notebook's code & markdown cells."""
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
        return ""


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
            # Tolerant fallback 1: Check if difference is solely Windows git checkout CRLF translation
            if sha256_normalized(path) == expected_digest:
                continue
            # Tolerant fallback 2: For interactive Jupyter notebooks (.ipynb), verify code and markdown cell structure
            if path.suffix == ".ipynb" and notebook_canonical_sha256(path) == expected_digest:
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
