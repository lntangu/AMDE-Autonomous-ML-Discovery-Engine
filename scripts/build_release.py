#!/usr/bin/env python3
"""Build, verify, and package AMDE 2.0.3 clean distribution release."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "runs",
    "artifacts",
    "reports",
    "storage",
    "dist",
    "build",
}

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".zip"}

EXCLUDE_FILES = {
    ".git",
    "SHA256SUMS.txt",
    ".DS_Store",
    "Thumbs.db",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_release_files(root: Path) -> list[Path]:
    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude directories in-place to prevent os.walk from recursing
        rel_dir = Path(dirpath).relative_to(root)
        parts = rel_dir.parts
        if any(part in EXCLUDE_DIRS for part in parts):
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in EXCLUDE_EXTS or filename in EXCLUDE_FILES:
                continue
            file_path = Path(dirpath) / filename
            if file_path.is_file() and not file_path.is_symlink():
                collected.append(file_path)

    collected.sort(key=lambda p: str(p.relative_to(root)).replace("\\", "/"))
    return collected


def main() -> int:
    print(f"Building AMDE 2.0.3 Release from: {ROOT}")
    files = collect_release_files(ROOT)
    print(f"Collected {len(files)} clean distribution files.")

    manifest_path = ROOT / "SHA256SUMS.txt"
    manifest_lines = []
    for f in files:
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        digest = sha256(f)
        manifest_lines.append(f"{digest}  {rel}\n")

    manifest_path.write_text("".join(manifest_lines), encoding="utf-8", newline="\n")
    print(f"Updated {manifest_path} ({len(manifest_lines)} entries).")

    # Verify manifest using verify_manifest.py
    verify_cmd = [sys.executable, str(ROOT / "scripts" / "verify_manifest.py")]
    proc = subprocess.run(verify_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAIL: verify_manifest.py failed:\n{proc.stderr}\n{proc.stdout}", file=sys.stderr)
        return 1
    print(f"PASS: {proc.stdout.strip()}")

    # Package dist/AMDE_2.0.3_Release.zip
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dist_dir / "AMDE_2.0.3_Release.zip"

    all_zip_files = files + [manifest_path]
    print(f"Writing release zip archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in all_zip_files:
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            # Normalize file permissions and dates
            zinfo = zipfile.ZipInfo(rel)
            zinfo.external_attr = 0o644 << 16  # standard read/write permissions
            if f.suffix in {".ps1", ".bat", ".sh"} or (f.parent.name == "scripts" and f.suffix == ".py"):
                zinfo.external_attr = 0o755 << 16  # executable
            zinfo.date_time = (2026, 10, 31, 14, 0, 0)
            with f.open("rb") as src_handle:
                zf.writestr(zinfo, src_handle.read())

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    zip_digest = sha256(zip_path)
    print(f"Created {zip_path.name}: {zip_size_mb:.2f} MiB | SHA-256: {zip_digest}")

    # Write dist checksum
    (dist_dir / "AMDE_2.0.3_Release.zip.sha256").write_text(
        f"{zip_digest}  {zip_path.name}\n", encoding="utf-8"
    )

    # Test clean room extraction and verification
    print("Verifying clean extraction...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_root)

        verify_clean = subprocess.run(
            [sys.executable, str(temp_root / "scripts" / "verify_manifest.py")],
            cwd=temp_root,
            capture_output=True,
            text=True,
        )
        if verify_clean.returncode != 0:
            print(f"FAIL: Extracted release failed verification:\n{verify_clean.stderr}\n{verify_clean.stdout}")
            return 1
        print(f"PASS: Extracted release verified: {verify_clean.stdout.strip()}")

        # Run unit tests in extracted directory
        test_files = [
            "tests/unit/test_amde2_runtime.py",
            "tests/unit/test_amde2_job.py",
            "tests/unit/test_backward_compatibility.py",
            "tests/unit/test_amde2_forced_termination.py",
        ]
        for tf in test_files:
            test_cmd = [sys.executable, tf]
            test_proc = subprocess.run(test_cmd, cwd=temp_root, capture_output=True, text=True)
            if test_proc.returncode != 0:
                print(f"FAIL: Unit test {tf} in extracted release failed:\n{test_proc.stderr}\n{test_proc.stdout}")
                return 1
            print(f"PASS: Extracted release unit test: {tf}")

    print("\nSUCCESS: AMDE 2.0.3 Release Packaging Completed and Verified!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
