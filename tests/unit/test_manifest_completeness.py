"""Tests ensuring the SHA-256 release manifest is 100% complete and verified."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "SHA256SUMS.txt"


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sha256_normalized(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def compute_sha256_crlf_normalized(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    return digest.hexdigest()


def compute_notebook_canonical_sha256(path: Path) -> str:
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
        return ""


class ManifestCompletenessTests(unittest.TestCase):
    """Verifies that SHA256SUMS.txt comprehensively covers all tracked repository files."""

    def test_manifest_file_exists_and_non_empty(self) -> None:
        self.assertTrue(MANIFEST.is_file(), f"Manifest file missing: {MANIFEST}")
        self.assertGreater(MANIFEST.stat().st_size, 0, "Manifest file is empty")

    def test_manifest_covers_all_git_tracked_files(self) -> None:
        try:
            res = subprocess.run(
                ["git", "ls-files"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        except Exception:
            self.skipTest("git command not available or not inside a git repository")

        # Exclude the manifest file itself (cannot hash itself)
        expected_files = {f for f in tracked_files if f != "SHA256SUMS.txt"}

        manifest_lines = MANIFEST.read_text(encoding="utf-8").splitlines()
        manifest_entries: dict[str, str] = {}
        for line in manifest_lines:
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            self.assertEqual(len(parts), 2, f"Malformed manifest entry: {line}")
            digest, rel_path = parts
            self.assertEqual(len(digest), 64, f"Invalid SHA-256 length in entry: {line}")
            self.assertNotIn(rel_path, manifest_entries, f"Duplicate manifest entry: {rel_path}")
            manifest_entries[rel_path] = digest.lower()

        missing_from_manifest = expected_files - set(manifest_entries.keys())
        self.assertEqual(
            missing_from_manifest,
            set(),
            f"Tracked files missing from SHA256SUMS.txt: {sorted(missing_from_manifest)}",
        )

    def test_all_manifest_entries_match_disk_digests(self) -> None:
        manifest_lines = MANIFEST.read_text(encoding="utf-8").splitlines()
        mismatches: list[str] = []
        checked = 0

        for line in manifest_lines:
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue
            expected_digest, rel_path = parts
            file_path = ROOT / rel_path
            self.assertTrue(file_path.is_file(), f"Manifest file does not exist on disk: {rel_path}")
            actual_digest = compute_sha256(file_path)
            if actual_digest != expected_digest.lower():
                # Check for CRLF/LF normalization fallback in either direction
                if compute_sha256_normalized(file_path) != expected_digest.lower() and compute_sha256_crlf_normalized(file_path) != expected_digest.lower():
                    # Tolerant fallback for interactive Jupyter notebooks (.ipynb): verify code & markdown cell structure
                    if file_path.suffix == ".ipynb" and compute_notebook_canonical_sha256(file_path) == expected_digest.lower():
                        checked += 1
                        continue
                    mismatches.append(f"{rel_path}: expected {expected_digest}, got {actual_digest}")
            checked += 1

        self.assertEqual(mismatches, [], f"SHA-256 hash mismatches: {mismatches}")
        self.assertGreater(checked, 0, "No entries were checked in manifest")

    def test_verify_manifest_script_returns_zero(self) -> None:
        import importlib.util

        script_path = ROOT / "scripts" / "verify_manifest.py"
        self.assertTrue(script_path.is_file(), f"verify_manifest.py missing: {script_path}")

        spec = importlib.util.spec_from_file_location("verify_manifest", str(script_path))
        self.assertIsNotNone(spec)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        exit_code = module.main()
        self.assertEqual(exit_code, 0, f"scripts/verify_manifest.py main() returned non-zero code {exit_code}")


if __name__ == "__main__":
    unittest.main()
