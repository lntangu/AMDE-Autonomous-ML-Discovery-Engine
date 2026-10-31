"""Utility-level invariants."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from amde.util import (
    atomic_write_json,
    canonical_json,
    chunked,
    deep_merge,
    hash_json,
    human_duration,
    read_json,
    safe_relpath,
    sanitize_id,
    seed_from,
    search_seed_for,
    stdev,
)


class UtilTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self) -> None:
        self.assertEqual(canonical_json({"b": 1, "a": 2}), canonical_json({"a": 2, "b": 1}))
        self.assertEqual(hash_json({"b": 1, "a": 2}), hash_json({"a": 2, "b": 1}))

    def test_canonical_json_normalizes_numeric_object_keys_for_json_roundtrip(self) -> None:
        import json
        value = {599415630: {1056575843: 0.25}, "name": "seed-map"}
        canonical = canonical_json(value)
        roundtripped = json.loads(canonical)
        self.assertEqual(canonical_json(roundtripped), canonical)
        self.assertEqual(roundtripped["599415630"]["1056575843"], 0.25)

    def test_canonical_json_rejects_colliding_normalized_keys(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({1: "numeric", "1": "text"})

    def test_canonical_json_rejects_nan_and_infinity(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({"x": math.nan})
        with self.assertRaises(ValueError):
            canonical_json({"x": math.inf})

    def test_seed_from_is_deterministic_and_type_sensitive(self) -> None:
        first = seed_from(42, "run", "search")
        self.assertEqual(first, seed_from(42, "run", "search"))
        self.assertNotEqual(first, seed_from(42, "run", "other"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**63)

    def test_search_seed_depends_on_experiment_seed_not_run_id(self) -> None:
        # Run IDs are output labels and must not alter the experiment random stream.
        self.assertEqual(search_seed_for(11), search_seed_for(11))
        self.assertNotEqual(search_seed_for(11), search_seed_for(12))

    def test_sanitize_id_removes_unsafe_characters(self) -> None:
        self.assertEqual(sanitize_id("run/with spaces?!"), "run_with_spaces")
        self.assertEqual(sanitize_id("  ../weird run id?!  "), "weird_run_id")
        self.assertEqual(sanitize_id(""), "run")
        self.assertEqual(sanitize_id("a" * 200), "a" * 96)

    def test_atomic_write_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "value.json"
            atomic_write_json(path, {"b": [1, 2], "a": "x"})
            self.assertEqual(read_json(path), {"a": "x", "b": [1, 2]})
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}, "b": [1]}
        override = {"a": {"y": 2}}
        merged = deep_merge(base, override)
        self.assertEqual(merged, {"a": {"x": 1, "y": 2}, "b": [1]})
        self.assertEqual(base, {"a": {"x": 1}, "b": [1]})
        merged["a"]["x"] = 99
        self.assertEqual(base["a"]["x"], 1)

    def test_safe_relpath_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(safe_relpath(root, root / "inside.txt").parent, root.resolve())
            with self.assertRaises(ValueError):
                safe_relpath(root, root.parent / "outside.txt")

    def test_human_duration_and_chunking(self) -> None:
        self.assertEqual(human_duration(2.5), "2.5s")
        self.assertEqual(human_duration(65), "1m 5s")
        self.assertEqual(human_duration(3660), "1h 1m")
        self.assertEqual(list(chunked([1, 2, 3, 4], 2)), [[1, 2], [3, 4]])
        self.assertEqual(stdev([1.0]), 0.0)

    def test_hash_json_changes_with_content(self) -> None:
        self.assertNotEqual(hash_json({"a": 1}), hash_json({"a": 2}))


if __name__ == "__main__":
    unittest.main()
