"""Persistent, cross-campaign discovery archive.

The archive stores canonical genome identity and append-only sightings. Exact
identity is authoritative; optional behavioural fingerprints are diagnostics
only and never cause an automatic rejection of a candidate.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from ..util import canonical_json, ensure_dir, utc_now
from .genome import Genome


class DiscoveryArchive:
    def __init__(self, path: str | Path, *, timeout_seconds: float = 15.0,
                 max_candidates: int = 100_000, max_sightings: int = 1_000_000) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_candidates = max(1, int(max_candidates))
        self.max_sightings = max(self.max_candidates, int(max_sightings))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def _connection(self):
        con = self._connect()
        try:
            yield con
        finally:
            con.close()

    def _initialize(self) -> None:
        with self._connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=FULL")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS discovery_candidates (
                    candidate_hash TEXT PRIMARY KEY,
                    canonical_genome_json TEXT NOT NULL,
                    level TEXT NOT NULL,
                    complexity REAL NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    sightings INTEGER NOT NULL DEFAULT 1,
                    best_quick_score REAL,
                    best_confirm_score REAL,
                    behavioural_signature TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_discovery_behaviour
                    ON discovery_candidates(behavioural_signature);
                CREATE TABLE IF NOT EXISTS discovery_sightings (
                    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    operator TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    score REAL,
                    protocol_hash TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(candidate_hash) REFERENCES discovery_candidates(candidate_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_sightings_run ON discovery_sightings(run_id, generation);
                """
            )

    def observe(
        self,
        genome: Genome,
        *,
        run_id: str,
        generation: int,
        operator: str,
        stage: str = "quick",
        score: float | None = None,
        protocol_hash: str | None = None,
        behavioural_signature: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not str(run_id).strip() or len(str(run_id)) > 128:
            raise ValueError("run_id must contain 1-128 non-whitespace characters")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("generation must be an integer")
        if not str(operator).strip() or len(str(operator)) > 96:
            raise ValueError("operator must contain 1-96 non-whitespace characters")
        if stage not in {"quick", "confirm"}:
            raise ValueError("stage must be quick or confirm")
        score_value = None if score is None else float(score)
        if score_value is not None and not math.isfinite(score_value):
            raise ValueError("score must be finite")
        if protocol_hash is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(protocol_hash)):
            raise ValueError("protocol_hash must be a 64-character SHA-256 hex digest")
        if behavioural_signature is not None and (not isinstance(behavioural_signature, str) or len(behavioural_signature) > 256):
            raise ValueError("behavioural_signature must be a string of at most 256 characters")
        digest = genome.hash
        payload = genome.canonical_json()
        now = utc_now()
        metadata_json = canonical_json(metadata or {})
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                "SELECT canonical_genome_json, best_quick_score, best_confirm_score, sightings "
                "FROM discovery_candidates WHERE candidate_hash=?", (digest,)
            ).fetchone()
            inserted = existing is None
            if existing is not None and existing["canonical_genome_json"] != payload:
                raise RuntimeError("SHA-256 collision detected for distinct canonical genomes")
            sighting_row = con.execute(
                "SELECT seq FROM sqlite_sequence WHERE name='discovery_sightings'"
            ).fetchone()
            sighting_count = int(sighting_row[0]) if sighting_row else 0
            candidate_capacity_reached = False
            if inserted:
                candidate_count = int(con.execute(
                    "SELECT COUNT(*) FROM discovery_candidates"
                ).fetchone()[0])
                candidate_capacity_reached = candidate_count >= self.max_candidates
            if candidate_capacity_reached or sighting_count >= self.max_sightings:
                return {"candidate_hash": digest, "inserted": False, "archived": False,
                        "reason": "global discovery archive capacity reached", "sightings": 0,
                        "behavioural_matches": []}
            if inserted:
                con.execute(
                    """INSERT INTO discovery_candidates(
                        candidate_hash, canonical_genome_json, level, complexity,
                        first_seen_at, last_seen_at, sightings, best_quick_score,
                        best_confirm_score, behavioural_signature, metadata_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (digest, payload, genome.data["level"], genome.complexity(),
                     now, now, 1, score_value if stage == "quick" else None,
                     score_value if stage == "confirm" else None,
                     behavioural_signature, metadata_json),
                )
            else:
                quick = existing["best_quick_score"]
                confirm = existing["best_confirm_score"]
                if stage == "quick" and score_value is not None:
                    quick = score_value if quick is None else max(float(quick), score_value)
                if stage == "confirm" and score_value is not None:
                    confirm = score_value if confirm is None else max(float(confirm), score_value)
                con.execute(
                    """UPDATE discovery_candidates SET last_seen_at=?, sightings=sightings+1,
                       best_quick_score=?, best_confirm_score=?,
                       behavioural_signature=COALESCE(behavioural_signature, ?),
                       metadata_json=? WHERE candidate_hash=?""",
                    (now, quick, confirm, behavioural_signature, metadata_json, digest),
                )
            con.execute(
                """INSERT INTO discovery_sightings(
                    candidate_hash, run_id, generation, operator, stage, score,
                    protocol_hash, created_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (digest, str(run_id), int(generation), str(operator), str(stage),
                 score_value, protocol_hash, now),
            )
            count = con.execute(
                "SELECT sightings FROM discovery_candidates WHERE candidate_hash=?", (digest,)
            ).fetchone()[0]
            con.commit()
        matches = self.behaviour_matches(behavioural_signature) if behavioural_signature else []
        return {"candidate_hash": digest, "inserted": inserted, "archived": True,
                "sightings": int(count),
                "behavioural_matches": [item for item in matches if item["candidate_hash"] != digest]}

    def behaviour_matches(self, signature: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
        if not signature:
            return []
        with self._connection() as con:
            rows = con.execute(
                "SELECT candidate_hash, level, complexity, sightings FROM discovery_candidates "
                "WHERE behavioural_signature=? ORDER BY sightings DESC, candidate_hash LIMIT ?",
                (str(signature), max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, candidate_hash: str) -> dict[str, Any] | None:
        with self._connection() as con:
            row = con.execute(
                "SELECT * FROM discovery_candidates WHERE candidate_hash=?", (candidate_hash,)
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["genome"] = json.loads(value.pop("canonical_genome_json"))
        value["metadata"] = json.loads(value.pop("metadata_json"))
        return value

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT candidate_hash, level, complexity, first_seen_at, last_seen_at, sightings, "
                "best_quick_score, best_confirm_score FROM discovery_candidates "
                "ORDER BY last_seen_at DESC, candidate_hash LIMIT ? OFFSET ?",
                (max(1, min(10000, int(limit))), max(0, int(offset))),
            ).fetchall()
        return [dict(row) for row in rows]

    def warm_start_genomes(self, *, limit: int = 4) -> list[dict[str, Any]]:
        """Return recent diverse candidates without comparing cross-protocol scores.

        Historical scores are not comparable across different benchmark manifests
        or protocols. They remain audit metadata only; warm-start ranking uses
        recency and exact identity, and the receiving campaign must re-evaluate.
        """
        with self._connection() as con:
            rows = con.execute(
                "SELECT canonical_genome_json FROM discovery_candidates "
                "ORDER BY last_seen_at DESC, candidate_hash LIMIT ?",
                (max(0, min(64, int(limit))),),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connection() as con:
            candidates = int(con.execute("SELECT COUNT(*) FROM discovery_candidates").fetchone()[0])
            sightings = int(con.execute("SELECT COUNT(*) FROM discovery_sightings").fetchone()[0])
            runs = int(con.execute("SELECT COUNT(DISTINCT run_id) FROM discovery_sightings").fetchone()[0])
        return {"candidate_count": candidates, "sighting_count": sightings,
                "campaign_count": runs, "max_candidates": self.max_candidates,
                "max_sightings": self.max_sightings,
                "capacity_remaining_candidates": max(0, self.max_candidates - candidates),
                "capacity_remaining_sightings": max(0, self.max_sightings - sightings),
                "database": str(self.path)}
