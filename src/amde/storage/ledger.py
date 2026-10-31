"""Transactional SQLite experiment ledger.

Each run has its own database. WAL mode allows status readers while a single
engine process is writing. Trials are inserted as `running` before evaluation
and finalized afterward, so a crash is visible and resumable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..errors import InfrastructureError
from ..util import canonical_json, ensure_dir, hash_json, safe_fsync, utc_now

SCHEMA_VERSION = 2

MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                protocol_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                manifest_hash TEXT,
                ledger_path TEXT,
                notes TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                genome_json TEXT NOT NULL,
                level TEXT NOT NULL,
                complexity REAL NOT NULL,
                generation INTEGER NOT NULL,
                parent_hash TEXT,
                operator TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, candidate_hash),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                candidate_hash TEXT NOT NULL,
                stage TEXT NOT NULL,
                baseline TEXT,
                status TEXT NOT NULL,
                error_type TEXT,
                error_code TEXT,
                message TEXT,
                seeds_json TEXT,
                task_ids_json TEXT,
                aggregate_json TEXT,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                cache_key TEXT,
                protocol_hash TEXT NOT NULL,
                manifest_hash TEXT,
                metric_hash TEXT,
                epochs_scale REAL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                elapsed_seconds REAL,
                peak_rss_mb REAL,
                exit_code INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_trials_candidate ON trials(run_id, candidate_hash, stage)",
            "CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(run_id, status)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trials_ok_cache ON trials(cache_key) WHERE status='ok' AND cache_key IS NOT NULL",
            """
            CREATE TABLE IF NOT EXISTS baselines (
                baseline_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                protocol_hash TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                aggregate_json TEXT,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                seeds_json TEXT,
                metric_version TEXT,
                metric_hash TEXT,
                elapsed_seconds REAL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, name, protocol_hash, manifest_hash),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generations (
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                proposals INTEGER,
                valid INTEGER,
                invalid INTEGER,
                confirmations INTEGER,
                promotions INTEGER,
                best_hash TEXT,
                best_score REAL,
                baseline_score REAL,
                plateau INTEGER,
                strategy_json TEXT,
                summary_json TEXT,
                PRIMARY KEY (run_id, generation),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                candidate_hash TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS archive (
                archive_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                candidate_hash TEXT,
                reason TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS challenge_runs (
                opening_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                opening_index INTEGER NOT NULL,
                cohort TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                candidate_aggregate_json TEXT NOT NULL,
                baseline_name TEXT,
                baseline_aggregate_json TEXT,
                protocol_hash TEXT NOT NULL,
                metric_hash TEXT NOT NULL,
                seeds_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                notes TEXT,
                UNIQUE(run_id, opening_index),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                event TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, sha256),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                sample_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """,
        ],
    ),
    (
        2,
        [
            # A trial ledger records every evaluation attempt. Multiple proposals can
            # legitimately converge on the same cache identity in one batch; a
            # UNIQUE partial index turns the second successful completion into a
            # run-fatal IntegrityError. Coalesce requests at orchestration level,
            # and keep this index non-unique as a final integrity safeguard.
            "DROP INDEX IF EXISTS idx_trials_ok_cache",
            "CREATE INDEX IF NOT EXISTS idx_trials_cache_status ON trials(cache_key, status, created_at) WHERE cache_key IS NOT NULL",
        ],
    ),
]


def _directory_supports_fsync(directory: Path) -> bool:
    probe = directory / f".amde_fsync_probe_{os.getpid()}.tmp"
    try:
        with probe.open("wb") as handle:
            handle.write(b"0")
            handle.flush()
            return safe_fsync(handle.fileno())
    except OSError:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fsync_supported = _directory_supports_fsync(self.path.parent)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        self._closed = False
        try:
            self.connection.row_factory = sqlite3.Row
            if self.fsync_supported:
                try:
                    self.connection.execute("PRAGMA journal_mode=WAL")
                    self.connection.execute("PRAGMA synchronous=NORMAL")
                except sqlite3.OperationalError:
                    self.connection.execute("PRAGMA journal_mode=MEMORY")
                    self.connection.execute("PRAGMA synchronous=OFF")
            else:
                # On virtual/FUSE mounts that reject fsync (Errno 95), WAL/NORMAL
                # causes SQLite commits to fail with disk I/O error.
                self.connection.execute("PRAGMA journal_mode=MEMORY")
                self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout=30000")
            self._migrate()
        except BaseException:
            # A partially initialized Ledger must not leave the database file
            # locked. This matters in particular on Windows, where an open
            # SQLite handle prevents temporary-run cleanup and archive moves.
            self.close()
            raise

    def close(self) -> None:
        """Close the SQLite handle exactly once.

        Do not silently swallow close errors: a still-open connection can
        prevent cleanup on Windows and should remain visible to the caller.
        """
        if self._closed:
            return
        self.connection.close()
        self._closed = True

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        # IMMEDIATE reduces writer races and keeps multi-row state changes
        # atomic. The engine is single-writer; status readers use separate
        # connections and WAL.
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                try:
                    self.connection.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
            raise

    def _migrate(self) -> None:
        applied = {
            int(row["version"])
            for row in self.connection.execute("SELECT version FROM schema_migrations").fetchall()
        } if self._has_table("schema_migrations") else set()
        for version, statements in MIGRATIONS:
            if version in applied:
                continue
            with self.transaction() as conn:
                for statement in statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )
        current = self.connection.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        if current is None or int(current["v"]) != SCHEMA_VERSION:
            raise InfrastructureError(
                f"ledger schema is {current['v'] if current else None}, expected {SCHEMA_VERSION}"
            )

    def _has_table(self, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def upsert_run(
        self,
        *,
        run_id: str,
        mode: str,
        engine_version: str,
        config_hash: str,
        protocol_hash: str,
        config_json: dict[str, Any],
        manifest_hash: str | None = None,
        status: str = "initialized",
    ) -> None:
        now = utc_now()
        with self.transaction() as conn:
            existing = conn.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO runs(run_id, mode, status, created_at, updated_at, engine_version,
                       config_hash, protocol_hash, config_json, manifest_hash, ledger_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        mode,
                        status,
                        now,
                        now,
                        engine_version,
                        config_hash,
                        protocol_hash,
                        canonical_json(config_json),
                        manifest_hash,
                        str(self.path),
                    ),
                )
            else:
                conn.execute(
                    """UPDATE runs SET status=?, updated_at=?, manifest_hash=COALESCE(?, manifest_hash)
                       WHERE run_id=?""",
                    (status, now, manifest_hash, run_id),
                )

    def set_run_status(self, run_id: str, status: str, notes: str | None = None) -> None:
        with self.transaction() as conn:
            if notes is None:
                conn.execute(
                    "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                    (status, utc_now(), run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET status=?, updated_at=?, notes=? WHERE run_id=?",
                    (status, utc_now(), notes, run_id),
                )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Candidates / trials
    # ------------------------------------------------------------------
    def save_candidate(
        self,
        *,
        run_id: str,
        candidate_hash: str,
        genome: dict[str, Any],
        level: str,
        complexity: float,
        generation: int,
        parent_hash: str | None,
        operator: str | None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO candidates
                   (run_id, candidate_hash, genome_json, level, complexity, generation,
                    parent_hash, operator, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    candidate_hash,
                    canonical_json(genome),
                    level,
                    float(complexity),
                    int(generation),
                    parent_hash,
                    operator,
                    utc_now(),
                ),
            )

    def begin_trial(
        self,
        *,
        run_id: str,
        trial_id: str,
        generation: int,
        candidate_hash: str,
        stage: str,
        baseline: str | None,
        seeds: list[int],
        task_ids: list[str],
        cache_key: str,
        protocol_hash: str,
        manifest_hash: str | None,
        metric_hash: str,
        epochs_scale: float,
    ) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO trials(
                    trial_id, run_id, generation, candidate_hash, stage, baseline, status,
                    seeds_json, task_ids_json, cache_key, protocol_hash, manifest_hash,
                    metric_hash, epochs_scale, started_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trial_id,
                    run_id,
                    int(generation),
                    candidate_hash,
                    stage,
                    baseline,
                    canonical_json(list(seeds)),
                    canonical_json(list(task_ids)),
                    cache_key,
                    protocol_hash,
                    manifest_hash,
                    metric_hash,
                    float(epochs_scale),
                    now,
                    now,
                ),
            )

    def finish_trial(
        self,
        *,
        trial_id: str,
        status: str,
        aggregate: dict[str, Any] | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        message: str | None = None,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        elapsed_seconds: float | None = None,
        peak_rss_mb: float | None = None,
        exit_code: int | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """UPDATE trials SET status=?, aggregate_json=?, error_type=?, error_code=?,
                   message=?, artifact_path=?, artifact_sha256=?, ended_at=?, elapsed_seconds=?,
                   peak_rss_mb=?, exit_code=? WHERE trial_id=?""",
                (
                    status,
                    canonical_json(aggregate) if aggregate is not None else None,
                    error_type,
                    error_code,
                    (message or "")[:4000],
                    artifact_path,
                    artifact_sha256,
                    utc_now(),
                    elapsed_seconds,
                    peak_rss_mb,
                    exit_code,
                    trial_id,
                ),
            )

    def find_successful_trial(self, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM trials WHERE cache_key=? AND status='ok' ORDER BY created_at DESC LIMIT 1",
            (cache_key,),
        ).fetchone()
        return self._row_with_json(row) if row else None

    def mark_interrupted_trials(self, run_id: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                """UPDATE trials SET status='interrupted', ended_at=?, error_type='infrastructure',
                   error_code='interrupted', message='engine stopped before trial completed'
                   WHERE run_id=? AND status='running'""",
                (utc_now(), run_id),
            )
            return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    def save_baseline(
        self,
        *,
        run_id: str,
        name: str,
        protocol_hash: str,
        manifest_hash: str,
        status: str,
        aggregate: dict[str, Any] | None,
        seeds: list[int],
        metric_version: str,
        metric_hash: str,
        elapsed_seconds: float | None,
        artifact_path: str | None,
        artifact_sha256: str | None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO baselines(
                    baseline_id, run_id, name, protocol_hash, manifest_hash, status, aggregate_json,
                    artifact_path, artifact_sha256, seeds_json, metric_version, metric_hash,
                    elapsed_seconds, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, name, protocol_hash, manifest_hash) DO UPDATE SET
                    status=excluded.status, aggregate_json=excluded.aggregate_json,
                    artifact_path=excluded.artifact_path, artifact_sha256=excluded.artifact_sha256,
                    elapsed_seconds=excluded.elapsed_seconds, seeds_json=excluded.seeds_json
                """,
                (
                    uuid.uuid4().hex,
                    run_id,
                    name,
                    protocol_hash,
                    manifest_hash,
                    status,
                    canonical_json(aggregate) if aggregate is not None else None,
                    artifact_path,
                    artifact_sha256,
                    canonical_json(seeds),
                    metric_version,
                    metric_hash,
                    elapsed_seconds,
                    utc_now(),
                ),
            )

    def get_baseline(self, name: str, protocol_hash: str, manifest_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT * FROM baselines WHERE name=? AND protocol_hash=? AND manifest_hash=?
               AND status='ok' ORDER BY created_at DESC LIMIT 1""",
            (name, protocol_hash, manifest_hash),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["aggregate"] = json.loads(result.pop("aggregate_json")) if result.get("aggregate_json") else None
        result["seeds"] = json.loads(result.pop("seeds_json")) if result.get("seeds_json") else []
        return result

    def list_baselines(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM baselines WHERE run_id=? ORDER BY name", (run_id,)
        ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["aggregate"] = json.loads(item.pop("aggregate_json")) if item.get("aggregate_json") else None
            item["seeds"] = json.loads(item.pop("seeds_json")) if item.get("seeds_json") else []
            output.append(item)
        return output

    # ------------------------------------------------------------------
    # Generations / decisions / archives / challenges
    # ------------------------------------------------------------------
    def save_generation(self, run_id: str, record: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO generations(
                    run_id, generation, started_at, ended_at, status, proposals, valid, invalid,
                    confirmations, promotions, best_hash, best_score, baseline_score, plateau,
                    strategy_json, summary_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, generation) DO UPDATE SET
                    ended_at=excluded.ended_at, status=excluded.status, proposals=excluded.proposals,
                    valid=excluded.valid, invalid=excluded.invalid, confirmations=excluded.confirmations,
                    promotions=excluded.promotions, best_hash=excluded.best_hash,
                    best_score=excluded.best_score, baseline_score=excluded.baseline_score,
                    plateau=excluded.plateau, strategy_json=excluded.strategy_json,
                    summary_json=excluded.summary_json
                """,
                (
                    run_id,
                    int(record["generation"]),
                    record.get("started_at", utc_now()),
                    record.get("ended_at", utc_now()),
                    record.get("status", "completed"),
                    int(record.get("proposals", 0)),
                    int(record.get("valid", 0)),
                    int(record.get("invalid", 0)),
                    int(record.get("confirmations", 0)),
                    int(record.get("promotions", 0)),
                    record.get("best_hash"),
                    record.get("best_score"),
                    record.get("baseline_score"),
                    int(record.get("plateau", 0)),
                    canonical_json(record.get("strategy", {})),
                    canonical_json(record.get("summary", {})),
                ),
            )

    def save_decision(
        self,
        *,
        run_id: str,
        generation: int,
        candidate_hash: str,
        decision: str,
        reason: str,
        evidence: dict[str, Any],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO decisions(decision_id, run_id, generation, candidate_hash, decision,
                   reason, evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    run_id,
                    int(generation),
                    candidate_hash,
                    decision,
                    reason,
                    canonical_json(evidence),
                    utc_now(),
                ),
            )

    def add_archive(
        self,
        *,
        run_id: str,
        kind: str,
        candidate_hash: str | None,
        reason: str,
        payload: dict[str, Any],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO archive(archive_id, run_id, kind, candidate_hash, reason, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (uuid.uuid4().hex, run_id, kind, candidate_hash, reason, canonical_json(payload), utc_now()),
            )

    def count_challenge_openings(self, run_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS n FROM challenge_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return int(row["n"])

    def latest_challenge_opening(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM challenge_runs WHERE run_id=? ORDER BY opening_index DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["candidate_aggregate"] = (
            json.loads(item.pop("candidate_aggregate_json"))
            if item.get("candidate_aggregate_json")
            else None
        )
        item["baseline_aggregate"] = (
            json.loads(item.pop("baseline_aggregate_json"))
            if item.get("baseline_aggregate_json")
            else None
        )
        item["seeds"] = json.loads(item.pop("seeds_json")) if item.get("seeds_json") else []
        return item

    def save_challenge_opening(self, record: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO challenge_runs(
                    opening_id, run_id, opening_index, cohort, candidate_hash,
                    candidate_aggregate_json, baseline_name, baseline_aggregate_json,
                    protocol_hash, metric_hash, seeds_json, created_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    record["run_id"],
                    int(record["opening_index"]),
                    record["cohort"],
                    record["candidate_hash"],
                    canonical_json(record["candidate_aggregate"]),
                    record.get("baseline_name"),
                    canonical_json(record["baseline_aggregate"]) if record.get("baseline_aggregate") else None,
                    record["protocol_hash"],
                    record["metric_hash"],
                    canonical_json(record.get("seeds", [])),
                    utc_now(),
                    record.get("notes", ""),
                ),
            )

    # ------------------------------------------------------------------
    # Artifacts / checkpoints / events
    # ------------------------------------------------------------------
    def record_artifact(self, *, run_id: str, sha256: str, relative_path: str, size_bytes: int, kind: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM artifacts WHERE run_id=? AND sha256=?", (run_id, sha256)
        ).fetchone()
        if row:
            return
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO artifacts(run_id, sha256, relative_path, size_bytes, kind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, sha256, relative_path, int(size_bytes), kind, utc_now()),
            )

    def artifact_stats(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS total FROM artifacts WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return {"count": int(row["n"]), "bytes": int(row["total"])}

    def save_checkpoint_metadata(
        self, *, run_id: str, generation: int, path: str, checksum: str, status: str
    ) -> None:
        self.connection.execute(
            """INSERT INTO checkpoints(checkpoint_id, run_id, generation, path, checksum, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex, run_id, int(generation), path, checksum, status, utc_now()),
        )

    def latest_checkpoint_metadata(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def record_event(
        self, run_id: str, event: str, *, level: str = "INFO", payload: dict[str, Any] | None = None
    ) -> None:
        self.connection.execute(
            "INSERT INTO events(run_id, ts, level, event, payload_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, utc_now(), level, event, canonical_json(payload or {})),
        )

    def record_resource_sample(self, run_id: str, sample: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO resource_samples(run_id, ts, sample_json) VALUES (?, ?, ?)",
            (run_id, utc_now(), canonical_json(sample)),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _row_with_json(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("aggregate_json", "seeds_json", "task_ids_json"):
            if key in item and item[key] is not None:
                target = key.replace("_json", "")
                item[target] = json.loads(item[key])
        return item

    def tabled_rows(self, table: str, run_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        allowed = {"trials", "baselines", "generations", "decisions", "archive", "events", "challenge_runs"}
        if table not in allowed:
            raise ValueError(f"table not allow-listed: {table}")
        rows = self.connection.execute(
            f"SELECT * FROM {table} WHERE run_id=? ORDER BY rowid DESC LIMIT ?", (run_id, int(limit))
        ).fetchall()
        return [dict(row) for row in rows]

    def trial_status_counts(self, run_id: str, *, stage: str | None = None) -> dict[str, int]:
        if stage:
            rows = self.connection.execute(
                "SELECT status, COUNT(*) AS n FROM trials WHERE run_id=? AND stage=? GROUP BY status",
                (run_id, stage),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT status, COUNT(*) AS n FROM trials WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
        return {str(row["status"]): int(row["n"]) for row in rows}

    def best_dev_candidate(self, run_id: str) -> dict[str, Any] | None:
        # Avoid a hard dependency on SQLite's optional JSON1 extension. The
        # number of confirmed candidates is small by construction.
        rows = self.connection.execute(
            """SELECT * FROM trials WHERE run_id=? AND stage='confirm' AND status='ok'
               ORDER BY created_at DESC LIMIT 500""",
            (run_id,),
        ).fetchall()
        best = None
        best_score = -1.0
        for row in rows:
            item = self._row_with_json(row)
            score = float(item.get("aggregate", {}).get("mean", -1.0))
            if score > best_score:
                best = item
                best_score = score
        return best

    def get_candidate(self, run_id: str, candidate_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM candidates WHERE run_id=? AND candidate_hash=?",
            (run_id, candidate_hash),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["genome"] = json.loads(item.pop("genome_json"))
        return item

    def find_candidates(self, run_id: str, prefix: str, *, limit: int = 20) -> list[dict[str, Any]]:
        if not prefix or len(prefix) < 4:
            raise ValueError("candidate prefix must contain at least four characters")
        rows = self.connection.execute(
            "SELECT * FROM candidates WHERE run_id=? AND candidate_hash LIKE ? ORDER BY created_at DESC LIMIT ?",
            (run_id, f"{prefix}%", int(limit)),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["genome"] = json.loads(item.pop("genome_json"))
            items.append(item)
        return items

    def trials_for_candidate(self, run_id: str, candidate_hash: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM trials WHERE run_id=? AND candidate_hash=? ORDER BY created_at",
            (run_id, candidate_hash),
        ).fetchall()
        return [self._row_with_json(row) for row in rows]

    def generation_summary(self, run_id: str) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT * FROM generations WHERE run_id=? ORDER BY generation", (run_id,)
        ).fetchall()
        return {"generations": [dict(row) for row in rows]}

    def latest_events(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY event_id DESC LIMIT ?", (run_id, int(limit))
        ).fetchall()
        return [dict(row) for row in rows]
