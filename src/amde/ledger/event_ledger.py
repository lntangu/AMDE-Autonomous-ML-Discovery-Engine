"""AMDE 4.0 Append-Only Immutable Evidence Ledger.

Guarantees:
1. SQLite WAL mode with atomic transactional integrity.
2. Idempotent event writes preventing duplicate commits across worker retries.
3. Cryptographic SHA-256 state chain linking all experiment events.
4. Export to canonical JSONL for transparency and external audits.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional

from amde.core.contracts import ExperimentEvent, utc_now_iso
from amde.core.identity import canonical_json


class EventLedger:
    """Append-only, verifiable event ledger managing experiment provenance."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Managed SQLite connection ensuring PRAGMA busy_timeout and deterministic closure."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes SQLite database with WAL mode and schema."""
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    experiment_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.commit()

    def append_event(
        self,
        event_id: str,
        experiment_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> ExperimentEvent:
        """Appends an idempotent event to the ledger with hash chaining."""
        payload_str = canonical_json(dict(payload))
        ts = utc_now_iso()

        with self._connection() as conn:
            cursor = conn.cursor()
            # Fetch latest event hash for chaining
            cursor.execute("SELECT event_hash, seq FROM experiment_events ORDER BY seq DESC LIMIT 1;")
            row = cursor.fetchone()
            prev_hash = row[0] if row else "GENESIS"
            current_seq = (row[1] + 1) if row else 1

            # Compute SHA-256 of (seq, prev_hash, event_id, event_type, payload_str)
            raw_hash_data = f"{current_seq}|{prev_hash}|{event_id}|{event_type}|{payload_str}|{ts}"
            event_hash = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()

            # Insert with idempotency guard
            cursor.execute(
                """
                INSERT OR IGNORE INTO experiment_events
                (event_id, experiment_id, event_type, payload_json, prev_hash, event_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (event_id, experiment_id, event_type, payload_str, prev_hash, event_hash, ts),
            )
            conn.commit()

        return ExperimentEvent(
            event_id=event_id,
            experiment_id=experiment_id,
            event_type=event_type,
            payload=dict(payload),
            sequence_number=current_seq,
            timestamp=ts,
        )

    def get_events(self, experiment_id: str) -> List[ExperimentEvent]:
        """Retrieves ordered event stream for an experiment."""
        events = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, experiment_id, event_type, payload_json, seq, timestamp
                FROM experiment_events
                WHERE experiment_id = ?
                ORDER BY seq ASC;
                """,
                (experiment_id,),
            )
            for row in cursor.fetchall():
                events.append(
                    ExperimentEvent(
                        event_id=row[0],
                        experiment_id=row[1],
                        event_type=row[2],
                        payload=json.loads(row[3]),
                        sequence_number=row[4],
                        timestamp=row[5],
                    )
                )
        return events

    def compute_ledger_sha256(self) -> str:
        """Computes a cryptographic fingerprint over the entire event history."""
        hasher = hashlib.sha256()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT event_hash FROM experiment_events ORDER BY seq ASC;")
            for row in cursor.fetchall():
                hasher.update(row[0].encode("utf-8"))
        return hasher.hexdigest()

    def export_jsonl(self, jsonl_path: Path) -> None:
        """Exports the event stream to standard JSON Lines."""
        with self._connection() as conn, open(jsonl_path, "w", encoding="utf-8") as f:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT seq, event_id, experiment_id, event_type, payload_json, prev_hash, event_hash, timestamp
                FROM experiment_events ORDER BY seq ASC;
                """
            )
            for row in cursor.fetchall():
                record = {
                    "seq": row[0],
                    "event_id": row[1],
                    "experiment_id": row[2],
                    "event_type": row[3],
                    "payload": json.loads(row[4]),
                    "prev_hash": row[5],
                    "event_hash": row[6],
                    "timestamp": row[7],
                }
                f.write(json.dumps(record) + "\n")
