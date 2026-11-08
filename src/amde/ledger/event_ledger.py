"""AMDE 4.0 Append-Only Immutable Evidence Ledger.

Guarantees:
1. SQLite WAL mode with atomic transactional integrity.
2. Idempotent event writes preventing duplicate commits across worker retries.
3. Cryptographic SHA-256 state chain linking all experiment events.
4. Export to canonical JSONL for transparency and external audits.
5. High-throughput asynchronous background batching eliminating main-thread disk I/O bottlenecks.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import queue
import sqlite3
import threading
from typing import Any, Iterator, List, Mapping, Optional, Tuple

from amde.core.contracts import ExperimentEvent, utc_now_iso
from amde.core.identity import canonical_json


class EventLedger:
    """Append-only, verifiable event ledger managing experiment provenance with async batching."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen_event_ids: set[str] = set()
        self._seen_event_map: dict[str, ExperimentEvent] = {}
        self._current_seq = 0
        self._last_hash = "GENESIS"
        self._closed = False

        self._init_db()

        self._write_queue: queue.Queue[Optional[tuple]] = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._writer_loop, name=f"amde-ledger-writer-{self.db_path.name}", daemon=True
        )
        self._worker_thread.start()
        atexit.register(self.close)

    def __enter__(self) -> EventLedger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Managed SQLite connection ensuring PRAGMA busy_timeout and deterministic closure."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes SQLite database with WAL mode and schema."""
        self._writer_exception: Optional[Exception] = None
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

            cursor = conn.cursor()
            cursor.execute("SELECT seq, event_hash, event_id FROM experiment_events ORDER BY seq ASC;")
            rows = cursor.fetchall()
            if rows:
                self._current_seq = rows[-1][0]
                self._last_hash = rows[-1][1]
                for r in rows:
                    self._seen_event_ids.add(r[2])
            else:
                self._current_seq = 0
                self._last_hash = "GENESIS"

    def _writer_loop(self) -> None:
        """Dedicated background thread worker executing batched commits to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
        except Exception:
            return

        try:
            while True:
                item = self._write_queue.get()
                if item is None:
                    self._write_queue.task_done()
                    break

                batch = [item]
                # Drain any additional pending writes into a single high-efficiency batch
                while not self._write_queue.empty() and len(batch) < 100:
                    try:
                        next_item = self._write_queue.get_nowait()
                        if next_item is None:
                            self._write_queue.put(None)
                            self._write_queue.task_done()
                            break
                        batch.append(next_item)
                    except queue.Empty:
                        break

                try:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO experiment_events
                        (event_id, experiment_id, event_type, payload_json, prev_hash, event_hash, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        batch,
                    )
                    conn.commit()
                except Exception as exc:
                    self._writer_exception = exc
                finally:
                    for _ in batch:
                        self._write_queue.task_done()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def flush(self) -> None:
        """Flushes all queued events to persistent SQLite storage."""
        if hasattr(self, "_write_queue"):
            self._write_queue.join()
        if getattr(self, "_writer_exception", None) is not None:
            exc = self._writer_exception
            self._writer_exception = None
            raise RuntimeError(f"EventLedger writer worker failed: {exc}") from exc

    def close(self) -> None:
        """Flushes pending events and terminates the background writer worker cleanly."""
        with self._lock:
            if getattr(self, "_closed", False):
                return
            self._closed = True

        if hasattr(self, "_write_queue") and hasattr(self, "_worker_thread"):
            self.flush()
            if self._worker_thread.is_alive():
                self._write_queue.put(None)
                self._worker_thread.join(timeout=2.0)

    def append_event(
        self,
        event_id: str,
        experiment_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> ExperimentEvent:
        """Appends an idempotent event with in-memory SHA-256 state chaining and non-blocking I/O."""
        with self._lock:
            # Idempotency check
            if event_id in self._seen_event_ids:
                if event_id in self._seen_event_map:
                    return self._seen_event_map[event_id]
                self.flush()
                with self._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT event_id, experiment_id, event_type, payload_json, seq, timestamp FROM experiment_events WHERE event_id = ?;",
                        (event_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        ev = ExperimentEvent(
                            event_id=row[0],
                            experiment_id=row[1],
                            event_type=row[2],
                            payload=json.loads(row[3]),
                            sequence_number=row[4],
                            timestamp=row[5],
                        )
                        self._seen_event_map[event_id] = ev
                        return ev

            payload_str = canonical_json(dict(payload))
            ts = utc_now_iso()
            self._current_seq += 1
            current_seq = self._current_seq
            prev_hash = self._last_hash

            raw_hash_data = f"{current_seq}|{prev_hash}|{event_id}|{experiment_id}|{event_type}|{payload_str}|{ts}"
            event_hash = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()
            self._last_hash = event_hash

            record = (event_id, experiment_id, event_type, payload_str, prev_hash, event_hash, ts)
            self._seen_event_ids.add(event_id)

            event = ExperimentEvent(
                event_id=event_id,
                experiment_id=experiment_id,
                event_type=event_type,
                payload=dict(payload),
                sequence_number=current_seq,
                timestamp=ts,
            )
            self._seen_event_map[event_id] = event
            self._write_queue.put(record)

        return event

    def get_events(self, experiment_id: str) -> List[ExperimentEvent]:
        """Retrieves ordered event stream for an experiment."""
        self.flush()
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
        self.flush()
        hasher = hashlib.sha256()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT event_hash FROM experiment_events ORDER BY seq ASC;")
            for row in cursor.fetchall():
                hasher.update(row[0].encode("utf-8"))
        return hasher.hexdigest()

    def export_jsonl(self, jsonl_path: Path) -> None:
        """Exports the event stream to standard JSON Lines."""
        self.flush()
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

    def verify_chain(self) -> Tuple[bool, str]:
        """Performs full cryptographic audit over the entire ledger.

        Validates:
        1. Monotonic sequence continuity with no gaps or reordering.
        2. Cryptographic hash chaining: genesis prev_hash is GENESIS, each event's
           prev_hash matches the preceding event's event_hash.
        3. Recalculation and validation of every event hash over:
           (seq, prev_hash, event_id, experiment_id, event_type, payload_json, timestamp).
        """
        self.flush()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT seq, event_id, experiment_id, event_type, payload_json, prev_hash, event_hash, timestamp
                FROM experiment_events ORDER BY seq ASC;
                """
            )
            rows = cursor.fetchall()

        if not rows:
            return True, "VALID_EMPTY"

        expected_seq = 1
        last_hash = "GENESIS"

        for row in rows:
            seq, ev_id, exp_id, ev_type, payload_str, p_hash, stored_hash, ts = row

            # 1. Sequence check
            if seq != expected_seq:
                return False, f"Sequence discontinuity: expected seq {expected_seq}, got {seq}"

            # 2. Hash chaining check
            if p_hash != last_hash:
                return False, f"Broken chain at seq {seq}: prev_hash '{p_hash}' != expected '{last_hash}'"

            # 3. Hash integrity check
            raw_hash_data = f"{seq}|{p_hash}|{ev_id}|{exp_id}|{ev_type}|{payload_str}|{ts}"
            recomputed = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()
            if recomputed != stored_hash:
                return False, f"Tamper detected at seq {seq} (event_id '{ev_id}'): recomputed hash '{recomputed}' != stored '{stored_hash}'"

            last_hash = stored_hash
            expected_seq += 1

        return True, f"VALID_{len(rows)}_EVENTS"
