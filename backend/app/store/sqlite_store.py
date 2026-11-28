"""SQLite-backed StateStore implementation with WAL mode and RLock safety."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .base import (
    StateStore,
    compute_alert_counts,
    job_matches,
    normalize_alert_record,
    parse_status_filter,
)


class SQLiteStore(StateStore):
    """Thread-safe SQLite StateStore using WAL mode and indexed tables."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                job_type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                finished_at REAL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

            CREATE TABLE IF NOT EXISTS job_queue (
                pos INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                enqueued_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                job_name TEXT,
                cam TEXT,
                track_id TEXT,
                artifact_id TEXT,
                decision TEXT,
                behavior TEXT,
                confidence REAL,
                evidence TEXT,
                start_s REAL,
                end_s REAL,
                review TEXT,
                img_url TEXT,
                created_at REAL NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_job ON alerts(job_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_decision ON alerts(decision);
            CREATE INDEX IF NOT EXISTS idx_alerts_behavior ON alerts(behavior);
            CREATE INDEX IF NOT EXISTS idx_alerts_review ON alerts(review);
            CREATE INDEX IF NOT EXISTS idx_alerts_conf ON alerts(confidence);
            """
        )
        self._conn.commit()

    # ---- Settings KV ----
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return default
            return json.loads(row["value_json"])

    def set_setting(self, key: str, value: Any) -> None:
        now = time.time()
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, payload, now),
            )
            self._conn.commit()

    # ---- Jobs CRUD ----
    def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        record = dict(job)
        record.setdefault("created_at", now)
        record.setdefault("updated_at", record["created_at"])
        record.setdefault("status", "queued")
        record.setdefault("job_type", "single")
        record.setdefault("name", record["job_id"])
        payload = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (job_id, status, job_type, name, created_at, updated_at, finished_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["job_id"],
                    record["status"],
                    record["job_type"],
                    record["name"],
                    float(record["created_at"]),
                    float(record["updated_at"]),
                    record.get("finished_at"),
                    payload,
                ),
            )
            self._conn.commit()
        return record

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            record = json.loads(row["payload_json"])
            record.update(fields)
            record["updated_at"] = time.time()
            payload = json.dumps(record, ensure_ascii=False)
            self._conn.execute(
                """
                UPDATE jobs
                SET status = ?, job_type = ?, name = ?, updated_at = ?, finished_at = ?, payload_json = ?
                WHERE job_id = ?
                """,
                (
                    record.get("status", "queued"),
                    record.get("job_type", "single"),
                    record.get("name", job_id),
                    float(record["updated_at"]),
                    record.get("finished_at"),
                    payload,
                    job_id,
                ),
            )
            self._conn.commit()
            return record

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        statuses = parse_status_filter(status)
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"LOWER(status) IN ({placeholders})")
            params.extend(sorted(statuses))
        if job_type:
            clauses.append("LOWER(job_type) = ?")
            params.append(job_type.strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload_json FROM jobs {where} ORDER BY created_at DESC, job_id ASC",
                params,
            ).fetchall()
        items = [json.loads(r["payload_json"]) for r in rows]
        if q:
            items = [j for j in items if job_matches(j, q=q)]
        total = len(items)
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        return items[safe_offset : safe_offset + safe_limit], total

    # ---- FIFO Queue ----
    def enqueue_job(self, job_id: str) -> int:
        now = time.time()
        with self._lock:
            existing = self.queue_position(job_id)
            if existing is not None:
                return existing
            self._conn.execute(
                "INSERT INTO job_queue (job_id, enqueued_at) VALUES (?, ?)",
                (job_id, now),
            )
            self._conn.commit()
            return self.queue_position(job_id) or 1

    def dequeue_job(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT pos, job_id FROM job_queue ORDER BY pos ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self._conn.execute("DELETE FROM job_queue WHERE pos = ?", (row["pos"],))
            self._conn.commit()
            return str(row["job_id"])

    def remove_from_queue(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM job_queue WHERE job_id = ?", (job_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id FROM job_queue ORDER BY pos ASC"
            ).fetchall()
            for idx, r in enumerate(rows, start=1):
                if r["job_id"] == job_id:
                    return idx
            return None

    def queue_length(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM job_queue").fetchone()
            return int(row["cnt"]) if row else 0

    # ---- Alerts ----
    def save_alerts(self, job_id: str, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.time()
        normalized = [normalize_alert_record(job_id, a, now) for a in alerts]
        with self._lock:
            self._conn.execute("DELETE FROM alerts WHERE job_id = ?", (job_id,))
            for a in normalized:
                self._conn.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, job_id, event_id, job_name, cam, track_id, artifact_id,
                        decision, behavior, confidence, evidence, start_s, end_s,
                        review, img_url, created_at, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        a["alert_id"],
                        a["job_id"],
                        a["event_id"],
                        a["job_name"],
                        a["cam"],
                        None if a["track_id"] is None else str(a["track_id"]),
                        None if a["artifact_id"] is None else str(a["artifact_id"]),
                        a["decision"],
                        a["behavior"],
                        a["confidence"],
                        a["evidence"],
                        a["start_s"],
                        a["end_s"],
                        a["review"],
                        a["img_url"],
                        a["created_at"],
                        json.dumps(a, ensure_ascii=False),
                    ),
                )
            self._conn.commit()
        return normalized

    def list_alerts(
        self,
        job_id: str | None = None,
        decision: str | None = None,
        behavior: str | None = None,
        review: str | None = None,
        min_conf: float | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        with self._lock:
            if job_id:
                scope_rows = self._conn.execute(
                    """
                    SELECT payload_json FROM alerts
                    WHERE job_id = ?
                    ORDER BY created_at DESC, COALESCE(start_s, 0.0) ASC, alert_id ASC
                    """,
                    (job_id,),
                ).fetchall()
            else:
                scope_rows = self._conn.execute(
                    """
                    SELECT payload_json FROM alerts
                    ORDER BY created_at DESC, COALESCE(start_s, 0.0) ASC, alert_id ASC
                    """
                ).fetchall()

        scope_items = [json.loads(r["payload_json"]) for r in scope_rows]
        counts = compute_alert_counts(scope_items)

        decisions: set[str] | None = None
        if decision:
            parts = {d.strip().upper() for d in decision.split(",") if d.strip()}
            if parts and "ALL" not in parts:
                decisions = parts

        clauses: list[str] = []
        params: list[Any] = []
        if job_id:
            clauses.append("job_id = ?")
            params.append(job_id)
        if decisions:
            placeholders = ",".join("?" for _ in decisions)
            clauses.append(f"UPPER(decision) IN ({placeholders})")
            params.extend(sorted(decisions))
        if behavior:
            clauses.append("LOWER(behavior) = ?")
            params.append(behavior.strip().lower())
        if review:
            rv = review.strip().lower()
            if rv in ("unreviewed", "none", "pending", "null"):
                clauses.append("review IS NULL")
            else:
                clauses.append("review = ?")
                params.append(rv)
        if min_conf is not None:
            clauses.append("confidence IS NOT NULL AND confidence >= ?")
            params.append(float(min_conf))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT payload_json FROM alerts
                {where}
                ORDER BY created_at DESC, COALESCE(start_s, 0.0) ASC, alert_id ASC
                """,
                params,
            ).fetchall()

        matched = [json.loads(r["payload_json"]) for r in rows]
        if q:
            needle = q.strip().lower()
            matched = [
                a
                for a in matched
                if needle
                in " ".join(
                    str(a.get(k) or "")
                    for k in (
                        "alert_id",
                        "job_id",
                        "job_name",
                        "event_id",
                        "cam",
                        "track_id",
                        "artifact_id",
                        "decision",
                        "behavior",
                        "evidence",
                    )
                ).lower()
            ]
        total = len(matched)
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        return matched[safe_offset : safe_offset + safe_limit], total, counts

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def update_alert_review(self, alert_id: str, review: str | None) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if row is None:
                return None
            record = json.loads(row["payload_json"])
            record["review"] = review
            payload = json.dumps(record, ensure_ascii=False)
            self._conn.execute(
                "UPDATE alerts SET review = ?, payload_json = ? WHERE alert_id = ?",
                (review, payload, alert_id),
            )
            self._conn.commit()
            return record

    def delete_job_alerts(self, job_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM alerts WHERE job_id = ?", (job_id,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
