"""Redis-backed StateStore implementation with injectable client support."""

from __future__ import annotations

import json
import time
from typing import Any

from .base import (
    StateStore,
    alert_matches,
    compute_alert_counts,
    job_matches,
    normalize_alert_record,
    parse_status_filter,
)


class RedisStore(StateStore):
    """StateStore backed by Redis (with optional injected client for unit testing)."""

    def __init__(
        self,
        url: str = "",
        client: Any | None = None,
        prefix: str = "ef:",
    ) -> None:
        if client is not None:
            self._client = client
        elif url:
            import redis

            self._client = redis.Redis.from_url(url, decode_responses=True)
        else:
            raise ValueError("RedisStore requires either url or an injected client")
        self.prefix = prefix

    def _k(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    # ---- Settings KV ----
    def get_setting(self, key: str, default: Any = None) -> Any:
        raw = self._client.hget(self._k("settings"), key)
        if raw is None:
            return default
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def set_setting(self, key: str, value: Any) -> None:
        self._client.hset(
            self._k("settings"),
            key,
            json.dumps(value, ensure_ascii=False),
        )

    # ---- Jobs CRUD ----
    def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        record = dict(job)
        record.setdefault("created_at", now)
        record.setdefault("updated_at", record["created_at"])
        record.setdefault("status", "queued")
        record.setdefault("job_type", "single")
        record.setdefault("name", record["job_id"])
        jid = record["job_id"]
        self._client.set(self._k(f"job:{jid}"), json.dumps(record, ensure_ascii=False))
        self._client.zadd(self._k("jobs:index"), {jid: float(record["created_at"])})
        return record

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._k(f"job:{job_id}"))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        record = self.get_job(job_id)
        if record is None:
            return None
        record.update(fields)
        record["updated_at"] = time.time()
        self._client.set(self._k(f"job:{job_id}"), json.dumps(record, ensure_ascii=False))
        return record

    def delete_job(self, job_id: str) -> bool:
        removed = self._client.delete(self._k(f"job:{job_id}"))
        self._client.zrem(self._k("jobs:index"), job_id)
        return bool(removed)

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        ids = self._client.zrevrange(self._k("jobs:index"), 0, -1)
        statuses = parse_status_filter(status)
        matched: list[dict[str, Any]] = []
        for jid in ids:
            if isinstance(jid, bytes):
                jid = jid.decode("utf-8")
            job = self.get_job(jid)
            if job is None:
                continue
            if job_matches(job, statuses=statuses, job_type=job_type, q=q):
                matched.append(job)
        matched.sort(key=lambda j: (-float(j.get("created_at") or 0.0), str(j.get("job_id") or "")))
        total = len(matched)
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        return matched[safe_offset : safe_offset + safe_limit], total

    # ---- FIFO Queue ----
    def enqueue_job(self, job_id: str) -> int:
        pos = self.queue_position(job_id)
        if pos is not None:
            return pos
        self._client.rpush(self._k("queue"), job_id)
        return self.queue_position(job_id) or 1

    def dequeue_job(self) -> str | None:
        val = self._client.lpop(self._k("queue"))
        if val is None:
            return None
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def remove_from_queue(self, job_id: str) -> bool:
        removed = self._client.lrem(self._k("queue"), 0, job_id)
        return int(removed or 0) > 0

    def queue_position(self, job_id: str) -> int | None:
        items = self._client.lrange(self._k("queue"), 0, -1)
        for idx, item in enumerate(items, start=1):
            val = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            if val == job_id:
                return idx
        return None

    def queue_length(self) -> int:
        return int(self._client.llen(self._k("queue")) or 0)

    # ---- Alerts ----
    def save_alerts(self, job_id: str, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.delete_job_alerts(job_id)
        now = time.time()
        normalized = [normalize_alert_record(job_id, a, now) for a in alerts]
        for a in normalized:
            aid = a["alert_id"]
            self._client.set(self._k(f"alert:{aid}"), json.dumps(a, ensure_ascii=False))
            self._client.zadd(self._k("alerts:index"), {aid: float(a["created_at"])})
            self._client.rpush(self._k(f"job:{job_id}:alerts"), aid)
        return normalized

    def _load_alert_ids(self, job_id: str | None) -> list[str]:
        if job_id:
            raw_ids = self._client.lrange(self._k(f"job:{job_id}:alerts"), 0, -1)
        else:
            raw_ids = self._client.zrevrange(self._k("alerts:index"), 0, -1)
        return [i.decode("utf-8") if isinstance(i, bytes) else str(i) for i in raw_ids]

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
        scope_items: list[dict[str, Any]] = []
        for aid in self._load_alert_ids(job_id):
            a = self.get_alert(aid)
            if a is not None:
                scope_items.append(a)
        scope_items.sort(
            key=lambda a: (
                -float(a.get("created_at") or 0.0),
                float(a.get("start_s") or 0.0),
                str(a.get("alert_id") or ""),
            )
        )
        counts = compute_alert_counts(scope_items)

        decisions: set[str] | None = None
        if decision:
            parts = {d.strip().upper() for d in decision.split(",") if d.strip()}
            if parts and "ALL" not in parts:
                decisions = parts

        matched = [
            a
            for a in scope_items
            if alert_matches(
                a,
                job_id=job_id,
                decisions=decisions,
                behavior=behavior,
                review=review,
                min_conf=min_conf,
                q=q,
            )
        ]
        total = len(matched)
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        return matched[safe_offset : safe_offset + safe_limit], total, counts

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._k(f"alert:{alert_id}"))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def update_alert_review(self, alert_id: str, review: str | None) -> dict[str, Any] | None:
        record = self.get_alert(alert_id)
        if record is None:
            return None
        record["review"] = review
        self._client.set(self._k(f"alert:{alert_id}"), json.dumps(record, ensure_ascii=False))
        return record

    def delete_job_alerts(self, job_id: str) -> int:
        list_key = self._k(f"job:{job_id}:alerts")
        raw_ids = self._client.lrange(list_key, 0, -1)
        count = 0
        for raw_id in raw_ids:
            aid = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            if self._client.delete(self._k(f"alert:{aid}")):
                count += 1
            self._client.zrem(self._k("alerts:index"), aid)
        self._client.delete(list_key)
        return count

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()
