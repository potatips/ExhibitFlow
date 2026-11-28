"""Abstract StateStore interface and shared filtering helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def parse_status_filter(status: str | None) -> set[str] | None:
    """Parse status query filter (supports comma-separated values and 'active' alias)."""
    if not status:
        return None
    raw = [s.strip().lower() for s in status.split(",") if s.strip()]
    if not raw or "all" in raw:
        return None
    out: set[str] = set()
    for s in raw:
        if s == "active":
            out.update({"queued", "running", "syncing"})
        else:
            out.add(s)
    return out or None


def job_matches(
    job: dict[str, Any],
    statuses: set[str] | None = None,
    job_type: str | None = None,
    q: str | None = None,
) -> bool:
    """Return True when a job record satisfies the given filters."""
    if statuses is not None and str(job.get("status", "")).lower() not in statuses:
        return False
    if job_type and str(job.get("job_type", "")).lower() != job_type.lower():
        return False
    if q:
        needle = q.strip().lower()
        haystack = " ".join(
            str(job.get(k) or "")
            for k in ("job_id", "name", "site_id", "source", "status")
        ).lower()
        if needle not in haystack:
            return False
    return True


def normalize_alert_record(job_id: str, raw: dict[str, Any], now: float) -> dict[str, Any]:
    """Normalize an alert dict before persisting to StateStore."""
    event_id = str(raw.get("event_id") or raw.get("id") or "")
    alert_id = str(raw.get("alert_id") or (f"{job_id}:{event_id}" if event_id else f"{job_id}:{now}"))
    conf = raw.get("confidence")
    if conf is not None:
        conf = float(conf)
    start_s = raw.get("start_s")
    if start_s is not None:
        start_s = float(start_s)
    end_s = raw.get("end_s")
    if end_s is not None:
        end_s = float(end_s)
    return {
        "alert_id": alert_id,
        "job_id": job_id,
        "event_id": event_id or alert_id,
        "job_name": raw.get("job_name") or "",
        "job_type": raw.get("job_type") or "single",
        "cam": raw.get("cam") or "",
        "track_id": raw.get("track_id"),
        "artifact_id": raw.get("artifact_id"),
        "decision": str(raw.get("decision") or "ANOMALY").upper(),
        "behavior": str(raw.get("behavior") or "unknown").lower(),
        "confidence": conf,
        "evidence": raw.get("evidence") or "",
        "start_s": start_s,
        "end_s": end_s,
        "task": raw.get("task"),
        "review": raw.get("review"),
        "img_url": raw.get("img_url"),
        "created_at": float(raw.get("created_at") or now),
    }


def alert_matches(
    alert: dict[str, Any],
    job_id: str | None = None,
    decisions: set[str] | None = None,
    behavior: str | None = None,
    review: str | None = None,
    min_conf: float | None = None,
    q: str | None = None,
) -> bool:
    """Return True when an alert record satisfies the given filters."""
    if job_id and alert.get("job_id") != job_id:
        return False
    if decisions is not None and str(alert.get("decision") or "").upper() not in decisions:
        return False
    if behavior and str(alert.get("behavior") or "").lower() != behavior.strip().lower():
        return False
    if review:
        rv = review.strip().lower()
        cur = alert.get("review")
        if rv in ("unreviewed", "none", "pending", "null"):
            if cur is not None:
                return False
        elif cur != rv:
            return False
    if min_conf is not None:
        conf = alert.get("confidence")
        if conf is None or float(conf) < float(min_conf):
            return False
    if q:
        needle = q.strip().lower()
        haystack = " ".join(
            str(alert.get(k) or "")
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
        if needle not in haystack:
            return False
    return True


def compute_alert_counts(alerts: list[dict[str, Any]]) -> dict[str, int]:
    """Compute decision and review breakdown counts over a list of alerts."""
    counts = {
        "total": len(alerts),
        "ANOMALY": 0,
        "CANDIDATE": 0,
        "NORMAL": 0,
        "ok": 0,
        "no": 0,
        "unreviewed": 0,
    }
    for a in alerts:
        dec = str(a.get("decision") or "").upper()
        if dec in counts:
            counts[dec] += 1
        rv = a.get("review")
        if rv == "ok":
            counts["ok"] += 1
        elif rv == "no":
            counts["no"] += 1
        else:
            counts["unreviewed"] += 1
    return counts


class StateStore(ABC):
    """Persistence interface for settings, jobs, FIFO queue, and alerts."""

    # ---- Settings KV ----
    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieve a JSON-deserialized runtime setting."""

    @abstractmethod
    def set_setting(self, key: str, value: Any) -> None:
        """Persist a JSON-serializable runtime setting."""

    # ---- Jobs CRUD ----
    @abstractmethod
    def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Insert a new job record and return the stored dict."""

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch a single job by ID, or None if not found."""

    @abstractmethod
    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        """Update fields on an existing job, returning the updated dict or None."""

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        """Delete a job record by ID. Returns True if a record was removed."""

    @abstractmethod
    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs ordered newest first, returning (items, total_matching)."""

    # ---- FIFO Queue ----
    @abstractmethod
    def enqueue_job(self, job_id: str) -> int:
        """Add job_id to the tail of the FIFO queue (idempotent) and return 1-based position."""

    @abstractmethod
    def dequeue_job(self) -> str | None:
        """Pop and return the oldest queued job_id, or None if empty."""

    @abstractmethod
    def remove_from_queue(self, job_id: str) -> bool:
        """Remove job_id from the queue if present."""

    @abstractmethod
    def queue_position(self, job_id: str) -> int | None:
        """Return 1-based position in the FIFO queue, or None if not queued."""

    @abstractmethod
    def queue_length(self) -> int:
        """Return current number of queued jobs."""

    # ---- Alerts ----
    @abstractmethod
    def save_alerts(self, job_id: str, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace alerts for job_id and return normalized stored records."""

    @abstractmethod
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
        """Filter alerts and return (items, total_matching, scope_counts)."""

    @abstractmethod
    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        """Fetch a single alert by alert_id."""

    @abstractmethod
    def update_alert_review(self, alert_id: str, review: str | None) -> dict[str, Any] | None:
        """Update the review field ('ok', 'no', or None) of an alert."""

    @abstractmethod
    def delete_job_alerts(self, job_id: str) -> int:
        """Delete all alerts belonging to job_id, returning count removed."""

    def close(self) -> None:
        """Release underlying connections or resources."""
        return None
