"""Parametrized StateStore tests across SQLiteStore and RedisStore (FakeRedis)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from backend.app.config import Settings
from backend.app.store import RedisStore, SQLiteStore, StateStore, create_state_store


class FakeRedis:
    """Minimal in-memory Redis mock supporting commands used by RedisStore."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    def hget(self, name: str, key: str) -> str | None:
        return self._hashes.get(name, {}).get(key)

    def hset(self, name: str, key: str, value: str) -> int:
        bucket = self._hashes.setdefault(name, {})
        is_new = key not in bucket
        bucket[key] = value
        return 1 if is_new else 0

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> bool:
        self._kv[name] = value
        return True

    def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            if name in self._kv:
                del self._kv[name]
                removed += 1
            if name in self._lists:
                del self._lists[name]
                removed += 1
            if name in self._zsets:
                del self._zsets[name]
                removed += 1
            if name in self._hashes:
                del self._hashes[name]
                removed += 1
        return removed

    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        zs = self._zsets.setdefault(name, {})
        added = 0
        for member, score in mapping.items():
            if member not in zs:
                added += 1
            zs[member] = float(score)
        return added

    def zrem(self, name: str, *members: str) -> int:
        zs = self._zsets.get(name, {})
        removed = 0
        for m in members:
            if m in zs:
                del zs[m]
                removed += 1
        return removed

    def zrevrange(self, name: str, start: int, end: int) -> list[str]:
        zs = self._zsets.get(name, {})
        ordered = [
            m for m, _ in sorted(zs.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        stop = len(ordered) if end == -1 else end + 1
        return ordered[start:stop]

    def rpush(self, name: str, *values: str) -> int:
        lst = self._lists.setdefault(name, [])
        lst.extend(values)
        return len(lst)

    def lpop(self, name: str) -> str | None:
        lst = self._lists.get(name)
        if not lst:
            return None
        return lst.pop(0)

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        lst = self._lists.get(name, [])
        stop = len(lst) if end == -1 else end + 1
        return list(lst[start:stop])

    def lrem(self, name: str, count: int, value: str) -> int:
        lst = self._lists.get(name, [])
        if count == 0:
            before = len(lst)
            self._lists[name] = [x for x in lst if x != value]
            return before - len(self._lists[name])
        removed = 0
        out: list[str] = []
        for item in lst:
            if item == value and removed < abs(count):
                removed += 1
                continue
            out.append(item)
        self._lists[name] = out
        return removed

    def llen(self, name: str) -> int:
        return len(self._lists.get(name, []))

    def close(self) -> None:
        return None


@pytest.fixture(params=["sqlite", "redis"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> StateStore:
    if request.param == "sqlite":
        st = SQLiteStore(tmp_path / "state.db")
    else:
        st = RedisStore(client=FakeRedis(), prefix="test:")
    yield st
    st.close()


def test_settings_kv(store: StateStore) -> None:
    assert store.get_setting("missing") is None
    assert store.get_setting("missing", "fallback") == "fallback"

    store.set_setting("gpu_url", "https://gpu.example.com")
    store.set_setting("sites", [{"site_id": "s1", "cameras": ["C1", "C2"]}])
    assert store.get_setting("gpu_url") == "https://gpu.example.com"
    assert store.get_setting("sites") == [{"site_id": "s1", "cameras": ["C1", "C2"]}]

    store.set_setting("gpu_url", "http://localhost:8001")
    assert store.get_setting("gpu_url") == "http://localhost:8001"


def test_jobs_crud_and_filtering(store: StateStore) -> None:
    t0 = time.time()
    store.create_job(
        {
            "job_id": "j1",
            "job_type": "single",
            "status": "queued",
            "name": "Hallway Cam 1",
            "created_at": t0,
        }
    )
    store.create_job(
        {
            "job_id": "j2",
            "job_type": "single",
            "status": "done",
            "name": "Gallery Porcelain",
            "created_at": t0 + 1,
        }
    )
    store.create_job(
        {
            "job_id": "j3",
            "job_type": "site",
            "status": "error",
            "name": "Site Main Room",
            "site_id": "demo_gallery",
            "created_at": t0 + 2,
        }
    )

    got = store.get_job("j2")
    assert got is not None and got["name"] == "Gallery Porcelain"

    updated = store.update_job("j1", status="running", progress=0.5)
    assert updated is not None and updated["status"] == "running" and updated["progress"] == 0.5

    all_jobs, total = store.list_jobs()
    assert total == 3
    assert [j["job_id"] for j in all_jobs] == ["j3", "j2", "j1"]

    active, active_total = store.list_jobs(status="active")
    assert active_total == 1 and active[0]["job_id"] == "j1"

    site_jobs, site_total = store.list_jobs(job_type="site")
    assert site_total == 1 and site_jobs[0]["job_id"] == "j3"

    searched, s_total = store.list_jobs(q="porcelain")
    assert s_total == 1 and searched[0]["job_id"] == "j2"

    paged, p_total = store.list_jobs(limit=1, offset=1)
    assert p_total == 3 and len(paged) == 1 and paged[0]["job_id"] == "j2"

    assert store.delete_job("j2") is True
    assert store.get_job("j2") is None
    assert store.delete_job("j2") is False


def test_fifo_queue_operations(store: StateStore) -> None:
    assert store.queue_length() == 0
    assert store.dequeue_job() is None

    assert store.enqueue_job("job_a") == 1
    assert store.enqueue_job("job_b") == 2
    assert store.enqueue_job("job_c") == 3
    # Idempotent re-enqueue keeps original position
    assert store.enqueue_job("job_b") == 2
    assert store.queue_length() == 3
    assert store.queue_position("job_a") == 1
    assert store.queue_position("job_c") == 3
    assert store.queue_position("nonexistent") is None

    assert store.remove_from_queue("job_b") is True
    assert store.remove_from_queue("job_b") is False
    assert store.queue_position("job_c") == 2
    assert store.queue_length() == 2

    assert store.dequeue_job() == "job_a"
    assert store.dequeue_job() == "job_c"
    assert store.dequeue_job() is None
    assert store.queue_length() == 0


def test_alerts_filtering_review_and_deletion(store: StateStore) -> None:
    t0 = time.time()
    store.save_alerts(
        "job1",
        [
            {
                "event_id": "e1",
                "job_name": "Cam 1",
                "track_id": 2,
                "artifact_id": 1,
                "decision": "ANOMALY",
                "behavior": "stealing",
                "confidence": 0.94,
                "evidence": "Put object in bag",
                "start_s": 10.0,
                "end_s": 14.0,
                "img_url": "/api/jobs/job1/files/k/vlm_inputs/t1/sheet.jpg",
                "created_at": t0,
            },
            {
                "event_id": "e2",
                "job_name": "Cam 1",
                "track_id": 3,
                "artifact_id": 2,
                "decision": "CANDIDATE",
                "behavior": "uncertain",
                "confidence": 0.52,
                "evidence": "Hand occluded",
                "start_s": 20.0,
                "end_s": 23.0,
                "created_at": t0,
            },
            {
                "event_id": "e3",
                "job_name": "Cam 1",
                "track_id": 4,
                "artifact_id": 2,
                "decision": "NORMAL",
                "behavior": "normal",
                "confidence": 0.88,
                "evidence": "Just looking",
                "start_s": 30.0,
                "end_s": 32.0,
                "created_at": t0,
            },
        ],
    )
    store.save_alerts(
        "job2",
        [
            {
                "event_id": "e1",
                "job_name": "Cam 2",
                "track_id": 1,
                "artifact_id": 1,
                "decision": "ANOMALY",
                "behavior": "touching",
                "confidence": 0.81,
                "evidence": "Touched ceramic vase",
                "start_s": 5.0,
                "end_s": 8.0,
                "created_at": t0 + 5,
            }
        ],
    )

    items, total, counts = store.list_alerts()
    assert total == 4
    assert len(items) == 4
    assert counts["total"] == 4
    assert counts["ANOMALY"] == 2
    assert counts["CANDIDATE"] == 1
    assert counts["NORMAL"] == 1
    assert counts["unreviewed"] == 4

    # Filter by job_id and decision
    j1_anom, j1_total, j1_counts = store.list_alerts(job_id="job1", decision="ANOMALY")
    assert j1_total == 1
    assert j1_anom[0]["alert_id"] == "job1:e1"
    assert j1_counts["total"] == 3

    # Filter by behavior, min_conf, q
    hi, hi_total, _ = store.list_alerts(min_conf=0.85, q="bag")
    assert hi_total == 1 and hi[0]["alert_id"] == "job1:e1"

    beh, beh_total, _ = store.list_alerts(behavior="touching")
    assert beh_total == 1 and beh[0]["alert_id"] == "job2:e1"

    # Review update
    reviewed = store.update_alert_review("job1:e1", "ok")
    assert reviewed is not None and reviewed["review"] == "ok"
    assert store.get_alert("job1:e1")["review"] == "ok"

    ok_items, ok_total, _ = store.list_alerts(review="ok")
    assert ok_total == 1 and ok_items[0]["alert_id"] == "job1:e1"

    unrev_items, unrev_total, _ = store.list_alerts(review="unreviewed")
    assert unrev_total == 3

    # Reset review back to None
    reset_rv = store.update_alert_review("job1:e1", None)
    assert reset_rv is not None and reset_rv["review"] is None

    # Delete job alerts
    deleted = store.delete_job_alerts("job1")
    assert deleted == 3
    remaining, rem_total, _ = store.list_alerts()
    assert rem_total == 1 and remaining[0]["job_id"] == "job2"


def test_create_state_store_factory(tmp_path: Path) -> None:
    s_sqlite = Settings(data_dir=tmp_path / "d1", redis_url="")
    st1 = create_state_store(s_sqlite)
    try:
        assert isinstance(st1, SQLiteStore)
    finally:
        st1.close()

    # Check RedisStore branch without requiring a live Redis server
    pytest.importorskip("redis")
    s_redis = Settings(data_dir=tmp_path / "d2", redis_url="redis://127.0.0.1:6379/0")
    st2 = create_state_store(s_redis)
    try:
        assert isinstance(st2, RedisStore)
    finally:
        st2.close()
