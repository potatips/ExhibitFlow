"""Background job queue dispatcher, GPU health monitor, and artifact syncer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts.base import ArtifactStore
from .config import Settings
from .gpu_client import GPUClient, GPUClientError, compute_gpu_ready
from .sse import EventBroker
from .store.base import StateStore


class JobDispatcher:
    """Orchestrates FIFO job dispatch to GPU_server and artifact synchronization."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        artifacts: ArtifactStore,
        gpu: GPUClient,
        broker: EventBroker,
    ) -> None:
        self.settings = settings
        self.store = store
        self.artifacts = artifacts
        self.gpu = gpu
        self.broker = broker
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_health_sig: tuple[Any, ...] | None = None
        self._last_meta_refresh: float = 0.0

    def notify(self) -> None:
        """Wake the dispatch loop immediately (e.g. after job enqueue or GPU config update)."""
        self._wake.set()

    async def start(self) -> None:
        """Recover stale jobs from prior crashes and launch background worker tasks."""
        self.recover_stale_jobs()
        self._stop.clear()
        # Probe health once synchronously on startup so initial /api/health is accurate
        await self.probe_health_once()
        self._tasks = [
            asyncio.create_task(self._health_loop(), name="ef-health-loop"),
            asyncio.create_task(self._dispatch_loop(), name="ef-dispatch-loop"),
        ]

    async def stop(self) -> None:
        """Stop background tasks cleanly."""
        self._stop.set()
        self._wake.set()
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def recover_stale_jobs(self) -> None:
        """Mark stale running/syncing jobs from prior server runs as error and preserve queued jobs."""
        stale, _ = self.store.list_jobs(status="running,syncing", limit=1000, offset=0)
        now = time.time()
        for job in stale:
            jid = job["job_id"]
            self.store.remove_from_queue(jid)
            self.store.update_job(
                jid,
                status="error",
                error="Backend đã khởi động lại khi job đang chạy hoặc đồng bộ.",
                finished_at=now,
            )
        queued, _ = self.store.list_jobs(status="queued", limit=1000, offset=0)
        # Ensure FIFO order from oldest created_at to newest
        for job in sorted(queued, key=lambda j: float(j.get("created_at") or 0.0)):
            if self.store.queue_position(job["job_id"]) is None:
                self.store.enqueue_job(job["job_id"])

    async def probe_health_once(self) -> dict[str, Any]:
        """Probe GPU health, update cached state, refresh metadata if online, and emit SSE on change."""
        health = await self.gpu.health()
        health["ready"] = compute_gpu_ready(health)
        self.store.set_setting("gpu_health", health)

        sig = (
            bool(health.get("online")),
            bool(health.get("ready")),
            bool(health.get("busy")),
            bool(health.get("demo")),
            (health.get("models") or {}).get("state"),
            (health.get("vlm") or {}).get("state"),
            health.get("url"),
            health.get("error"),
        )
        changed = sig != self._last_health_sig
        was_online = bool(self._last_health_sig[0]) if self._last_health_sig else False
        self._last_health_sig = sig

        now = time.time()
        if health.get("online"):
            if (not was_online) or (now - self._last_meta_refresh >= 30.0):
                self._last_meta_refresh = now
                sites = await self.gpu.sites()
                if sites is not None:
                    self.store.set_setting("sites", sites)
                schema = await self.gpu.tuning_schema()
                if schema is not None:
                    self.store.set_setting("tuning_schema", schema)
                sys_info = await self.gpu.system()
                if sys_info is not None:
                    self.store.set_setting("gpu_system", sys_info)

            models_state = str((health.get("models") or {}).get("state") or "ready").lower()
            if not health.get("busy") and models_state not in ("loading", "error"):
                self._wake.set()

        if changed:
            self.broker.publish("global", "gpu_status", health)

        return health

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(max(0.2, self.settings.health_interval_s))
                if self._stop.is_set():
                    break
                await self.probe_health_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass

    def _gpu_ready(self, health: dict[str, Any] | None) -> bool:
        if not health or not health.get("online"):
            return False
        if health.get("busy"):
            return False
        models_state = str((health.get("models") or {}).get("state") or "ready").lower()
        return models_state not in ("loading", "error")

    async def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._wake.clear()
                if self.store.queue_length() > 0:
                    health = self.store.get_setting("gpu_health")
                    if not self._gpu_ready(health):
                        health = await self.probe_health_once()
                    if self._gpu_ready(health):
                        job_id = self.store.dequeue_job()
                        if job_id:
                            job = self.store.get_job(job_id)
                            if job and job.get("status") == "queued":
                                await self._execute_job(job)
                                # Immediately check if another job is waiting
                                continue
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=max(0.1, self.settings.poll_interval_s),
                    )
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.5)

    async def _materialize_artifact(self, key: str, suffix: str = "") -> tuple[Path, bool]:
        """Return (local_path, should_unlink_after) for submitting an uploaded artifact to GPU."""
        lp = self.artifacts.local_path(key)
        if lp is not None:
            return lp, False
        fd, tmp_name = tempfile.mkstemp(prefix="ef_up_", suffix=suffix or Path(key).suffix)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in self.artifacts.open_stream(key):
                    fh.write(chunk)
            return tmp_path, True
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    async def _execute_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        is_site = job.get("job_type") == "site"
        temp_paths: list[Path] = []
        now = time.time()

        updated = self.store.update_job(job_id, status="running", started_at=now)
        if updated:
            self.broker.publish("global", "job_update", updated)
            self.broker.publish(f"job:{job_id}", "status", updated)

        try:
            if is_site:
                form_fields = dict(job.get("site_form") or {})
                files_map: dict[str, tuple[str, Path, str]] = {}
                for field_name, meta in (job.get("site_files") or {}).items():
                    art_key = meta["key"]
                    lp, is_tmp = await self._materialize_artifact(art_key)
                    if is_tmp:
                        temp_paths.append(lp)
                    files_map[field_name] = (
                        meta.get("filename") or lp.name,
                        lp,
                        meta.get("content_type") or "application/octet-stream",
                    )
                await self.gpu.submit_site_job(job_id, form_fields, files_map)
            else:
                vpath: Path | None = None
                vname: str | None = None
                if upload_key := job.get("upload_key"):
                    lp, is_tmp = await self._materialize_artifact(upload_key)
                    if is_tmp:
                        temp_paths.append(lp)
                    vpath = lp
                    vname = job.get("name") or lp.name
                await self.gpu.submit_job(
                    job_id=job_id,
                    video_path=vpath,
                    video_name=vname,
                    source=job.get("source") or None,
                    options=job.get("options") or {},
                )
        except GPUClientError as exc:
            if exc.retryable:
                # Re-queue so another attempt runs once GPU is online/idle
                requeued = self.store.update_job(job_id, status="queued", started_at=None)
                self.store.enqueue_job(job_id)
                if requeued:
                    self.broker.publish("global", "job_update", requeued)
                    self.broker.publish(f"job:{job_id}", "status", requeued)
                return
            failed = self.store.update_job(
                job_id,
                status="error",
                error=exc.detail,
                finished_at=time.time(),
            )
            if failed:
                self.broker.publish("global", "job_update", failed)
                self.broker.publish(f"job:{job_id}", "done", failed)
            return
        finally:
            for tp in temp_paths:
                tp.unlink(missing_ok=True)

        await self._poll_running_job(job_id, is_site=is_site)

    async def _poll_running_job(self, job_id: str, is_site: bool) -> None:
        last_status_sig: tuple[Any, ...] | None = None
        last_event_seq = 0
        last_map_sig: tuple[Any, ...] | None = None
        consecutive_errors = 0

        while not self._stop.is_set():
            await asyncio.sleep(max(0.15, self.settings.poll_interval_s))
            if self._stop.is_set():
                return

            try:
                gpu_st = (
                    await self.gpu.site_job_status(job_id)
                    if is_site
                    else await self.gpu.job_status(job_id)
                )
                consecutive_errors = 0
            except GPUClientError as exc:
                consecutive_errors += 1
                if consecutive_errors < 10:
                    continue
                failed = self.store.update_job(
                    job_id,
                    status="error",
                    error=f"Mất kết nối GPU khi đang chạy: {exc.detail}",
                    finished_at=time.time(),
                )
                if failed:
                    self.broker.publish("global", "job_update", failed)
                    self.broker.publish(f"job:{job_id}", "done", failed)
                return

            if gpu_st is None:
                failed = self.store.update_job(
                    job_id,
                    status="error",
                    error="Không tìm thấy job trên GPU Worker.",
                    finished_at=time.time(),
                )
                if failed:
                    self.broker.publish("global", "job_update", failed)
                    self.broker.publish(f"job:{job_id}", "done", failed)
                return

            g_status = str(gpu_st.get("status") or "running")
            patch: dict[str, Any] = {
                "stage": int(gpu_st.get("stage") or 0),
                "stages": gpu_st.get("stages") or [],
                "progress": float(gpu_st.get("progress") or 0.0),
                "log": gpu_st.get("log") or [],
            }
            if gpu_st.get("key"):
                patch["key"] = gpu_st["key"]
            if gpu_st.get("error"):
                patch["error"] = gpu_st["error"]

            if g_status == "done":
                self.store.update_job(job_id, **patch)
                await self._sync_job_artifacts(job_id, is_site=is_site, gpu_status=gpu_st)
                return

            if g_status in ("error", "cancelled"):
                patch["status"] = g_status
                patch["finished_at"] = time.time()
                done_job = self.store.update_job(job_id, **patch)
                if done_job:
                    self.broker.publish("global", "job_update", done_job)
                    self.broker.publish(f"job:{job_id}", "status", done_job)
                    self.broker.publish(f"job:{job_id}", "done", done_job)
                return

            updated = self.store.update_job(job_id, **patch)
            sig = (
                patch["status"] if "status" in patch else "running",
                patch["stage"],
                round(patch["progress"], 3),
                len(patch["log"]),
            )
            if updated and sig != last_status_sig:
                last_status_sig = sig
                self.broker.publish(f"job:{job_id}", "status", updated)

            if not is_site:
                le = await self.gpu.live_events(job_id, after=last_event_seq)
                if le:
                    evs = le.get("events") or []
                    for ev in evs:
                        seq = int(ev.get("seq") or 0)
                        if seq > last_event_seq:
                            last_event_seq = seq
                    if evs or le.get("status") or le.get("meta"):
                        self.broker.publish(f"job:{job_id}", "live", le)

                md = await self.gpu.live_map_data(job_id, full=0)
                if md:
                    msig = (
                        bool(md.get("has_map")),
                        md.get("version"),
                        len(md.get("people") or []),
                        tuple(
                            (p.get("id"), p.get("x"), p.get("y"), p.get("state"))
                            for p in (md.get("people") or [])
                        ),
                    )
                    if msig != last_map_sig:
                        last_map_sig = msig
                        self.broker.publish(f"job:{job_id}", "map", md)

    async def _sync_job_artifacts(
        self,
        job_id: str,
        is_site: bool,
        gpu_status: dict[str, Any],
    ) -> None:
        syncing = self.store.update_job(job_id, status="syncing", progress=1.0)
        if syncing:
            self.broker.publish("global", "job_update", syncing)
            self.broker.publish(f"job:{job_id}", "status", syncing)

        prefix = f"site_jobs/{job_id}" if is_site else f"jobs/{job_id}"
        try:
            manifest = await self.gpu.list_artifacts(job_id, is_site=is_site)
            if not manifest:
                raise RuntimeError("Không lấy được danh sách artifact từ GPU Worker.")

            key = manifest.get("key") or gpu_status.get("key")
            files = manifest.get("files") or []
            for item in files:
                rel_path = str(item.get("rel_path") or "").lstrip("/")
                if not rel_path:
                    continue
                dst_key = f"{prefix}/{rel_path}"
                ok = await self.gpu.download_artifact_to_store(
                    job_id=job_id,
                    rel_path=rel_path,
                    store=self.artifacts,
                    dst_key=dst_key,
                    is_site=is_site,
                    content_type=item.get("content_type"),
                )
                if not ok:
                    raise RuntimeError(f"Thiếu file artifact khi đồng bộ: {rel_path}")

            job = self.store.get_job(job_id) or {"job_id": job_id, "name": job_id}
            n_alerts, summary, resolved_key = self._ingest_results(
                job=job,
                prefix=prefix,
                is_site=is_site,
                fallback_key=key,
            )

            done_job = self.store.update_job(
                job_id,
                status="done",
                progress=1.0,
                key=resolved_key or key,
                artifacts_synced=True,
                n_alerts=n_alerts,
                summary=summary,
                finished_at=time.time(),
            )
            if done_job:
                self.broker.publish("global", "job_update", done_job)
                self.broker.publish(f"job:{job_id}", "status", done_job)
                self.broker.publish(f"job:{job_id}", "done", done_job)
        except Exception as exc:  # noqa: BLE001
            failed = self.store.update_job(
                job_id,
                status="error",
                error=f"Lỗi đồng bộ artifact: {exc}",
                finished_at=time.time(),
            )
            if failed:
                self.broker.publish("global", "job_update", failed)
                self.broker.publish(f"job:{job_id}", "done", failed)

    def _ingest_results(
        self,
        job: dict[str, Any],
        prefix: str,
        is_site: bool,
        fallback_key: str | None,
    ) -> tuple[int, dict[str, Any], str | None]:
        job_id = job["job_id"]
        job_name = job.get("name") or job_id
        alerts_to_save: list[dict[str, Any]] = []
        resolved_key = fallback_key

        if is_site:
            rkey = f"{prefix}/site_result.json"
            if self.artifacts.exists(rkey):
                res = json.loads(self.artifacts.get_bytes(rkey).decode("utf-8"))
                for idx, ev in enumerate(res.get("events") or []):
                    ev_id = str(ev.get("id") or f"s{idx + 1}")
                    cams = ev.get("cams") or []
                    alerts_to_save.append(
                        {
                            "event_id": ev_id,
                            "job_name": job_name,
                            "job_type": "site",
                            "cam": ",".join(str(c) for c in cams),
                            "track_id": ev.get("person_id"),
                            "artifact_id": ev.get("artifact"),
                            "decision": "ANOMALY",
                            "behavior": ev.get("behavior") or "touching",
                            "confidence": ev.get("conf"),
                            "evidence": ev.get("evidence")
                            or "Engine phát hiện hành vi bất thường.",
                            "start_s": ev.get("t0"),
                            "end_s": ev.get("t1"),
                            "img_url": None,
                        }
                    )
        else:
            rkey = f"{prefix}/result.json"
            if self.artifacts.exists(rkey):
                res = json.loads(self.artifacts.get_bytes(rkey).decode("utf-8"))
                video_meta = res.get("video") or {}
                resolved_key = video_meta.get("key") or resolved_key
                cam_name = video_meta.get("name") or job_name
                alert_by_ev = {
                    str(a.get("event_id")): a
                    for a in (res.get("alerts") or [])
                    if a.get("event_id") is not None
                }
                events = res.get("events") or []
                if not events and res.get("alerts"):
                    events = res["alerts"]
                for idx, ev in enumerate(events):
                    ev_id = str(ev.get("event_id") or f"e{idx + 1}")
                    matched = alert_by_ev.get(ev_id) or {}
                    decision = str(ev.get("decision") or "ANOMALY").upper()
                    behavior = ev.get("behavior")
                    if not behavior:
                        behavior = "uncertain" if decision == "CANDIDATE" else "normal"
                    conf = ev.get("vlm_confidence")
                    if conf is None:
                        conf = matched.get("confidence", ev.get("confidence"))
                    evidence = (
                        matched.get("evidence")
                        or ev.get("vlm_evidence")
                        or ev.get("evidence")
                        or ""
                    )
                    task = ev.get("task")
                    img_url: str | None = None
                    if task and resolved_key:
                        sheet_rel = f"{resolved_key}/vlm_inputs/{task}/sheet.jpg"
                        if self.artifacts.exists(f"{prefix}/files/{sheet_rel}"):
                            img_url = f"/api/jobs/{job_id}/files/{sheet_rel}"
                    alerts_to_save.append(
                        {
                            "event_id": ev_id,
                            "job_name": job_name,
                            "job_type": "single",
                            "cam": cam_name,
                            "track_id": ev.get("track_id"),
                            "artifact_id": ev.get("artifact_id"),
                            "decision": decision,
                            "behavior": behavior,
                            "confidence": conf,
                            "evidence": evidence,
                            "start_s": ev.get("start_time_s", ev.get("start_s")),
                            "end_s": ev.get("end_time_s", ev.get("end_s")),
                            "task": task,
                            "img_url": img_url,
                        }
                    )

        saved = self.store.save_alerts(job_id, alerts_to_save)
        anomalies = sum(1 for a in saved if a.get("decision") == "ANOMALY")
        candidates = sum(1 for a in saved if a.get("decision") == "CANDIDATE")
        normal = sum(1 for a in saved if a.get("decision") == "NORMAL")
        summary = {
            "total": len(saved),
            "anomalies": anomalies,
            "candidates": candidates,
            "normal": normal,
        }
        return anomalies, summary, resolved_key
