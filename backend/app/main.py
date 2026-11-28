"""FastAPI application factory and Browser API routes for ExhibitFlow CPU backend."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .artifacts import ArtifactStore, create_artifact_store, validate_key
from .auth import AuthManager, require_header_auth, require_header_or_stream_ticket
from .config import Settings, load_settings
from .dispatcher import JobDispatcher
from .gpu_client import GPUClient, compute_gpu_ready
from .metrics import MetricsRegistry, PrometheusASGIMiddleware
from .models import (
    DEFAULT_STAGES,
    SITE_STAGES,
    SITE_STAGES_EV,
    AlertReviewUpdate,
    GpuConfigResponse,
    GpuConfigUpdate,
)
from .sse import EventBroker, format_sse
from .store import StateStore, create_state_store

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _sniff_video_ext(head: bytes) -> str | None:
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".mov" if head[8:12] == b"qt  " else ".mp4"
    if head[:4] == bytes.fromhex("1a45dfa3"):
        return ".mkv"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return ".avi"
    return None


def _safe_filename(name: str | None, fallback: str) -> str:
    raw = Path(name or fallback).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or fallback


def _backend_system_stats() -> dict[str, Any]:
    cpu_pct = psutil.cpu_percent(interval=None)
    try:
        load_1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        load_1 = round(cpu_pct / 100.0 * (os.cpu_count() or 1), 2)
    vm = psutil.virtual_memory()
    return {
        "cpu": {"load": round(float(load_1), 2), "pct": round(float(cpu_pct), 1)},
        "ram": {
            "used_gb": round(vm.used / (1024**3), 1),
            "total_gb": round(vm.total / (1024**3), 1),
        },
        "cores": {
            "logical": os.cpu_count() or 1,
            "usable": os.cpu_count() or 1,
        },
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the ExhibitFlow CPU FastAPI application."""
    cfg = settings or load_settings()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    store: StateStore = create_state_store(cfg)
    artifacts: ArtifactStore = create_artifact_store(cfg)

    persisted_url = store.get_setting("gpu_url")
    persisted_token = store.get_setting("gpu_token")
    gpu_url = str(persisted_url if persisted_url is not None else cfg.gpu_worker_url).strip().rstrip("/")
    gpu_token = str(persisted_token if persisted_token is not None else cfg.gpu_token)

    gpu = GPUClient(base_url=gpu_url, token=gpu_token)
    broker = EventBroker()
    auth = AuthManager(api_key=cfg.backend_api_key, stream_ttl_s=cfg.stream_token_ttl_s)
    metrics = MetricsRegistry()
    dispatcher = JobDispatcher(
        settings=cfg,
        store=store,
        artifacts=artifacts,
        gpu=gpu,
        broker=broker,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await dispatcher.start()
        try:
            yield
        finally:
            await dispatcher.stop()
            await gpu.aclose()
            store.close()

    app = FastAPI(title="ExhibitFlow Backend API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list(),
        allow_methods=["*"],
        allow_headers=[
            "Authorization",
            "X-Backend-Key",
            "Content-Type",
            "Range",
            "ngrok-skip-browser-warning",
        ],
        expose_headers=[
            "Content-Range",
            "Accept-Ranges",
            "Content-Length",
            "Content-Disposition",
        ],
    )

    app.state.settings = cfg
    app.state.store = store
    app.state.artifacts = artifacts
    app.state.gpu = gpu
    app.state.broker = broker
    app.state.auth = auth
    app.state.metrics = metrics
    app.state.dispatcher = dispatcher

    app.add_middleware(PrometheusASGIMiddleware, metrics=metrics)

    hdr_auth = [Depends(require_header_auth)]
    media_auth = [Depends(require_header_or_stream_ticket)]

    # ---- Helpers ----
    def _enrich_job(job: dict[str, Any]) -> dict[str, Any]:
        out = dict(job)
        if out.get("status") == "queued":
            out["queue_position"] = store.queue_position(out["job_id"])
        else:
            out["queue_position"] = None
        out.setdefault("stages", DEFAULT_STAGES)
        out.setdefault("stage", 0)
        out.setdefault("progress", 0.0)
        out.setdefault("key", None)
        out.setdefault("error", None)
        out.setdefault("realtime", False)
        out.setdefault("job_type", "single")
        out.setdefault("artifacts_synced", False)
        out.setdefault("n_alerts", 0)
        out.setdefault("summary", None)
        out.setdefault("site_id", None)
        out.setdefault("cameras", None)
        out.setdefault("log", [])
        return out

    def _get_job_or_404(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Không tìm thấy job này")
        return job

    def _serve_artifact(
        art_key: str,
        media_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        try:
            norm = validate_key(art_key)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        lp = artifacts.local_path(norm)
        if lp is not None:
            return FileResponse(lp, media_type=media_type, headers=headers)
        if artifacts.exists(norm):
            st = artifacts.stat(norm)
            ct = media_type or (st.content_type if st else "application/octet-stream")
            return StreamingResponse(
                artifacts.open_stream(norm), media_type=ct, headers=headers
            )
        raise HTTPException(404, "Không tìm thấy file artifact")

    async def _save_upload_to_store(
        upload: UploadFile, dst_key: str, validate_video: bool = True
    ) -> tuple[str, str]:
        orig_name = upload.filename or "upload.mp4"
        ext = Path(orig_name).suffix.lower()
        head = b""
        if validate_video and ext not in ALLOWED_VIDEO_EXTS:
            head = await upload.read(1 << 16)
            ext = _sniff_video_ext(head) or ext
            if ext not in ALLOWED_VIDEO_EXTS:
                raise HTTPException(
                    400,
                    f"Định dạng video không hỗ trợ ({ext or 'không có đuôi'}). Dùng: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}",
                )
        fd, tmp_name = tempfile.mkstemp(prefix="ef_recv_", suffix=ext or ".bin")
        tmp_path = Path(tmp_name)
        total_size = 0
        try:
            with os.fdopen(fd, "wb") as fh:
                if head:
                    fh.write(head)
                    total_size += len(head)
                    if total_size > cfg.max_upload_bytes:
                        raise HTTPException(
                            413, f"File vượt quá giới hạn {cfg.max_upload_mb} MB"
                        )
                while chunk := await upload.read(1 << 20):
                    total_size += len(chunk)
                    if total_size > cfg.max_upload_bytes:
                        raise HTTPException(
                            413, f"File vượt quá giới hạn {cfg.max_upload_mb} MB"
                        )
                    fh.write(chunk)
            stored_key = artifacts.put_file(
                dst_key,
                tmp_path,
                content_type=upload.content_type or None,
            )
            return stored_key, orig_name
        finally:
            tmp_path.unlink(missing_ok=True)

    # ---- Health, Auth, System, Config, Metadata ----
    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        store_kind = "redis" if cfg.redis_url else "sqlite"
        authenticated = auth.verify_request_header(request)
        if auth.enabled and not authenticated:
            return {
                "ok": True,
                "version": __version__,
                "auth_required": True,
                "authenticated": False,
                "store": store_kind,
                "artifact_store": cfg.artifact_store,
            }
        cached_gpu = store.get_setting("gpu_health")
        if cached_gpu is None:
            cached_gpu = await dispatcher.probe_health_once()
        gpu_block = {
            "online": bool(cached_gpu.get("online")),
            "ready": compute_gpu_ready(cached_gpu),
            "busy": bool(cached_gpu.get("busy")),
            "demo": bool(cached_gpu.get("demo")),
            "models": cached_gpu.get("models") or {"state": "unknown", "msg": ""},
            "vlm": cached_gpu.get("vlm") or {"state": "unknown"},
            "error": cached_gpu.get("error"),
            "url": cached_gpu.get("url") or gpu.base_url,
        }
        return {
            "ok": True,
            "version": __version__,
            "auth_required": auth.enabled,
            "authenticated": True,
            "gpu": gpu_block,
            "queue_length": store.queue_length(),
            "store": store_kind,
            "artifact_store": cfg.artifact_store,
        }

    @app.post("/api/auth/verify", dependencies=hdr_auth)
    def verify_auth() -> dict[str, Any]:
        return {
            "ok": True,
            "auth_required": auth.enabled,
            "authenticated": True,
        }

    @app.post("/api/auth/stream-token", dependencies=hdr_auth)
    def create_stream_token() -> dict[str, Any]:
        return auth.issue_stream_ticket()

    @app.get("/api/system", dependencies=hdr_auth)
    async def system_stats() -> dict[str, Any]:
        be = _backend_system_stats()
        gpu_health = store.get_setting("gpu_health") or {}
        gpu_sys = None
        if gpu_health.get("online"):
            gpu_sys = await gpu.system()
            if gpu_sys is not None:
                store.set_setting("gpu_system", gpu_sys)
        if gpu_sys is None:
            gpu_sys = store.get_setting("gpu_system") or {}
        return {
            "cpu": be["cpu"],
            "ram": be["ram"],
            "gpus": gpu_sys.get("gpus") or [],
            "cores": be["cores"],
            "backend": be,
            "gpu": gpu_sys if gpu_sys else None,
        }

    @app.get("/api/config/gpu", response_model=GpuConfigResponse, dependencies=hdr_auth)
    def get_gpu_config() -> GpuConfigResponse:
        return GpuConfigResponse(url=gpu.base_url, has_token=bool(gpu.token))

    @app.put("/api/config/gpu", dependencies=hdr_auth)
    async def update_gpu_config(payload: GpuConfigUpdate) -> dict[str, Any]:
        new_url = (payload.url or "").strip().rstrip("/")
        if new_url:
            parsed = urlparse(new_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise HTTPException(400, "GPU Worker URL phải là http:// hoặc https:// hợp lệ")

        if payload.clear_token:
            new_token = ""
        elif payload.token is not None and payload.token != "":
            new_token = payload.token
        else:
            new_token = gpu.token

        store.set_setting("gpu_url", new_url)
        store.set_setting("gpu_token", new_token)
        gpu.configure(new_url, new_token)
        gpu_health = await dispatcher.probe_health_once()
        return {
            "ok": True,
            "url": gpu.base_url,
            "has_token": bool(gpu.token),
            "gpu": gpu_health,
        }

    @app.get("/api/tuning/schema", dependencies=hdr_auth)
    async def get_tuning_schema() -> dict[str, Any]:
        cached_health = store.get_setting("gpu_health") or {}
        if cached_health.get("online"):
            live = await gpu.tuning_schema()
            if live is not None:
                store.set_setting("tuning_schema", live)
                return live
        return store.get_setting("tuning_schema") or {}

    @app.get("/api/sites", dependencies=hdr_auth)
    async def get_sites() -> list[dict[str, Any]]:
        cached_health = store.get_setting("gpu_health") or {}
        if cached_health.get("online"):
            live = await gpu.sites()
            if live is not None:
                store.set_setting("sites", live)
                return live
        cached = store.get_setting("sites")
        return cached if isinstance(cached, list) else []

    # ---- Global SSE ----
    @app.get("/api/events", dependencies=media_auth)
    async def global_events() -> StreamingResponse:
        cached_gpu = store.get_setting("gpu_health")
        initial = [("gpu_status", cached_gpu)] if cached_gpu is not None else None
        return StreamingResponse(
            broker.subscribe("global", initial=initial),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- Single-cam Jobs ----
    @app.post("/api/jobs", dependencies=hdr_auth)
    async def create_job(
        video: UploadFile | None = File(None),
        source: str = Form(""),
        options: str = Form("{}"),
    ) -> dict[str, Any]:
        try:
            opts = json.loads(options or "{}")
            if not isinstance(opts, dict):
                opts = {}
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"options không phải JSON hợp lệ: {exc}") from exc

        src_clean = (source or "").strip()
        if (video is None or not video.filename) and not src_clean:
            raise HTTPException(400, "Cần gửi file video hoặc đường dẫn / URL nguồn")

        job_id = uuid.uuid4().hex[:12]
        upload_key: str | None = None
        if video is not None and video.filename:
            safe_name = _safe_filename(video.filename, "video.mp4")
            upload_key, job_name = await _save_upload_to_store(
                video, f"uploads/{job_id}/{safe_name}", validate_video=True
            )
        else:
            parsed = urlparse(src_clean)
            job_name = Path(parsed.path or src_clean).name or src_clean[:60]

        now = time.time()
        record = {
            "job_id": job_id,
            "job_type": "single",
            "status": "queued",
            "stage": 0,
            "stages": list(DEFAULT_STAGES),
            "progress": 0.0,
            "key": None,
            "error": None,
            "name": job_name,
            "source": src_clean,
            "upload_key": upload_key,
            "options": opts,
            "realtime": bool(opts.get("realtime")),
            "artifacts_synced": False,
            "n_alerts": 0,
            "summary": None,
            "site_id": None,
            "cameras": None,
            "log": [],
            "created_at": now,
            "finished_at": None,
        }
        store.create_job(record)
        pos = store.enqueue_job(job_id)
        enriched = _enrich_job(record)
        enriched["queue_position"] = pos
        broker.publish("global", "job_update", enriched)
        dispatcher.notify()
        return {"job_id": job_id, "status": "queued", "queue_position": pos}

    @app.get("/api/jobs", dependencies=hdr_auth)
    def list_jobs(
        status: str | None = None,
        job_type: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        items, total = store.list_jobs(
            status=status, job_type=job_type, q=q, limit=limit, offset=offset
        )
        return {"items": [_enrich_job(j) for j in items], "total": total}

    @app.get("/api/jobs/{job_id}", dependencies=hdr_auth)
    def get_job(job_id: str) -> dict[str, Any]:
        return _enrich_job(_get_job_or_404(job_id))

    @app.delete("/api/jobs/{job_id}", dependencies=hdr_auth)
    def delete_job(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.get("status") in ("running", "syncing"):
            raise HTTPException(
                409, "Job đang chạy hoặc đang đồng bộ; hãy hủy trước khi xoá."
            )
        store.remove_from_queue(job_id)
        store.delete_job_alerts(job_id)
        artifacts.delete_prefix(f"jobs/{job_id}/")
        artifacts.delete_prefix(f"site_jobs/{job_id}/")
        artifacts.delete_prefix(f"uploads/{job_id}/")
        store.delete_job(job_id)
        broker.publish("global", "job_update", {"job_id": job_id, "deleted": True})
        return {"deleted": True, "job_id": job_id}

    @app.post("/api/jobs/{job_id}/cancel", dependencies=hdr_auth)
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        st = job.get("status")
        if st in ("done", "error", "cancelled"):
            return {"status": st}
        if st == "queued":
            store.remove_from_queue(job_id)
            updated = store.update_job(
                job_id, status="cancelled", finished_at=time.time()
            )
            if updated:
                enriched = _enrich_job(updated)
                broker.publish("global", "job_update", enriched)
                broker.publish(f"job:{job_id}", "status", enriched)
                broker.publish(f"job:{job_id}", "done", enriched)
            return {"status": "cancelled"}
        await gpu.cancel_job(job_id)
        return {"status": "cancelling"}

    @app.post("/api/jobs/{job_id}/finish", dependencies=hdr_auth)
    async def finish_job(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        st = job.get("status")
        if st in ("done", "error", "cancelled"):
            return {"status": st}
        if st == "running":
            await gpu.finish_job(job_id)
            return {"status": "finishing"}
        return {"status": st}

    # ---- Per-Job Live & SSE ----
    @app.get("/api/jobs/{job_id}/events", dependencies=media_auth)
    async def job_events(job_id: str) -> StreamingResponse:
        job = _enrich_job(_get_job_or_404(job_id))
        if job["status"] in ("done", "error", "cancelled"):
            async def _terminal_stream():
                yield format_sse("status", job)
                yield format_sse("done", job)

            return StreamingResponse(
                _terminal_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        initial: list[tuple[str, Any]] = [("status", job)]
        return StreamingResponse(
            broker.subscribe(f"job:{job_id}", initial=initial),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/live", dependencies=media_auth)
    async def live_frame(job_id: str) -> Response:
        job = _get_job_or_404(job_id)
        if job.get("status") != "running":
            raise HTTPException(404, "Job không ở trạng thái đang chạy")
        jpg = await gpu.live_frame(job_id)
        if not jpg:
            raise HTTPException(404, "Chưa có khung hình")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/live/stream", dependencies=media_auth)
    async def live_stream(job_id: str) -> StreamingResponse:
        _get_job_or_404(job_id)

        async def _proxy_stream():
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                cur = store.get_job(job_id)
                if cur is None or cur.get("status") in ("done", "error", "cancelled"):
                    return
                if cur.get("status") == "running":
                    break
                await asyncio.sleep(0.1)
            async for chunk in gpu.stream_mjpeg(job_id):
                yield chunk

        return StreamingResponse(
            _proxy_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/jobs/{job_id}/live/events", dependencies=hdr_auth)
    async def live_events(job_id: str, after: int = 0) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.get("status") == "running":
            res = await gpu.live_events(job_id, after=after)
            if res is not None:
                return res
        return {"meta": None, "status": None, "events": [], "job": job.get("status")}

    # ---- Synced Single-Cam Replay Artifacts ----
    @app.get("/api/jobs/{job_id}/result", dependencies=hdr_auth)
    async def job_result(job_id: str) -> Response:
        job = _get_job_or_404(job_id)
        if job.get("artifacts_synced"):
            return _serve_artifact(f"jobs/{job_id}/result.json", media_type="application/json")
        if job.get("status") == "done":
            raise HTTPException(404, "Không tìm thấy result.json")
        raise HTTPException(409, "Job chưa xong")

    @app.get("/api/jobs/{job_id}/video", dependencies=media_auth)
    async def job_video(job_id: str, kind: str = "live") -> Response:
        job = _get_job_or_404(job_id)
        upload_key = job.get("upload_key")
        if kind == "original":
            if upload_key and artifacts.exists(upload_key):
                return _serve_artifact(upload_key, media_type="video/mp4")
            raise HTTPException(404, "Không còn video gốc")

        live_key = f"jobs/{job_id}/live.mp4"
        if job.get("artifacts_synced"):
            if artifacts.exists(live_key):
                return _serve_artifact(live_key, media_type="video/mp4")
            if upload_key and artifacts.exists(upload_key):
                return _serve_artifact(upload_key, media_type="video/mp4")
            raise HTTPException(404, "Không có video kết quả")

        if job.get("status") == "running":
            raw = await gpu.fetch_artifact_bytes(job_id, "live.mp4")
            if raw is not None:
                return Response(raw, media_type="video/mp4")
        if upload_key and artifacts.exists(upload_key):
            return _serve_artifact(upload_key, media_type="video/mp4")
        raise HTTPException(404, "Chưa có video")

    @app.get("/api/jobs/{job_id}/map", dependencies=media_auth)
    async def job_map(job_id: str) -> Response:
        job = _get_job_or_404(job_id)
        map_key = f"jobs/{job_id}/map.png"
        if job.get("artifacts_synced"):
            if artifacts.exists(map_key):
                return _serve_artifact(
                    map_key,
                    media_type="image/png",
                    headers={"Cache-Control": "no-store"},
                )
            raise HTTPException(404, "Không có ảnh map")
        if job.get("status") == "running":
            png = await gpu.live_map(job_id)
            if png:
                return Response(
                    png, media_type="image/png", headers={"Cache-Control": "no-store"}
                )
        raise HTTPException(404, "Chưa có map")

    @app.get("/api/jobs/{job_id}/map/data", dependencies=hdr_auth)
    async def job_map_data(job_id: str, full: int = 0) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        md_key = f"jobs/{job_id}/map_data.json"
        if job.get("artifacts_synced") and artifacts.exists(md_key):
            data = json.loads(artifacts.get_bytes(md_key).decode("utf-8"))
            if not full and isinstance(data, dict) and "frames" in data:
                data = {k: v for k, v in data.items() if k != "frames"}
            return data
        if job.get("status") == "running":
            live_md = await gpu.live_map_data(job_id, full=full)
            if live_md is not None:
                return live_md
        return {
            "has_map": False,
            "version": 0,
            "fps": 25.0,
            "people": [],
            "artifacts": [],
        }

    @app.get("/api/jobs/{job_id}/files/{path:path}", dependencies=media_auth)
    async def job_file(job_id: str, path: str) -> Response:
        job = _get_job_or_404(job_id)
        try:
            rel = validate_key(path)
        except ValueError as exc:
            raise HTTPException(400, f"Đường dẫn không hợp lệ: {exc}") from exc

        if job.get("artifacts_synced"):
            cand1 = f"jobs/{job_id}/files/{rel}"
            if artifacts.exists(cand1):
                return _serve_artifact(cand1)
            cand2 = f"jobs/{job_id}/{rel}"
            if artifacts.exists(cand2):
                return _serve_artifact(cand2)
            raise HTTPException(404, "Không tìm thấy file")

        if job.get("status") == "running":
            raw = await gpu.fetch_artifact_bytes(job_id, f"files/{rel}")
            if raw is None:
                raw = await gpu.fetch_artifact_bytes(job_id, rel)
            if raw is not None:
                return Response(raw)
        raise HTTPException(404, "Chưa có kết quả")

    # ---- Alerts & Review ----
    @app.get("/api/alerts/export.csv", dependencies=media_auth)
    def export_alerts_csv(
        job_id: str | None = None,
        decision: str | None = None,
        behavior: str | None = None,
        review: str | None = None,
        min_conf: float | None = None,
        q: str | None = None,
    ) -> Response:
        items, _, _ = store.list_alerts(
            job_id=job_id,
            decision=decision,
            behavior=behavior,
            review=review,
            min_conf=min_conf,
            q=q,
            limit=10000,
            offset=0,
        )
        buf = io.StringIO()
        buf.write("\ufeff")
        writer = csv.writer(buf)
        writer.writerow(
            [
                "alert_id",
                "job_id",
                "job_name",
                "event_id",
                "cam",
                "track_id",
                "artifact_id",
                "decision",
                "behavior",
                "confidence",
                "start_s",
                "end_s",
                "review",
                "evidence",
            ]
        )
        for a in items:
            writer.writerow(
                [
                    a.get("alert_id", ""),
                    a.get("job_id", ""),
                    a.get("job_name", ""),
                    a.get("event_id", ""),
                    a.get("cam", ""),
                    a.get("track_id", ""),
                    a.get("artifact_id", ""),
                    a.get("decision", ""),
                    a.get("behavior", ""),
                    "" if a.get("confidence") is None else a.get("confidence"),
                    "" if a.get("start_s") is None else a.get("start_s"),
                    "" if a.get("end_s") is None else a.get("end_s"),
                    a.get("review") or "",
                    a.get("evidence", ""),
                ]
            )
        return Response(
            content=buf.getvalue().encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="alerts.csv"'},
        )

    @app.get("/api/alerts", dependencies=hdr_auth)
    def list_alerts(
        job_id: str | None = None,
        decision: str | None = None,
        behavior: str | None = None,
        review: str | None = None,
        min_conf: float | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        items, total, counts = store.list_alerts(
            job_id=job_id,
            decision=decision,
            behavior=behavior,
            review=review,
            min_conf=min_conf,
            q=q,
            limit=limit,
            offset=offset,
        )
        return {"items": items, "total": total, "counts": counts}

    @app.patch("/api/alerts/{alert_id}", dependencies=hdr_auth)
    def review_alert(alert_id: str, payload: AlertReviewUpdate) -> dict[str, Any]:
        updated = store.update_alert_review(alert_id, payload.review)
        if updated is None:
            raise HTTPException(404, "Không tìm thấy cảnh báo")
        broker.publish(
            "global",
            "alert_reviewed",
            {
                "alert_id": updated["alert_id"],
                "job_id": updated["job_id"],
                "event_id": updated["event_id"],
                "review": updated["review"],
            },
        )
        return updated

    # ---- Calibrated Site Jobs ----
    @app.post("/api/site-jobs", dependencies=hdr_auth)
    async def create_site_job(request: Request) -> dict[str, Any]:
        form = await request.form()
        site_id = str(form.get("site") or "").strip()
        if not site_id:
            raise HTTPException(400, "Thiếu trường site")
        events_flag = str(form.get("events", "1")) != "0"

        job_id = uuid.uuid4().hex[:12]
        site_form: dict[str, str] = {"site": site_id, "events": "1" if events_flag else "0"}
        site_files: dict[str, dict[str, str]] = {}
        cam_videos: dict[str, str] = {}

        for field_name, val in form.multi_items():
            if field_name in ("site", "events"):
                continue
            if hasattr(val, "read") and getattr(val, "filename", None):
                is_calib = field_name == "calib_all" or field_name.startswith("calib_")
                safe_name = _safe_filename(val.filename, f"{field_name}.bin")
                sub = "calib" if is_calib else "cams"
                art_key = f"uploads/{job_id}/{sub}/{_safe_filename(field_name, 'f')}_{safe_name}"
                stored_key, orig_name = await _save_upload_to_store(
                    val, art_key, validate_video=not is_calib
                )
                site_files[field_name] = {
                    "key": stored_key,
                    "filename": orig_name,
                    "content_type": val.content_type or "application/octet-stream",
                }
                if not is_calib:
                    cam_id = (
                        f"C{len(cam_videos) + 1}"
                        if field_name.startswith("cam_")
                        else field_name
                    )
                    cam_videos[cam_id] = stored_key
            else:
                site_form[field_name] = str(val)

        if not cam_videos:
            artifacts.delete_prefix(f"uploads/{job_id}/")
            raise HTTPException(400, "Cần ít nhất một video camera cho site-job")

        cameras = list(cam_videos.keys())
        stages = list(SITE_STAGES_EV if events_flag else SITE_STAGES)
        now = time.time()
        record = {
            "job_id": job_id,
            "job_type": "site",
            "status": "queued",
            "stage": 0,
            "stages": stages,
            "progress": 0.0,
            "key": None,
            "error": None,
            "name": f"Site {site_id} ({len(cameras)} cam)",
            "source": "",
            "upload_key": None,
            "options": {"events": events_flag},
            "realtime": False,
            "artifacts_synced": False,
            "n_alerts": 0,
            "summary": None,
            "site_id": site_id,
            "cameras": cameras,
            "cam_videos": cam_videos,
            "site_form": site_form,
            "site_files": site_files,
            "log": [],
            "created_at": now,
            "finished_at": None,
        }
        store.create_job(record)
        pos = store.enqueue_job(job_id)
        enriched = _enrich_job(record)
        enriched["queue_position"] = pos
        broker.publish("global", "job_update", enriched)
        dispatcher.notify()
        return {
            "job_id": job_id,
            "status": "queued",
            "queue_position": pos,
            "cameras": cameras,
            "events": events_flag,
        }

    @app.get("/api/site-jobs/{job_id}", dependencies=hdr_auth)
    def get_site_job(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.get("job_type") != "site":
            raise HTTPException(404, "Không tìm thấy site-job này")
        enriched = _enrich_job(job)
        end_t = job.get("finished_at") or time.time()
        enriched["elapsed"] = max(0, round(float(end_t) - float(job.get("created_at") or end_t)))
        return enriched

    @app.get("/api/site-jobs/{job_id}/result", dependencies=hdr_auth)
    def site_job_result(job_id: str) -> Response:
        job = _get_job_or_404(job_id)
        if job.get("job_type") != "site":
            raise HTTPException(404, "Không tìm thấy site-job này")
        if job.get("artifacts_synced"):
            return _serve_artifact(
                f"site_jobs/{job_id}/site_result.json", media_type="application/json"
            )
        if job.get("status") == "done":
            raise HTTPException(404, "Không tìm thấy site_result.json")
        raise HTTPException(409, "Job chưa xong")

    @app.get("/api/site-jobs/{job_id}/map", dependencies=media_auth)
    def site_job_map(job_id: str) -> Response:
        job = _get_job_or_404(job_id)
        if job.get("job_type") != "site":
            raise HTTPException(404, "Không tìm thấy site-job này")
        map_key = f"site_jobs/{job_id}/global_2d_map.png"
        if job.get("artifacts_synced") and artifacts.exists(map_key):
            return _serve_artifact(
                map_key, media_type="image/png", headers={"Cache-Control": "no-store"}
            )
        raise HTTPException(404, "Không có ảnh mặt bằng")

    @app.get("/api/site-jobs/{job_id}/video/{cam}", dependencies=media_auth)
    def site_job_video(job_id: str, cam: str) -> Response:
        job = _get_job_or_404(job_id)
        if job.get("job_type") != "site":
            raise HTTPException(404, "Không tìm thấy site-job này")
        try:
            safe_cam = validate_key(cam)
        except ValueError as exc:
            raise HTTPException(400, f"Tên camera không hợp lệ: {exc}") from exc

        if "/" in safe_cam:
            raise HTTPException(400, "Tên camera không hợp lệ")

        if job.get("artifacts_synced"):
            exact = f"site_jobs/{job_id}/videos/{safe_cam}.mp4"
            if artifacts.exists(exact):
                return _serve_artifact(exact, media_type="video/mp4")
            for k in artifacts.list_keys(f"site_jobs/{job_id}/videos/"):
                if Path(k).stem == safe_cam:
                    return _serve_artifact(k, media_type="video/mp4")

        cam_videos = job.get("cam_videos") or {}
        if safe_cam in cam_videos and artifacts.exists(cam_videos[safe_cam]):
            return _serve_artifact(cam_videos[safe_cam], media_type="video/mp4")

        raise HTTPException(404, "Không có video camera này")

    @app.get("/metrics")
    def prometheus_metrics(request: Request) -> Response:
        if auth.enabled and not cfg.metrics_public and not auth.verify_request_header(request):
            raise HTTPException(
                status_code=401,
                detail="Cần xác thực Backend (Authorization: Bearer hoặc X-Backend-Key)",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = metrics.render_prometheus(
            store=store,
            broker=broker,
            gpu_ready_fn=compute_gpu_ready,
        )
        return Response(
            content=payload,
            media_type="text/plain; version=0.0.4; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    # Mount static web UI AFTER all /api routes
    if cfg.web_dir is not None and cfg.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(cfg.web_dir), html=True), name="web")

    return app


app = create_app()
