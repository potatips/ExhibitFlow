"""Standalone FastAPI GPU Worker service (/internal/* API)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from .auth import require_gpu_token, resolve_token, verify_ws_token
from .demo import DemoRuntime

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_MAX_MB = int(os.environ.get("EXHIBITFLOW_MAX_MB", "600"))
MAX_BYTES = DEFAULT_MAX_MB * 1024 * 1024
PKG_DIR = Path(__file__).resolve().parent.parent


def sniff_ext(head: bytes) -> Optional[str]:
    """Infer video container extension from magic bytes."""
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".mov" if head[8:12] == b"qt  " else ".mp4"
    if head[:4] == bytes.fromhex("1a45dfa3"):
        return ".mkv"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return ".avi"
    return None


def _cpu_cores() -> dict[str, Any]:
    info: dict[str, Any] = {"logical": os.cpu_count() or 1}
    try:
        info["usable"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["usable"] = info["logical"]
    quota: Optional[float] = None
    try:
        q, per = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()[:2]
        quota = None if q == "max" else int(q) / int(per)
    except Exception:  # noqa: BLE001
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8"))
            per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8"))
            quota = q / per if q > 0 else None
        except Exception:  # noqa: BLE001
            pass
    if quota:
        info["quota"] = round(quota, 1)
        info["usable"] = max(1, min(info["usable"], int(quota)))
    return info


def get_system_stats() -> dict[str, Any]:
    """Collect CPU, RAM, and optional GPU stats (works cleanly on CPU without nvidia-smi)."""
    out: dict[str, Any] = {"cpu": None, "ram": None, "gpus": [], "cores": _cpu_cores()}
    try:
        la = os.getloadavg()
        n = os.cpu_count() or 1
        out["cpu"] = {"load": round(la[0], 2), "pct": round(min(100.0, la[0] / n * 100), 1)}
    except (AttributeError, OSError):
        try:
            import psutil

            out["cpu"] = {"load": 0.0, "pct": round(float(psutil.cpu_percent()), 1)}
        except Exception:  # noqa: BLE001
            pass

    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                meminfo[k] = int(v.split()[0])
        tot = meminfo["MemTotal"] / 1048576.0
        avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) / 1048576.0
        out["ram"] = {"used_gb": round(tot - avail, 1), "total_gb": round(tot, 1)}
    except Exception:  # noqa: BLE001
        try:
            import psutil

            vm = psutil.virtual_memory()
            out["ram"] = {"used_gb": round(vm.used / 1e9, 1), "total_gb": round(vm.total / 1e9, 1)}
        except Exception:  # noqa: BLE001
            pass

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 4:
                    name, util, mu, mt = parts
                    out["gpus"].append(
                        {
                            "name": name,
                            "util": float(util),
                            "mem_used_gb": round(float(mu) / 1024.0, 1),
                            "mem_total_gb": round(float(mt) / 1024.0, 1),
                        }
                    )
    except Exception:  # noqa: BLE001
        pass

    return out


def _parse_det_conf(val: Any) -> Optional[float]:
    if val is None or val == "" or isinstance(val, bool):
        return None
    try:
        c = float(val)
    except (TypeError, ValueError):
        return None
    return c if 0.001 <= c <= 0.99 else None


def _source_roots() -> list[Path]:
    raw = os.environ.get("EXHIBITFLOW_SOURCE_ROOTS") or os.pathsep.join(
        ["/workspace", "/kaggle/input", "/kaggle/working", "/content", str(PKG_DIR)]
    )
    return [Path(p).resolve() for p in raw.split(os.pathsep) if p]


def create_worker_app(
    data_dir: str | Path = "./gpu_data",
    *,
    demo: bool = False,
    preload: bool = True,
    mode: Optional[str] = None,
    local: bool = False,
    token: Optional[str] = None,
) -> FastAPI:
    """Create the standalone GPU worker FastAPI application."""
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    det_conf = _parse_det_conf(os.environ.get("EXHIBITFLOW_DET_CONF"))
    if demo:
        runtime: Any = DemoRuntime(data_path, mode=mode or "full", det_conf=det_conf)
    else:
        # Lazy import ONLY when not demo so --demo and CPU unit tests NEVER require torch/cv2/ultralytics
        from .runtime import RealRuntime

        runtime = RealRuntime(
            data_path,
            preload=preload,
            mode=mode,
            local=local,
            det_conf=det_conf,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="ExhibitFlow GPU Worker", version="ef-gpu-1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.gpu_token = resolve_token(token)
    app.state.demo = demo
    app.state.runtime = runtime

    auth_dep = [Depends(require_gpu_token)]

    # ------------------------------------------------------------------ Helpers
    def _check_ext(ext: str) -> None:
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                400,
                f'Unsupported format ({ext or "no extension"}). Allowed: {", ".join(sorted(ALLOWED_VIDEO_EXTS))}',
            )

    async def _save_upload(video: UploadFile, dst_path: Optional[Path] = None) -> tuple[Path, str]:
        filename = video.filename or "upload.mp4"
        ext = Path(filename).suffix.lower()
        head = b""
        if ext not in ALLOWED_VIDEO_EXTS:
            head = await video.read(1 << 16)
            ext = sniff_ext(head) or ext
        _check_ext(ext)
        vpath = Path(f"{dst_path}{ext}") if dst_path is not None else (uploads_dir / f"{uuid.uuid4().hex[:10]}{ext}")
        size = 0
        with open(vpath, "wb") as fh:
            if head:
                fh.write(head)
                size = len(head)
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    fh.close()
                    vpath.unlink(missing_ok=True)
                    raise HTTPException(413, f"Video too large (max {MAX_BYTES >> 20} MB)")
                fh.write(chunk)
        return vpath, filename

    def _resolve_source(src: str) -> tuple[Path, str]:
        if re.match(r"^(rtsps?|rtmps?)://", src, re.I) or re.match(r"^push://([\w-]+)(?:\?fps=(\d+))?$", src):
            if hasattr(runtime, "resolve_stream_source"):
                return runtime.resolve_stream_source(src, uploads_dir)
            stream_file = uploads_dir / f"cam_{uuid.uuid4().hex[:8]}.efstream"
            stream_file.write_text(json.dumps({"url": src}), encoding="utf-8")
            return stream_file, src

        if re.match(r"^https?://", src, re.I):
            ext = Path(urllib.parse.urlparse(src).path).suffix.lower() or ".mp4"
            _check_ext(ext)
            dst = uploads_dir / f"{uuid.uuid4().hex[:8]}{ext}"
            size = 0
            try:
                with urllib.request.urlopen(src, timeout=30) as resp, open(dst, "wb") as fh:
                    while chunk := resp.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, f"Video too large (max {MAX_BYTES >> 20} MB)")
                        fh.write(chunk)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"Cannot download URL: {exc}") from exc
            return dst, Path(urllib.parse.urlparse(src).path).name or dst.name

        p = Path(src).expanduser().resolve()
        roots = _source_roots()
        if not any(r == p or r in p.parents for r in roots):
            raise HTTPException(403, f"Source path outside allowed roots: {', '.join(map(str, roots))}")
        if p.is_dir():
            vids = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTS)
            if not vids:
                raise HTTPException(404, f"No video in directory {p}")
            p = vids[0]
        if not p.is_file():
            raise HTTPException(404, f"File not found: {p}")
        ext = p.suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            with open(p, "rb") as fh:
                ext = sniff_ext(fh.read(64)) or ext
        _check_ext(ext)
        return p, p.name

    async def _read_json_field(val: Any, label: str) -> Any:
        if val is None:
            return None
        text = (await val.read()).decode("utf-8-sig", errors="replace") if hasattr(val, "read") else str(val)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"{label}: invalid JSON ({exc.msg}, line {exc.lineno})") from exc

    # ------------------------------------------------------------------ Health / System / Meta
    @app.get("/internal/health", dependencies=auth_dep)
    def health():
        h = runtime.health_info()
        return {
            "ok": True,
            "version": "ef-gpu-1",
            "demo": demo,
            "models": h["models"],
            "vlm": h["vlm"],
            "busy": h["busy"],
            "active_job_id": h["active_job_id"],
            "det_conf": h.get("det_conf"),
            "det_conf_default": h.get("det_conf_default", 0.01),
            "det_conf_effective": h.get("det_conf_effective", 0.01 if h.get("det_conf") is None else h.get("det_conf")),
            "capabilities": {
                "single_cam": True,
                "multi_cam": True,
                "site_calibrated": True,
                "floorplan": True,
            },
        }

    @app.post("/internal/config", dependencies=auth_dep)
    async def set_config(request: Request):
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, "Body phải là JSON") from exc
        v = body.get("det_conf") if isinstance(body, dict) else None
        if v is not None:
            parsed = _parse_det_conf(v)
            if parsed is None:
                raise HTTPException(400, "det_conf phải nằm trong [0.001, 0.99] hoặc null")
        else:
            parsed = None
        return runtime.set_det_conf(parsed)

    @app.get("/internal/system", dependencies=auth_dep)
    def system():
        return get_system_stats()

    @app.get("/internal/tuning/schema", dependencies=auth_dep)
    def tuning_schema():
        return runtime.tuning_schema()

    @app.get("/internal/sites", dependencies=auth_dep)
    def sites():
        return runtime.list_sites()

    # ------------------------------------------------------------------ Single Jobs
    @app.post("/internal/jobs", dependencies=auth_dep)
    async def create_job(
        job_id: Optional[str] = Form(None),
        video: Optional[UploadFile] = File(None),
        source: str = Form(""),
        options: str = Form("{}"),
    ):
        try:
            opts = json.loads(options or "{}")
            if not isinstance(opts, dict):
                opts = {}
        except json.JSONDecodeError:
            opts = {}

        h = runtime.health_info()
        if h["models"].get("state") == "loading":
            raise HTTPException(409, "Models are still loading")
        if h["busy"]:
            raise HTTPException(409, "GPU worker is busy running another job")

        vpath: Optional[Path] = None
        vname = ""
        src_str = (source or "").strip()
        if video is not None and video.filename:
            vpath, vname = await _save_upload(video)
        elif src_str:
            if demo and src_str == "demo":
                vname = "demo.mp4"
            else:
                vpath, vname = _resolve_source(src_str)
        elif demo:
            vname = "demo.mp4"
        else:
            raise HTTPException(400, "Must provide video upload or source")

        jid = (job_id or "").strip() or uuid.uuid4().hex[:12]
        try:
            return runtime.submit_job(jid, vpath, vname, src_str, opts)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/internal/jobs/{job_id}", dependencies=auth_dep)
    def get_job(job_id: str):
        j = runtime.get_job(job_id)
        if j is None:
            raise HTTPException(404, "Job not found")
        return j

    @app.post("/internal/jobs/{job_id}/cancel", dependencies=auth_dep)
    def cancel_job(job_id: str):
        res = runtime.cancel_job(job_id)
        if res is None:
            raise HTTPException(404, "Job not found")
        return res

    @app.post("/internal/jobs/{job_id}/finish", dependencies=auth_dep)
    def finish_job(job_id: str):
        res = runtime.finish_job(job_id)
        if res is None:
            raise HTTPException(404, "Job not found")
        return res

    @app.get("/internal/jobs/{job_id}/live", dependencies=auth_dep)
    def live_frame(job_id: str):
        if runtime.get_job(job_id) is None:
            raise HTTPException(404, "Job not found")
        jpg = runtime.get_live_frame(job_id)
        if jpg is None:
            raise HTTPException(404, "No live frame available yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/internal/jobs/{job_id}/live/stream", dependencies=auth_dep)
    async def live_stream(job_id: str):
        if runtime.get_job(job_id) is None:
            raise HTTPException(404, "Job not found")

        async def _gen():
            last_seq = -1
            idle = 0
            while True:
                seq, jpg, status = runtime.get_live_frame_with_seq(job_id)
                if jpg is not None and seq != last_seq:
                    last_seq = seq
                    idle = 0
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode("ascii")
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
                else:
                    idle += 1
                    if status in ("done", "error", "cancelled") and idle > 15:
                        return
                await asyncio.sleep(0.03)

        return StreamingResponse(
            _gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/internal/jobs/{job_id}/live/events", dependencies=auth_dep)
    def live_events(job_id: str, after: int = 0):
        res = runtime.get_live_events(job_id, after=after)
        if res is None:
            raise HTTPException(404, "Job not found")
        return res

    @app.get("/internal/jobs/{job_id}/map", dependencies=auth_dep)
    def job_map(job_id: str):
        if runtime.get_job(job_id) is None:
            raise HTTPException(404, "Job not found")
        mp = runtime.get_map_image(job_id)
        if mp is None:
            raise HTTPException(404, "No map available")
        if isinstance(mp, bytes):
            return Response(mp, media_type="image/png", headers={"Cache-Control": "no-store"})
        return FileResponse(mp, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/internal/jobs/{job_id}/map/data", dependencies=auth_dep)
    def job_map_data(job_id: str, full: int = 0):
        res = runtime.get_map_data(job_id, full=full)
        if res is None:
            raise HTTPException(404, "Job not found")
        return res

    @app.get("/internal/jobs/{job_id}/artifacts", dependencies=auth_dep)
    def list_job_artifacts(job_id: str):
        res = runtime.list_job_artifacts(job_id)
        if res is None:
            raise HTTPException(404, "Job not found")
        key, _base, files = res
        return {"job_id": job_id, "key": key, "files": files}

    @app.get("/internal/jobs/{job_id}/artifacts/{rel_path:path}", dependencies=auth_dep)
    def download_job_artifact(job_id: str, rel_path: str):
        try:
            target = runtime.resolve_job_artifact(job_id, rel_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if target is None:
            raise HTTPException(404, "Artifact not found")
        return FileResponse(target)

    # ------------------------------------------------------------------ Calibrated Site Jobs
    @app.post("/internal/site-jobs", dependencies=auth_dep)
    async def create_site_job(request: Request):
        form = await request.form()
        h = runtime.health_info()
        if h["models"].get("state") == "loading":
            raise HTTPException(409, "Models are still loading")
        if h["busy"]:
            raise HTTPException(409, "GPU worker is busy running another job")

        jid = str(form.get("job_id") or "").strip() or uuid.uuid4().hex[:12]
        sid = str(form.get("site") or "")
        custom = sid == "__custom__"
        videos: dict[str, Path] = {}

        if custom:
            keys = [k[4:] for k in form.keys() if k.startswith("cam_") and getattr(form.get(k), "filename", None)]
            if not keys:
                raise HTTPException(400, "At least one cam_<k> video upload is required")
            allcal = await _read_json_field(form.get("calib_all"), "calib_all")
            cams_def: dict[str, Any] = {}
            for pos, k in enumerate(keys):
                up = form.get(f"cam_{k}")
                label = str(form.get(f"label_{k}") or up.filename or f"C{pos + 1}")
                raw = await _read_json_field(form.get(f"calib_{k}"), f"calib_{k}")
                if demo:
                    if raw is None and allcal is None:
                        raise HTTPException(400, f"Missing calibration for {label}")
                    cam_id = f"C{pos + 1}"
                    cams_def[cam_id] = raw if isinstance(raw, dict) else {"label": label}
                    cams_def[cam_id].setdefault("label", label)
                else:
                    cam_id = f"C{pos + 1}"
                    cams_def[cam_id] = runtime.normalize_custom_cam(k, label, up.filename, raw, allcal, pos)
                vpath, _ = await _save_upload(up, uploads_dir / f"{jid}_{cam_id}")
                videos[cam_id] = vpath
            site_def: dict[str, Any] = {
                "site_id": "custom",
                "name": "Custom Calibration",
                "room_id": "custom",
                "sample_fps": 4.0,
                "cameras": cams_def,
                "room_bounds": runtime.extract_room_bounds(allcal) if hasattr(runtime, "extract_room_bounds") else None,
            }
        else:
            try:
                site_def = runtime.load_site(sid)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            for cam in site_def["cameras"]:
                up = form.get(cam)
                if up is None or not getattr(up, "filename", None):
                    continue
                vpath, _ = await _save_upload(up, uploads_dir / f"{jid}_{cam}")
                videos[cam] = vpath
            if not videos:
                raise HTTPException(
                    400,
                    f'At least one camera video required ({", ".join(site_def["cameras"])})',
                )

        ev_flag = str(form.get("events", "1")) != "0"
        if demo:
            events_enabled = ev_flag
        else:
            events_enabled = ev_flag and os.environ.get("EXHIBITFLOW_SITE_EVENTS", "1") != "0"

        try:
            return runtime.submit_site_job(jid, sid, site_def, videos, events_enabled)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/internal/site-jobs/{job_id}", dependencies=auth_dep)
    def get_site_job(job_id: str):
        j = runtime.get_site_job(job_id)
        if j is None:
            raise HTTPException(404, "Site job not found")
        return j

    @app.get("/internal/site-jobs/{job_id}/artifacts", dependencies=auth_dep)
    def list_site_artifacts(job_id: str):
        res = runtime.list_site_artifacts(job_id)
        if res is None:
            raise HTTPException(404, "Site job not found")
        key, _base, files = res
        return {"job_id": job_id, "key": key, "files": files}

    @app.get("/internal/site-jobs/{job_id}/artifacts/{rel_path:path}", dependencies=auth_dep)
    def download_site_artifact(job_id: str, rel_path: str):
        try:
            target = runtime.resolve_site_artifact(job_id, rel_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if target is None:
            raise HTTPException(404, "Artifact not found")
        return FileResponse(target)

    # ------------------------------------------------------------------ Multi-Jobs
    @app.post("/internal/multi-jobs", dependencies=auth_dep)
    async def create_multi_job(request: Request):
        form = await request.form()
        try:
            opts = json.loads(str(form.get("options") or "{}"))
            specs = json.loads(str(form.get("cameras") or "[]"))
        except (json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(400, "Invalid multi-job options or cameras JSON") from exc

        if not isinstance(specs, list) or not (2 <= len(specs) <= 8):
            raise HTTPException(400, "Between 2 and 8 cameras required")

        h = runtime.health_info()
        if h["models"].get("state") == "loading":
            raise HTTPException(409, "Models are still loading")
        if h["busy"]:
            raise HTTPException(409, "GPU worker is busy")

        cams: list[dict[str, Any]] = []
        for i, sp in enumerate(specs):
            sp = sp if isinstance(sp, dict) else {}
            up = form.get(f"video_{i}")
            if up is not None and getattr(up, "filename", None):
                vpath, label = await _save_upload(up)
            else:
                src = str(sp.get("source") or "").strip()
                if not src:
                    raise HTTPException(400, f"Camera {i + 1} missing source or video")
                if src.lower().startswith("push://"):
                    raise HTTPException(400, "Multi-camera does not accept push:// webcam streams")
                vpath, label = _resolve_source(src)
            cams.append({"name": str(sp.get("name") or "").strip()[:40] or label, "path": vpath})

        try:
            return runtime.submit_multi_job(cams, opts)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/internal/multi-jobs/{mid}", dependencies=auth_dep)
    def get_multi_job(mid: str):
        res = runtime.get_multi_job(mid)
        if res is None:
            raise HTTPException(404, "Multi-job not found")
        return res

    @app.get("/internal/multi-jobs/{mid}/results", dependencies=auth_dep)
    def get_multi_results(mid: str):
        res = runtime.get_multi_results(mid)
        if res is None:
            raise HTTPException(404, "Multi-job not found")
        return res

    @app.post("/internal/multi-jobs/{mid}/finish", dependencies=auth_dep)
    def finish_multi_job(mid: str):
        res = runtime.finish_multi_job(mid)
        if res is None:
            raise HTTPException(404, "Multi-job not found")
        return res

    # ------------------------------------------------------------------ Camera Push Session
    @app.post("/internal/camera/session", dependencies=auth_dep)
    def create_camera_session():
        return runtime.create_camera_session()

    @app.websocket("/internal/camera/{sid}/push")
    async def camera_push(ws: WebSocket, sid: str):
        if not await verify_ws_token(ws):
            return
        buf = runtime.get_push_buffer(sid)
        if buf is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        try:
            while True:
                buf.put(await ws.receive_bytes())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            buf.close()
            runtime.pop_push_buffer(sid)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="ExhibitFlow Standalone GPU Worker")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--data-dir", default=str(PKG_DIR / "gpu_data"))
    ap.add_argument("--demo", action="store_true", help="Run pure-CPU demo mode without GPU/weights")
    ap.add_argument("--mode", choices=("tracking", "tracking_seg", "full"), default=None)
    ap.add_argument("--no-preload", action="store_true", help="Do not preload ML models at startup")
    ap.add_argument("--local", action="store_true", help="Low-VRAM staged mode")
    ap.add_argument("--token", default=None, help="Shared X-GPU-Token (defaults to GPU_TOKEN env)")
    args = ap.parse_args()

    import uvicorn

    app = create_worker_app(
        data_dir=args.data_dir,
        demo=args.demo,
        preload=not args.no_preload,
        mode=args.mode,
        local=args.local,
        token=args.token,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
