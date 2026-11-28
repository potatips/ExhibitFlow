"""Real GPU runtime bridging worker/app.py to exhibitflow_gpu.* (lazily imported)."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from exhibitflow_gpu import cam_worker  # noqa: E402
from exhibitflow_gpu import multicam  # noqa: E402
from exhibitflow_gpu import source as efsource  # noqa: E402
from exhibitflow_gpu import still_gate  # noqa: E402
from exhibitflow_gpu import tuning as _tuning  # noqa: E402
from exhibitflow_gpu.floorplan import (  # noqa: E402
    CalibrationError,
    FloorplanService,
    GlobalIdLabeler,
    art_match,
    build_incidents,
    engine_events,
    engine_summary,
    fit_to_video,
    floor_mapper,
    homography_mapper,
    normalize_calibration,
    room_bounds_from,
    site_fusion,
    site_live,
    site_parallel,
    split_calibration,
)
from exhibitflow_gpu.runner import PipelineRunner, State  # noqa: E402
from exhibitflow_gpu.vlm_server import SharedVLMServer  # noqa: E402

from .demo import SITE_STAGES, SITE_STAGES_EV, STAGES, _guess_content_type, _safe_rel_path

NW = 960
PIPE_MODES = ("tracking", "tracking_seg", "full")
PIPE_OPTS = {
    "pipeline_mode": lambda v: v if v in PIPE_MODES else "full",
    "tracker_process": bool,
    "hand_process": bool,
    "pose_process": bool,
    "pose_gpu_preproc": bool,
    "hand_batch": bool,
    "drop_frames": bool,
    "limit_cpu_threads": bool,
    "alarm_show_sec": float,
    "tracker_threads": int,
    "hand_threads": int,
    "pose_threads": int,
    "queue_size": int,
    "pose_gate_norm_dist": float,
    "multi_parallel": bool,
    "multi_max_parallel": int,
    "tuning": _tuning.validate_tuning,
}
MAKE_MAP = os.environ.get("EXHIBITFLOW_MAKE_MAP", "1") != "0"
SITE_MAX_SECONDS = float(os.environ.get("EXHIBITFLOW_SITE_MAX_SECONDS", "30"))


def _pipe_of(opts: dict[str, Any]) -> dict[str, Any]:
    return {k: t(opts[k]) for k, t in PIPE_OPTS.items() if opts.get(k) is not None}


def _apply_det_conf(runner: PipelineRunner, conf: Optional[float]) -> bool:
    if conf is None:
        return False
    det = getattr(getattr(getattr(runner, "engine", None), "M", None), "detector", None)
    if det is None:
        return False
    if hasattr(det, "conf"):
        det.conf = float(conf)
    elif isinstance(getattr(det, "kw", None), dict):
        det.kw["conf"] = float(conf)
    else:
        return False
    return True


def _video_info(path: Optional[Path]) -> dict[str, Any]:
    if path is None or efsource.is_stream(path):
        return {"fps": 25.0, "n_frames": 0, "width": 0, "height": 0}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        info = {
            "fps": fps if fps > 1 else 25.0,
            "n_frames": max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)),
            "width": max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)),
            "height": max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)),
        }
        cap.release()
        return info
    except Exception:  # noqa: BLE001
        return {"fps": 25.0, "n_frames": 0, "width": 0, "height": 0}


def _cam_match(cid: str, k: str, label: str, filename: str) -> bool:
    if not cid:
        return False
    c = cid.lower()
    return c in (k.lower(), label.lower()) or (len(c) >= 3 and c in Path(filename).stem.lower())


def _site_device() -> str:
    dev = os.environ.get("EXHIBITFLOW_SITE_DEVICE")
    if dev:
        return dev
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class RealRuntime:
    """Real GPU worker runtime using exhibitflow_gpu.*."""

    def __init__(
        self,
        data_dir: Path,
        *,
        preload: bool = True,
        mode: Optional[str] = None,
        local: bool = False,
        det_conf: Optional[float] = None,
    ) -> None:
        self.root = Path(data_dir)
        self.state = State(self.root)
        self.runner = PipelineRunner(self.state, local=local)
        self.local = local
        self.det_conf = det_conf
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cur: dict[str, Optional[str]] = {"id": None}
        self._site_jobs: dict[str, dict[str, Any]] = {}
        self._multis: dict[str, Any] = {}
        self._app_site_ids: dict[str, Any] = {"labeler": None, "map": {}, "votes": {}}

        if mode:
            self.state.save_settings({"pipeline_mode": mode})
        self.state.save_settings(
            {
                "tracker_process": True,
                "hand_process": True,
                "pose_process": True,
                "hand_batch": True,
                "drop_frames": True,
            }
        )
        if os.environ.get("EXHIBITFLOW_SAVE_VIDEO", "1") != "0":
            import exhibitflow_gpu.engine as eng_mod
            import exhibitflow_gpu.service as svc

            orig_run_video = svc.run_video

            def _run_video_with_live(*a, **k):
                svc.RT["render_live"] = True
                if _apply_det_conf(self.runner, self.det_conf):
                    self.state.log(f"Detector conf = {self.det_conf}")
                return orig_run_video(*a, **k)

            svc.run_video = _run_video_with_live

            orig_render_live = eng_mod.render_live

            def _render_live_global_ids(ctx, p, *a, **k):
                lab = self._app_site_ids.get("labeler")
                self._app_site_ids["map"] = {}
                if lab is not None:
                    try:
                        h, w = p.work.shape[:2]
                        self._app_site_ids["map"] = lab.labels(
                            p.idx,
                            [{"id": r["id"], "box": r["box"]} for r in (p.tracks or [])],
                            (w, h),
                            image=p.work,
                        )
                        for tid, gid in self._app_site_ids["map"].items():
                            v = self._app_site_ids.setdefault("votes", {}).setdefault(int(tid), {})
                            v[gid] = v.get(gid, 0) + 1
                    except Exception:  # noqa: BLE001
                        self._app_site_ids["map"] = {}
                return orig_render_live(ctx, p, *a, **k)

            eng_mod.render_live = _render_live_global_ids

            orig_put_label = eng_mod.put_label

            def _put_label_no_hud(img, text, *a, **k):
                if isinstance(text, str) and text.startswith(("frame ", "pipeline ", "VLM queue")):
                    return None
                gm = self._app_site_ids.get("map")
                if gm and isinstance(text, str):
                    m = re.match(r"ID (\d+)(.*)$", text)
                    if m and int(m.group(1)) in gm:
                        text = f"ID {gm[int(m.group(1))]}" + m.group(2)
                return orig_put_label(img, text, *a, **k)

            eng_mod.put_label = _put_label_no_hud

        if preload:
            self.runner.preload()

    def det_conf_state(self) -> dict[str, Any]:
        from exhibitflow_gpu import settings as efs

        default = float(efs.MODELS.get("detector", {}).get("conf", 0.01))
        cur = self.det_conf
        return {
            "det_conf": cur,
            "det_conf_default": default,
            "det_conf_effective": default if cur is None else cur,
        }

    def set_det_conf(self, conf: Optional[float]) -> dict[str, Any]:
        self.det_conf = conf
        applied = _apply_det_conf(self.runner, conf)
        return {**self.det_conf_state(), "applied_now": applied}

    def health_info(self) -> dict[str, Any]:
        with self.state.lock:
            msg = self.state.models_msg or (
                self.state.models_stage() if self.state.models_state == "loading" else ""
            )
            ms = {"state": self.state.models_state, "msg": msg}
        vlm_st = SharedVLMServer.instance().status()
        vlm_info = {
            "state": vlm_st.get("state", "stopped"),
            "url": vlm_st.get("url"),
            "model": vlm_st.get("model"),
            "shared": bool(vlm_st.get("shared", True)),
        }
        site_running = next((jid for jid, sj in self._site_jobs.items() if sj["status"] == "running"), None)
        busy = self.runner.running() or (site_running is not None)
        active_id = site_running or (self._cur["id"] if self.runner.running() else None)
        return {
            "models": ms,
            "vlm": vlm_info,
            "busy": busy,
            "active_job_id": active_id,
            **self.det_conf_state(),
        }

    def tuning_schema(self) -> dict[str, Any]:
        return _tuning.schema()

    def list_sites(self) -> list[dict[str, Any]]:
        return list(FloorplanService.list_sites().values())

    def load_site(self, site_id: str) -> dict[str, Any]:
        return FloorplanService.load_site(site_id)

    def resolve_stream_source(self, src: str, dst_dir: Path) -> tuple[Path, str]:
        import re

        if re.match(r"^(rtsps?|rtmps?)://", src, re.I):
            return (
                efsource.write_stream_file(dst_dir / f"cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}", url=src),
                "Camera " + efsource._mask(src),
            )
        m = re.match(r"^push://([\w-]+)(?:\?fps=(\d+))?$", src)
        if m:
            if m.group(1) not in efsource.PUSH:
                raise ValueError("Camera push session does not exist or has closed.")
            return (
                efsource.write_stream_file(
                    dst_dir / f"cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}",
                    push=m.group(1),
                    fps=int(m.group(2) or 15),
                ),
                "Browser Camera",
            )
        raise ValueError(f"Unsupported stream source: {src}")

    def _vinfo_of(self, j: dict[str, Any]) -> dict[str, Any]:
        if efsource.is_stream(j["path"]) and (m := efsource.INFO.get(str(j["path"]))):
            j["vinfo"] = {"fps": m["fps"], "n_frames": 0, "width": m["width"], "height": m["height"]}
        return j["vinfo"]

    def _result_dir(self, j: dict[str, Any]) -> Optional[Path]:
        if j.get("rdir") is not None:
            return j["rdir"]
        root_dir = (j.get("root") or self.state.root) / "results"
        if not root_dir.is_dir():
            return None
        found = [p for p in root_dir.rglob("result.json") if p.stat().st_mtime >= j["t0"] - 2]
        if not found:
            return None
        j["rdir"] = max(found, key=lambda p: p.stat().st_mtime).parent
        return j["rdir"]

    def _view(self, j: dict[str, Any]) -> dict[str, Any]:
        if j.get("remote") is not None:
            if j.get("final") is not None:
                return j["final"]
            try:
                v = cam_worker.view_of(j["remote"].snap(), cancelled=j["cancelled"])
            except Exception as exc:  # noqa: BLE001
                v = {"status": "error", "stage": 0, "progress": 0.0, "error": f"Camera process error: {exc}"[:300]}
            if v["status"] in ("done", "error", "cancelled"):
                j["final"] = v
            return v

        if j.get("final") is not None:
            return j["final"]
        if self._cur["id"] != j["id"]:
            j["final"] = {"status": "cancelled", "stage": 0, "progress": 0.0, "error": None}
            return j["final"]
        with self.state.lock:
            st, phase, msg, fi, nf = (
                self.state.status,
                self.state.phase,
                self.state.message,
                self.state.frame_idx,
                self.state.n_frames,
            )
        alive = self.runner.running()
        if j["cancelled"] and not alive:
            j["final"] = {"status": "cancelled", "stage": 4, "progress": 0.0, "error": None}
        elif st == "error":
            j["final"] = {"status": "error", "stage": 0, "progress": 0.0, "error": msg or "Pipeline error"}
        elif st == "done" and not alive:
            j["final"] = {"status": "done", "stage": 4, "progress": 1.0, "error": None}
        elif st in ("idle", "loading"):
            stage = 1 if "map" in (phase or "").lower() else 0
            return {"status": "running", "stage": stage, "progress": 0.0, "error": None}
        elif st == "running":
            return {"status": "running", "stage": 3, "progress": round(fi / nf, 3) if nf else 0.0, "error": None}
        else:
            return {"status": "running", "stage": 4, "progress": 0.5, "error": None}
        return j["final"]

    def submit_job(
        self,
        job_id: str,
        video_path: Optional[Path],
        video_name: str,
        source: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        if video_path is None:
            raise ValueError("Video path is required in GPU mode")
        try:
            pipe = _pipe_of(options)
        except (ValueError, TypeError) as exc:
            raise ValueError(str(exc)) from exc

        realtime = bool(options.get("realtime"))
        if efsource.is_stream(video_path):
            from exhibitflow_gpu import settings as efs

            efs.RT["stream_max_sec"] = max(10.0, float(options.get("camera_max_min") or 30) * 60)
        self.state.save_settings(dict(pipe, realtime=realtime, make_map=MAKE_MAP, map_image="auto"))
        self.runner.start(str(video_path))
        self._jobs[job_id] = {
            "id": job_id,
            "path": Path(video_path),
            "name": video_name or Path(video_path).name,
            "t0": time.time(),
            "cancelled": False,
            "final": None,
            "rdir": None,
            "realtime": realtime,
            "vinfo": _video_info(Path(video_path)),
        }
        self._cur["id"] = job_id
        return {"job_id": job_id, "status": "running"}

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        v = self._view(j)
        key = None
        if v["status"] == "done":
            rd = self._result_dir(j)
            key = rd.name if rd else (self.state.cam_name or j["path"].stem)
        with self.state.lock:
            logs = list(self.state.logs)[-40:] if self._cur["id"] == job_id else []
        return {
            "job_id": job_id,
            "status": v["status"],
            "stage": v["stage"],
            "stages": list(STAGES),
            "progress": v["progress"],
            "key": key,
            "error": v["error"],
            "name": j["name"],
            "realtime": j["realtime"],
            "log": logs,
        }

    def _stop_job(self, j: dict[str, Any]) -> None:
        if j.get("remote") is not None:
            j["remote"].stop()
        else:
            self.runner.stop()

    def cancel_job(self, job_id: str) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        v = self._view(j)
        if v["status"] in ("done", "error", "cancelled"):
            return {"status": v["status"]}
        j["cancelled"] = True
        self._stop_job(j)
        return {"status": "cancelling"}

    def finish_job(self, job_id: str) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        v = self._view(j)
        if v["status"] in ("done", "error", "cancelled"):
            return {"status": v["status"]}
        self._stop_job(j)
        return {"status": "finishing"}

    def get_live_frame(self, job_id: str) -> Optional[bytes]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        if j.get("remote") is not None:
            r = j["remote"].frame(-1)
            return r["jpg"] if r else None
        with self.state.lock:
            return self.state.frame_jpeg

    def get_live_frame_with_seq(self, job_id: str) -> tuple[int, Optional[bytes], str]:
        j = self._jobs.get(job_id)
        if j is None:
            return -1, None, "error"
        if j.get("remote") is not None:
            r = j["remote"].frame(-1)
            if r is None:
                return -1, None, "error"
            return r["seq"], r["jpg"], r["st"]
        with self.state.lock:
            return self.state.frame_seq, self.state.frame_jpeg, self.state.status

    def _art_boxes(self, j: dict[str, Any]) -> list[dict[str, Any]]:
        vi = self._vinfo_of(j)
        ratio = (vi["height"] / vi["width"]) if vi["width"] else 0.5625
        with self.state.lock:
            scenes = list(self.state.scene_contours)
        out = []
        for a in scenes[0]["arts"] if scenes else []:
            pts = [pt for poly in a["c"] for pt in poly]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(
                {
                    "id": a["id"],
                    "label": f"Artifact {a['id']}",
                    "pedestal": False,
                    "box": [
                        round(min(xs) * NW, 1),
                        round(min(ys) * NW * ratio, 1),
                        round(max(xs) * NW, 1),
                        round(max(ys) * NW * ratio, 1),
                    ],
                }
            )
        return out

    def get_live_events(self, job_id: str, after: int = 0) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        if j.get("remote") is not None:
            return {"meta": None, "status": None, "events": [], "job": self._view(j)["status"]}
        vi = self._vinfo_of(j)
        with self.state.lock:
            fi, nf, fps_now, alerts = (
                self.state.frame_idx,
                self.state.n_frames,
                self.state.fps,
                list(self.state.alerts),
            )
            running = self.state.status in ("running", "finishing", "done")
        nf = nf or vi["n_frames"]
        vfps = vi["fps"] or 25.0
        meta = (
            {"artifacts": self._art_boxes(j), "n_frames": nf, "fps": vfps, "width": vi["width"], "height": vi["height"]}
            if (running and nf)
            else None
        )
        status = (
            {
                "frame": fi,
                "n_frames": nf,
                "t_s": round(fi / vfps, 2),
                "live_fps": round(fps_now, 1),
                "dropped": 0,
                "vlm_pending": None,
            }
            if running
            else None
        )
        events = []
        for a in alerts:
            if a["id"] <= after:
                continue
            vt = float(a.get("video_time") or 0)
            art_ids = a.get("artifact_ids") or []
            events.append(
                {
                    "seq": a["id"],
                    "kind": "vlm",
                    "track": a["track_id"],
                    "artifact": art_ids[0] if art_ids else None,
                    "artifacts": art_ids,
                    "decision": "ANOMALY",
                    "behavior": a["behavior"],
                    "confidence": a["confidence"],
                    "evidence": a.get("evidence") or "",
                    "label": a["behavior"],
                    "start_s": max(0.0, vt - 3.0),
                    "end_s": vt + 1.0,
                }
            )
        return {"meta": meta, "status": status, "events": events, "job": self._view(j)["status"]}

    def get_map_image(self, job_id: str) -> Optional[Path]:
        if job_id not in self._jobs:
            return None
        with self.state.lock:
            p = self.state.map_path
        return Path(p) if p is not None and Path(p).is_file() else None

    def get_map_data(self, job_id: str, full: int = 0) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        vfps = j["vinfo"]["fps"] or 25.0
        v = self._view(j)
        with self.state.lock:
            has = self.state.map_path is not None and Path(self.state.map_path).is_file()
            out: dict[str, Any] = {
                "has_map": has,
                "version": self.state.map_version,
                "fps": vfps,
                "people": [
                    {
                        "id": p["id"],
                        "x": round(p["x"], 4),
                        "y": round(p["y"], 4),
                        "state": p.get("state", "idle"),
                    }
                    for p in self.state.people
                ],
                "artifacts": [
                    {"id": a["id"], "x": round(a["x"], 4), "y": round(a["y"], 4)} for a in self.state.artifacts
                ],
            }
            if full and v["status"] == "done":
                out["frames"] = {int(k): [[q[0], q[1], q[2]] for q in val[1]] for k, val in self.state.timeline.items() if val[1]}
        return out

    def _gated_result_path(self, j: dict[str, Any], rd: Path) -> Path:
        raw_path = rd / "result.json"
        gated_path = rd / "result_gated.json"
        if not raw_path.is_file():
            return raw_path
        if gated_path.is_file() and gated_path.stat().st_mtime >= raw_path.stat().st_mtime:
            return gated_path
        try:
            res = json.loads(raw_path.read_text(encoding="utf-8"))
            runs = sorted(
                ((j.get("root") or self.state.root) / "runs").glob(f"*_{rd.name}/timeline.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if runs:
                res = still_gate.apply(res, json.loads(runs[-1].read_text(encoding="utf-8")))[0]
            gated_path.write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
            return gated_path
        except Exception as exc:  # noqa: BLE001
            self.state.log(f"Không áp được quy tắc đứng yên: {exc}")
            return raw_path

    def _artifact_map_for_job(self, job_id: str) -> Optional[tuple[str, Path, dict[str, Path]]]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        rd = self._result_dir(j)
        if rd is None:
            return None
        key = rd.name
        mapping: dict[str, Path] = {}
        for name in ("result.json", "live.mp4", "diagnostics.csv"):
            p = self._gated_result_path(j, rd) if name == "result.json" else (rd / name)
            if p.is_file():
                mapping[name] = p

        # Map image & map_data.json
        with self.state.lock:
            mp = Path(self.state.map_path) if self.state.map_path else None
        if mp is not None and mp.is_file():
            mapping["map.png"] = mp
        md_path = rd / "map_data.json"
        if not md_path.is_file():
            md = self.get_map_data(job_id, full=1)
            if md is not None:
                md_path.write_text(json.dumps(md), encoding="utf-8")
        if md_path.is_file():
            mapping["map_data.json"] = md_path

        vlm_dir = rd / "vlm_inputs"
        if vlm_dir.is_dir():
            for fp in sorted(vlm_dir.rglob("*")):
                if fp.is_file():
                    rel = f"files/{key}/vlm_inputs/{fp.relative_to(vlm_dir).as_posix()}"
                    mapping[rel] = fp

        return key, rd, mapping

    def list_job_artifacts(self, job_id: str) -> Optional[tuple[str, Path, list[dict[str, Any]]]]:
        res = self._artifact_map_for_job(job_id)
        if res is None:
            return None
        key, rd, mapping = res
        files = [
            {"rel_path": rel, "content_type": _guess_content_type(path), "size": path.stat().st_size}
            for rel, path in sorted(mapping.items())
        ]
        return key, rd, files

    def resolve_job_artifact(self, job_id: str, rel_path: str) -> Optional[Path]:
        if not rel_path or "\x00" in rel_path or rel_path.startswith(("/", "\\")) or "\\" in rel_path:
            raise ValueError("Invalid path")
        if any(p in ("", ".", "..") for p in rel_path.split("/")):
            raise ValueError("Path traversal rejected")
        res = self._artifact_map_for_job(job_id)
        if res is None:
            return None
        _key, _rd, mapping = res
        target = mapping.get(rel_path)
        return target if target is not None and target.is_file() else None

    # ---- Calibrated Site Jobs ----
    def normalize_custom_cam(
        self,
        k: str,
        label: str,
        filename: str,
        raw: Any,
        allcal: Any,
        pos: int,
    ) -> dict[str, Any]:
        from fastapi import HTTPException

        pool = split_calibration(allcal) if allcal is not None else []
        if raw is not None:
            items = split_calibration(raw)
            raw = next((c for cid, c in items if _cam_match(cid, k, label, filename)), items[0][1])
        elif pool:
            raw = next((c for cid, c in pool if _cam_match(cid, k, label, filename)), pool[pos][1] if pos < len(pool) else None)
        if raw is None:
            raise HTTPException(400, f"Missing calibration for {label}")
        try:
            cam = normalize_calibration(raw, label)
        except CalibrationError as exc:
            raise HTTPException(400, str(exc)) from exc
        cam["label"] = label
        return cam

    @staticmethod
    def extract_room_bounds(raw: Any) -> Any:
        return room_bounds_from(raw)

    def _analyze_video(self, path: Path) -> dict[str, Any]:
        end = time.time() + 900
        while self.runner.loading() and time.time() < end:
            time.sleep(1.0)
        if self.runner.running():
            raise RuntimeError("Engine is busy with another video")
        info = _video_info(path)
        t_start = time.time()
        self.state.save_settings(
            {
                "realtime": False,
                "make_map": False,
                "map_image": "auto",
                "max_frames": int(SITE_MAX_SECONDS * info["fps"]),
            }
        )
        self.runner.start(str(path))
        while self.runner.running():
            time.sleep(0.5)
        with self.state.lock:
            st, msg, rid = self.state.status, self.state.message, getattr(self.state, "run_id", None)
        tl = (self.state.root / "runs" / str(rid) / "timeline.json") if rid else None
        if st != "done" or tl is None or not tl.is_file() or tl.stat().st_mtime < t_start - 1:
            raise RuntimeError(msg or "Engine returned no timeline")
        out = json.loads(tl.read_text(encoding="utf-8"))
        try:
            out["_run"] = json.loads((tl.parent / "run.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        res_dir = self.state.root / "results"
        cands = [res_dir / path.stem / "live.mp4"] + sorted(
            res_dir.rglob("live.mp4"), key=lambda q: q.stat().st_mtime, reverse=True
        )
        for lv in cands:
            if lv.is_file() and lv.stat().st_mtime >= t_start - 1:
                out["_live"], out["_result_dir"] = str(lv), str(lv.parent)
                break
        return out

    @staticmethod
    def _build_site_cam_result(job: dict[str, Any], cam: str, raw_res_path: Path) -> dict[str, Any]:
        res = copy.deepcopy((job.get("fused") or {}).get(cam)) or json.loads(raw_res_path.read_text(encoding="utf-8"))
        res = art_match.renumber_result(res, (job.get("art_map") or {}).get(cam))
        m = (job.get("id_map") or {}).get(cam) or {}
        if m:
            g = lambda t: m.get(int(t), t) if t is not None else t  # noqa: E731
            for k in ("alerts", "events", "vlm_tasks"):
                for e in res.get(k) or []:
                    if e.get("track_id") is not None:
                        e["track_id"] = g(e["track_id"])
            res["track_summary"] = {str(g(t)): v for t, v in (res.get("track_summary") or {}).items()}
            dg = res.get("diagnostics") or {}
            if isinstance(dg.get("tracks"), dict):
                dg["tracks"] = {
                    str(g(t)): dict(v, track_id=g(t)) if isinstance(v, dict) else v
                    for t, v in dg["tracks"].items()
                }
        return res

    def _site_events_stage(self, job: dict[str, Any], site: dict[str, Any]) -> None:
        rpath = job["out"] / "site_result.json"
        result = json.loads(rpath.read_text(encoding="utf-8"))
        errors, seen, cam_events = [], [], []
        tls: dict[str, dict[str, Any]] = {}
        id_maps: dict[str, dict[int, int]] = job.setdefault("id_map", {})
        job["stage"] = 2

        def cam_ctx(cam: str, path: Path):
            vi = _video_info(path)
            spec = fit_to_video(site["cameras"][cam], int(vi["width"]), int(vi["height"]))
            H = next((c.get("H") for c in result.get("cameras", []) if c.get("id") == cam), None)
            return vi, spec, H

        par_n = int(os.environ.get("EXHIBITFLOW_SITE_PARALLEL", "0") or 0) or len(job["videos"])
        if len(job["videos"]) > 1 and par_n > 1:
            try:
                args = {}
                for cam, path in job["videos"].items():
                    vi, spec, H = cam_ctx(cam, path)
                    args[cam] = dict(
                        spec=spec,
                        H=H,
                        frames=result.get("frames", []),
                        fps=vi["fps"],
                        wh=(vi["width"], vi["height"]),
                        det_conf=self.det_conf,
                    )
                vlm_url: Optional[str] = None
                if self.runner.engine is not None:
                    vlm_url = (self.runner.engine.models or {}).get("vlm", {}).get("server_url")
                    self.runner.engine.close()
                    self.runner.engine = None
                    with self.state.lock:
                        self.state.models_state = "idle"
                if not vlm_url:
                    st_vlm = SharedVLMServer.instance().status()
                    if st_vlm.get("state") == "ready" and st_vlm.get("url"):
                        vlm_url = st_vlm["url"]
                maxf = int(SITE_MAX_SECONDS * max(_video_info(p)["fps"] for p in job["videos"].values()))
                job["log"].append(f"[events] chạy song song {len(job['videos'])} camera...")
                got = site_parallel.run(
                    job["videos"],
                    self.state.root / "sites" / job["id"] / "cams",
                    dict(self.state.settings, realtime=False, make_map=False, map_image="auto", max_frames=maxf),
                    args,
                    job["log"].append,
                    local=self.local,
                    max_parallel=par_n,
                    on_progress=lambda f: job.__setitem__("par_progress", f),
                    vlm_url=vlm_url,
                )
                for cam, (tl, idm) in got.items():
                    tls[cam], id_maps[cam] = tl, idm
            except Exception as exc:  # noqa: BLE001
                job["log"].append(
                    f"[events] chạy song song lỗi ({type(exc).__name__}: {str(exc)[:200]}): chuyển sang chạy tuần tự"
                )
                job.pop("par_progress", None)
                tls.clear()
                self.runner.preload()

        if not tls:
            gallery: dict[int, Any] = {}
            try:
                for k, (cam, path) in enumerate(job["videos"].items()):
                    job["seq"] = (k, len(job["videos"]))
                    job["log"].append(f"[events] {site['cameras'][cam].get('label', cam)}: analyzing...")
                    try:
                        self._app_site_ids["labeler"], self._app_site_ids["votes"] = None, {}
                        try:
                            vi, spec0, H0 = cam_ctx(cam, path)
                            mp0 = (
                                homography_mapper(H0, spec0["K"], spec0.get("dist"))
                                if H0
                                else floor_mapper(spec0)
                            )
                            self._app_site_ids["labeler"] = GlobalIdLabeler(
                                result.get("frames", []),
                                mp0,
                                vi["fps"],
                                (vi["width"], vi["height"]),
                                gallery=gallery,
                            )
                        except Exception as exc:  # noqa: BLE001
                            job["log"].append(f"[events] {cam}: không đổi được sang ID chung ({exc})")
                        tls[cam] = self._analyze_video(path)
                        id_maps[cam] = {
                            t: max(v, key=v.get) for t, v in self._app_site_ids.get("votes", {}).items() if v
                        }
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{cam}: {type(exc).__name__}: {exc}")
                        job["log"].append(f"[events] {cam} error: {exc}")
            finally:
                self._app_site_ids["labeler"] = None
                self.state.save_settings({"max_frames": 0})

        for cam, tl in tls.items():
            if tl.get("_live"):
                job.setdefault("live", {})[cam] = tl["_live"]
            if tl.get("_result_dir"):
                job.setdefault("result_dirs", {})[cam] = tl["_result_dir"]

        gated: dict[str, dict[str, Any]] = {}
        for cam, tl in list(tls.items()):
            rd = (job.get("result_dirs") or {}).get(cam)
            if rd and (Path(rd) / "result.json").is_file():
                try:
                    gated[cam], tls[cam] = still_gate.apply(
                        json.loads((Path(rd) / "result.json").read_text(encoding="utf-8")), tl
                    )
                    job.setdefault("fused", {})[cam] = gated[cam]
                except Exception as exc:  # noqa: BLE001
                    job["log"].append(f"[events] {cam}: không áp được quy tắc đứng yên: {exc}")

        fx: dict[str, dict[str, Any]] = {}
        try:
            for cam, tl in tls.items():
                rd = (job.get("result_dirs") or {}).get(cam)
                if rd and (Path(rd) / "result.json").is_file():
                    fx[cam] = dict(
                        tl=tls[cam],
                        res=gated.get(cam) or json.loads((Path(rd) / "result.json").read_text(encoding="utf-8")),
                        id_map=id_maps.get(cam) or {},
                    )
            if len(fx) > 1:
                fused, info = site_fusion.fuse(fx)
                result["fusion"] = info
                for cam, v in fused.items():
                    job.setdefault("fused", {})[cam] = v["res"]
                    tls[cam] = site_fusion.add_promoted_alerts(
                        site_fusion.drop_demoted_alerts(tls[cam], v["drop"]), v.get("add")
                    )
                    lv = (job.get("live") or {}).get(cam)
                    if v.get("add") and lv:
                        try:
                            if site_live.paint_promoted(lv, tls[cam], v["add"], id_maps.get(cam) or {}):
                                job["log"].append(
                                    f"[events] {cam}: đã vẽ khung đỏ cho {len(v['add'])} sự kiện lan truyền vào live.mp4"
                                )
                        except Exception as exc:  # noqa: BLE001
                            job["log"].append(
                                f"[events] {cam}: không vẽ được khung đỏ vào video ({type(exc).__name__}: {exc})"
                            )
                n_dem = sum(1 for e in info["events"] if e["status"] == "conflict")
                job["log"].append(
                    f"[events] đối chiếu chéo camera ({info['policy']}): người bất thường {info.get('anomaly_persons') or []}, "
                    f"{len(info.get('propagated') or [])} sự kiện lan truyền, {n_dem} hạ xuống 'cần xem lại'"
                )
        except Exception as exc:  # noqa: BLE001
            job["log"].append(f"[events] đối chiếu chéo lỗi: {type(exc).__name__}: {exc}")

        try:
            results, specs = {}, {}
            for cam, tl in tls.items():
                rd = (job.get("result_dirs") or {}).get(cam)
                if rd and (Path(rd) / "result.json").is_file():
                    results[cam] = json.loads((Path(rd) / "result.json").read_text(encoding="utf-8"))
                    specs[cam] = fit_to_video(site["cameras"][cam], int(tl["width"]), int(tl["height"]))
            job["art_map"] = art_match.match(results, specs, site.get("room_bounds"))
            hs = {c["id"]: c.get("H") for c in result.get("cameras", [])}
            result["exhibits"] = art_match.map_exhibits(results, specs, job["art_map"], site.get("room_bounds"), hs)
            for cam, ids in job["art_map"].items():
                tls[cam] = art_match.renumber_timeline(tls[cam], ids)
        except Exception as exc:  # noqa: BLE001
            job["art_map"] = {}
            job["log"].append(f"[events] không ghép được số hiện vật giữa các camera: {type(exc).__name__}: {exc}")

        for cam, tl in tls.items():
            try:
                vi, spec, H = cam_ctx(cam, job["videos"][cam])
                spec = fit_to_video(site["cameras"][cam], int(tl["width"]), int(tl["height"]))
                mapper = homography_mapper(H, spec["K"], spec.get("dist")) if H else floor_mapper(spec)
                cam_events += engine_events(tl, mapper, cam)
                seen.append({"cam": cam, **engine_summary(tl, tl.get("_run"))})
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{cam}: {type(exc).__name__}: {exc}")
                job["log"].append(f"[events] {cam} error: {exc}")

        # Đồng bộ live.mp4 + result.json từng camera + vlm_inputs vào thư mục out của site job để Backend tải về
        vids_dir = job["out"] / "videos"
        vids_dir.mkdir(parents=True, exist_ok=True)
        for cam, lv in (job.get("live") or {}).items():
            if lv and Path(lv).is_file():
                try:
                    shutil.copyfile(lv, vids_dir / f"{cam}.mp4")
                except Exception:  # noqa: BLE001
                    pass
        for cam, rd in (job.get("result_dirs") or {}).items():
            rp = Path(rd) / "result.json"
            if not rp.is_file():
                continue
            try:
                cam_out = job["out"] / "cams" / cam
                cam_out.mkdir(parents=True, exist_ok=True)
                cam_res = self._build_site_cam_result(job, cam, rp)
                (cam_out / "result.json").write_text(json.dumps(cam_res, separators=(",", ":")), encoding="utf-8")
                vlm_in = Path(rd) / "vlm_inputs"
                if vlm_in.is_dir():
                    dst_vlm = cam_out / "vlm_inputs"
                    if dst_vlm.exists():
                        shutil.rmtree(dst_vlm)
                    shutil.copytree(vlm_in, dst_vlm)
            except Exception as exc:  # noqa: BLE001
                job["log"].append(f"[events] {cam}: không lưu được result.json cho backend ({exc})")

        job["stage"] = 3
        result["events"] = build_incidents(cam_events, result.get("frames", []))
        result["engine"] = seen
        result["events_error"] = "; ".join(errors) or None
        rpath.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    def _site_worker(self, job: dict[str, Any]) -> None:
        try:
            site = job.get("site_def") or FloorplanService.load_site(job["site"])
            repo = FloorplanService.find_repo()
            if repo is None:
                raise RuntimeError("multicam-2d-floorplan-tracker repository not found (set FLOORPLAN_DIR)")
            job["stage"] = 0
            work = self.root / "sites" / job["id"]
            data_dir, n = FloorplanService.prepare_dataset(
                site, job["videos"], work / "data", seconds=SITE_MAX_SECONDS, log=job["log"].append
            )
            job["stage"] = 1
            out = FloorplanService.run_repo(
                repo, data_dir, work / "out", n, site["room_id"], device=_site_device(), log=job["log"].append
            )
            if not (out / "site_result.json").is_file():
                raise RuntimeError("Floorplan pipeline did not produce site_result.json")
            vids_dir = out / "videos"
            vids_dir.mkdir(parents=True, exist_ok=True)
            for cam, src_p in job["videos"].items():
                dst = vids_dir / f"{cam}.mp4"
                if not dst.exists() and Path(src_p).is_file():
                    shutil.copyfile(src_p, dst)
            job["out"] = out
            if job["events"]:
                self._site_events_stage(job, site)
            job["stage"] = len(job["stages"]) - 1
            job["status"] = "done"
        except BaseException as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"

    def submit_site_job(
        self,
        job_id: str,
        site_id: str,
        site_def: dict[str, Any],
        videos: dict[str, Path],
        events: bool,
    ) -> dict[str, Any]:
        job = {
            "id": job_id,
            "site": site_id,
            "site_def": site_def if site_id == "__custom__" else None,
            "videos": videos,
            "status": "running",
            "stage": 0,
            "error": None,
            "log": [],
            "out": None,
            "t0": time.time(),
            "events": events,
            "stages": list(SITE_STAGES_EV if events else SITE_STAGES),
        }
        self._site_jobs[job_id] = job
        threading.Thread(target=self._site_worker, args=(job,), daemon=True, name=f"site-{job_id[:8]}").start()
        return {"job_id": job_id, "cameras": list(videos.keys()), "events": events}

    def _site_progress(self, j: dict[str, Any]) -> float:
        if j["stage"] != 2:
            return 1.0 if j["status"] == "done" else 0.0
        if "par_progress" in j:
            return round(float(j["par_progress"]), 3)
        k, n = j.get("seq") or (0, max(1, len(j.get("videos") or {})))
        part = 0.0
        if self.runner.running():
            with self.state.lock:
                if self.state.status == "running" and self.state.n_frames:
                    part = min(1.0, max(0.0, self.state.frame_idx / self.state.n_frames))
        return round((k + part) / max(1, n), 3)

    def get_site_job(self, job_id: str) -> Optional[dict[str, Any]]:
        j = self._site_jobs.get(job_id)
        if j is None:
            return None
        return {
            "job_id": j["id"],
            "status": j["status"],
            "stage": j["stage"],
            "stages": j["stages"],
            "error": j["error"],
            "cameras": list(j["videos"].keys()),
            "log": list(j["log"][-8:]),
            "elapsed": round(time.time() - j["t0"]),
            "progress": self._site_progress(j),
        }

    def list_site_artifacts(self, job_id: str) -> Optional[tuple[str, Path, list[dict[str, Any]]]]:
        j = self._site_jobs.get(job_id)
        if j is None or j.get("out") is None:
            return None
        out = Path(j["out"])
        files = []
        for p in sorted(out.rglob("*")):
            if p.is_file():
                rel = p.relative_to(out).as_posix()
                files.append({"rel_path": rel, "content_type": _guess_content_type(p), "size": p.stat().st_size})
        return j["site"], out, files

    def resolve_site_artifact(self, job_id: str, rel_path: str) -> Optional[Path]:
        j = self._site_jobs.get(job_id)
        if j is None or j.get("out") is None:
            return None
        target = _safe_rel_path(Path(j["out"]), rel_path)
        return target if target.is_file() else None

    # ---- Multi Jobs ----
    def submit_multi_job(self, cams: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        pipe_opts = _pipe_of(options)
        maxp = min(max(1, int(options.get("multi_max_parallel") or 2)), len(cams))
        par = bool(options.get("multi_parallel", True)) and maxp > 1
        run_root = self.state.root / "multi" / uuid.uuid4().hex[:12]

        # Ensure SharedVLMServer is started before spawning parallel camera workers so all workers share one llama-server
        vlm_url: Optional[str] = None
        if par:
            for i, c in enumerate(cams):
                c["root"] = run_root / f"cam{i + 1}"
            if self.runner.engine is not None:
                vlm_url = (self.runner.engine.models or {}).get("vlm", {}).get("server_url")
                self.runner.engine.close()
                self.runner.engine = None
                with self.state.lock:
                    self.state.models_state = "idle"
            if not vlm_url:
                st_vlm = SharedVLMServer.instance().status()
                if st_vlm.get("state") == "ready" and st_vlm.get("url"):
                    vlm_url = st_vlm["url"]

        def launch_seq(c: dict[str, Any]) -> str:
            while self.runner.loading() or self.runner.running():
                time.sleep(0.3)
            jid = uuid.uuid4().hex[:12]
            self.submit_job(jid, c["path"], c["name"], "", options)
            return jid

        def launch_par(c: dict[str, Any]) -> str:
            realtime = efsource.is_stream(c["path"])
            vlm_extra = {"server_url": vlm_url} if vlm_url else {}
            cfg = {
                "root": str(c["root"]),
                "video": str(c["path"]),
                "demo": False,
                "local": self.local,
                "settings": dict(self.state.settings, **pipe_opts, realtime=realtime),
                "extra": {"models": {"vlm": vlm_extra}},
                "flock": str(run_root / "load.lock"),
                "stream_max_sec": max(10.0, float(options.get("camera_max_min") or 30) * 60) if realtime else None,
            }
            proc = cam_worker.CamProc(cfg)
            jid = uuid.uuid4().hex[:12]
            self._jobs[jid] = {
                "id": jid,
                "path": Path(c["path"]),
                "name": c["name"],
                "t0": time.time(),
                "cancelled": False,
                "final": None,
                "rdir": None,
                "realtime": realtime,
                "vinfo": _video_info(Path(c["path"])),
                "root": c["root"],
                "remote": proc,
            }
            return jid

        def release_par(jid: str) -> None:
            j = self._jobs.get(jid)
            if j is not None and j.get("remote") is not None:
                j["remote"].close()

        def poll(jid: str) -> dict[str, Any]:
            j = self._jobs[jid]
            v = self._view(j)
            key = None
            if v["status"] == "done":
                rd = self._result_dir(j)
                key = rd.name if rd else (self.state.cam_name if j.get("remote") is None else j["path"].stem) or j["path"].stem
            return {"status": v["status"], "key": key, "error": v["error"]}

        def alerts_of(jid: str, final: bool):
            j = self._jobs[jid]
            if not final:
                if j.get("remote") is not None:
                    return multicam.from_live_alerts(j["remote"].snap()["alerts"])
                with self.state.lock:
                    return multicam.from_live_alerts(list(self.state.alerts))
            rd = self._result_dir(j)
            if rd is None or not (rd / "result.json").is_file():
                return None
            return multicam.from_result_alerts(json.loads((rd / "result.json").read_text(encoding="utf-8")))

        def stop_cam(jid: str) -> None:
            self._stop_job(self._jobs[jid])

        mr = multicam.MultiRun(
            cams,
            launch=launch_par if par else launch_seq,
            poll=poll,
            stop=stop_cam,
            alerts_of=alerts_of,
            parallel=maxp if par else 1,
            release=release_par if par else None,
        ).start()
        self._multis[mr.id] = mr
        return {"multi_id": mr.id, "cameras": [c["name"] for c in mr.cams], "parallel": maxp if par else 1}

    def get_multi_job(self, mid: str) -> Optional[dict[str, Any]]:
        mr = self._multis.get(mid)
        return mr.snapshot() if mr is not None else None

    def get_multi_results(self, mid: str) -> Optional[list[dict[str, Any]]]:
        mr = self._multis.get(mid)
        if mr is None:
            return None
        out = []
        for c in mr.snapshot()["cams"]:
            j = self._jobs.get(c["job_id"]) if c["job_id"] else None
            rd = self._result_dir(j) if (j is not None and c["status"] == "done") else None
            if rd is not None and (rd / "result.json").is_file():
                out.append(
                    {
                        "idx": c["idx"],
                        "name": c["name"],
                        "job_id": c["job_id"],
                        "key": rd.name,
                        "result": json.loads((rd / "result.json").read_text(encoding="utf-8")),
                    }
                )
        return out

    def finish_multi_job(self, mid: str) -> Optional[dict[str, Any]]:
        mr = self._multis.get(mid)
        if mr is None:
            return None
        mr.finish()
        return {"status": "finishing"}

    # ---- Camera Push Sessions ----
    def create_camera_session(self) -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        efsource.PUSH[sid] = efsource.PushBuffer()
        return {"sid": sid, "ws": f"/internal/camera/{sid}/push"}

    def get_push_buffer(self, sid: str) -> Any:
        return efsource.PUSH.get(sid)

    def pop_push_buffer(self, sid: str) -> None:
        efsource.PUSH.pop(sid, None)

    def close(self) -> None:
        try:
            self.runner.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            cam_worker.close_all()
        except Exception:  # noqa: BLE001
            pass
        try:
            SharedVLMServer.instance().close()
        except Exception:  # noqa: BLE001
            pass
