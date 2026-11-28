"""Pure-CPU demo runtime for GPU_server (--demo mode).

STRICT CONSTRAINT: This module MUST NEVER import torch, cv2, numpy, or ultralytics.
"""
from __future__ import annotations

import io
import json
import math
import os
import platform
import re
import shutil
import struct
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Optional

STAGES = [
    "Khởi động & nạp model",
    "Tách hiện vật",
    "Nạp VLM",
    "Phân tích video",
    "Ghi kết quả",
]

SITE_STAGES = ["Cắt khung đã khử méo", "Phát hiện người + ghép các camera", "Hoàn tất"]
SITE_STAGES_EV = [
    "Cắt khung đã khử méo",
    "Phát hiện người + ghép các camera",
    "Phân tích hành vi bất thường (từng camera)",
    "Chiếu hành vi bất thường lên map",
    "Hoàn tất",
]

# Tiny valid 1x1 JPEG bytes (fallback if Pillow is unavailable)
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010101004800480000"
    "ffdb004300030202030202030303030403030405"
    "0805050404050a070706080c0a0c0c0b0a0b0b0d"
    "0e12100d0e110e0b0b1016101113141515150c0f"
    "171816141812141514ffc0000b08000100010101"
    "1100ffc4001f0000010501010101010100000000"
    "000000000102030405060708090a0bffc400b510"
    "0002010303020403050504040000017d01020300"
    "041105122131410613516107227114328191a108"
    "2342b1c11552d1f02433627282090a161718191a"
    "25262728292a3435363738393a43444546474849"
    "4a535455565758595a636465666768696a737475"
    "767778797a838485868788898a92939495969798"
    "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9ba"
    "c2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2"
    "e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda"
    "0008010100003f00fb28a2803fffd9"
)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _pure_python_png(width: int = 320, height: int = 240, rgb: tuple[int, int, int] = (34, 30, 26)) -> bytes:
    """Create a valid RGB PNG using only Python stdlib."""
    raw = bytearray()
    row = bytes(rgb) * width
    for _ in range(height):
        raw.append(0)
        raw.extend(row)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=6)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def minimal_mp4_bytes() -> bytes:
    """Return minimal valid ISO BMFF MP4 container bytes (ftyp + moov)."""
    ftyp_body = b"isom" + struct.pack(">I", 512) + b"isomiso2mp41"
    ftyp = struct.pack(">I", 8 + len(ftyp_body)) + b"ftyp" + ftyp_body
    # mvhd (version 0, 100 bytes body)
    mvhd_body = struct.pack(">IIII", 0, 0, 0, 1000) + struct.pack(">IIHH", 1000, 0x00010000, 0x0100, 0)
    mvhd_body += b"\x00" * 8
    mvhd_body += struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    mvhd_body += b"\x00" * 24 + struct.pack(">I", 2)
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    return ftyp + moov


def _render_frame_jpeg(
    frame_idx: int,
    n_frames: int,
    people: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    width: int = 640,
    height: int = 360,
) -> bytes:
    """Render a synthetic JPEG frame using Pillow (falls back to tiny JPEG if PIL missing)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _TINY_JPEG

    img = Image.new("RGB", (width, height), (38, 34, 30))
    draw = ImageDraw.Draw(img)
    # Gallery floor
    draw.rectangle([0, int(height * 0.62), width, height], fill=(58, 52, 46))
    # Exhibits
    for art in artifacts:
        ax = int(art["x"] * width)
        ay = int(art["y"] * height)
        draw.rectangle([ax - 28, ay - 46, ax + 28, ay + 10], fill=(75, 118, 175), outline=(140, 185, 235), width=2)
        draw.text((ax - 24, ay + 14), f"E{art['id']}", fill=(215, 220, 230))
    # People boxes
    for p in people:
        px = int(p["x"] * width)
        py = int(p["y"] * height)
        bw, bh = 42, 110
        x1, y1, x2, y2 = px - bw // 2, py - bh, px + bw // 2, py
        st = p.get("state", "idle")
        color = (225, 65, 65) if st == "alert" else (240, 195, 55) if st == "contact" else (70, 200, 110)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(4, y1 - 14)), f"ID {p['id']}", fill=color)
    draw.text((12, 10), f"DEMO {frame_idx + 1}/{n_frames}", fill=(220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return buf.getvalue()


def _render_evidence_sheet_jpeg(task_name: str, behavior: str, track_id: int, artifact_id: int) -> bytes:
    """Render a valid JPEG evidence sheet for a VLM task."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _TINY_JPEG

    w, h = 640, 360
    img = Image.new("RGB", (w, h), (26, 28, 34))
    draw = ImageDraw.Draw(img)
    cols, rows = 4, 2
    cw, ch = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw + 6, r * ch + 6
            x1, y1 = (c + 1) * cw - 6, (r + 1) * ch - 6
            draw.rectangle([x0, y0, x1, y1], fill=(44, 48, 58), outline=(100, 120, 150), width=1)
            draw.rectangle([x0 + 48, y0 + 40, x0 + 96, y0 + 135], outline=(235, 80, 80) if behavior != "normal" else (80, 200, 120), width=2)
            draw.text((x0 + 8, y0 + 6), f"f{(r * cols + c) * 2}", fill=(200, 205, 215))
    draw.text((12, h - 20), f"{task_name} | track={track_id} art={artifact_id} [{behavior}]", fill=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def _render_map_png(
    artifacts: list[dict[str, Any]],
    people: list[dict[str, Any]] | None = None,
    width: int = 800,
    height: int = 600,
) -> bytes:
    """Render a valid PNG 2D floorplan map."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _pure_python_png(320, 240)

    img = Image.new("RGB", (width, height), (30, 32, 38))
    draw = ImageDraw.Draw(img)
    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill=(48, 52, 60), width=1)
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill=(48, 52, 60), width=1)
    draw.rectangle([24, 24, width - 24, height - 24], outline=(165, 175, 190), width=3)
    for art in artifacts:
        ax, ay = int(art["x"] * width), int(art["y"] * height)
        draw.rectangle([ax - 14, ay - 14, ax + 14, ay + 14], fill=(70, 130, 210), outline=(180, 215, 255), width=2)
        draw.text((ax - 12, ay + 18), art.get("label") or f"Art {art['id']}", fill=(210, 220, 235))
    for p in people or []:
        px, py = int(p["x"] * width), int(p["y"] * height)
        draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(235, 75, 75))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def demo_tuning_schema() -> dict[str, Any]:
    """Lightweight tuning schema in --demo mode without importing torch/cv2."""
    return {
        "groups": [
            {
                "id": "alarm_id",
                "title": "Khung đỏ & ID",
                "description": "Giữ khung đỏ sau cảnh báo và kế thừa cờ ANOMALY khi tracker đổi ID.",
                "needs_reload": False,
            },
            {
                "id": "contact_gate",
                "title": "Cổng chạm & Sự kiện",
                "description": "Ngưỡng khoảng cách/thời lượng để tạo sự kiện chạm gửi VLM.",
                "needs_reload": False,
            },
            {
                "id": "pose",
                "title": "Pose & Làm mượt",
                "description": "Làm mượt keypoint theo thời gian và độ trễ chuyển trạng thái.",
                "needs_reload": False,
            },
        ],
        "params": [
            {
                "key": "alarm_persist",
                "path": "RT.alarm_persist",
                "type": "bool",
                "group": "alarm_id",
                "label_vi": "Giữ khung đỏ suốt video sau cảnh báo",
                "help_vi": "Bật: giữ khung đỏ cho người bị gắn cờ ANOMALY tới hết video.",
                "min": None,
                "max": None,
                "step": None,
                "needs_reload": False,
                "default": True,
            },
            {
                "key": "contact_hold_sec",
                "path": "CFG.contact_hold_sec",
                "type": "float",
                "group": "contact_gate",
                "label_vi": "Thời lượng chạm tối thiểu (giây)",
                "help_vi": "Thời gian tiếp xúc liên tục trước khi gửi VLM.",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "needs_reload": False,
                "default": 0.4,
            },
            {
                "key": "pose_gate_norm_dist",
                "path": "RT.pose_gate_norm_dist",
                "type": "float",
                "group": "pose",
                "label_vi": "Ngưỡng khoảng cách chuẩn hoá",
                "help_vi": "Khoảng cách chuẩn hoá tối đa để kích hoạt kiểm tra tay.",
                "min": 0.5,
                "max": 4.0,
                "step": 0.1,
                "needs_reload": False,
                "default": 1.5,
            },
        ],
    }


DEMO_SITES: dict[str, dict[str, Any]] = {
    "demo_gallery": {
        "site_id": "demo_gallery",
        "name": "Demo Gallery (2 cams)",
        "room_id": "demo_gallery",
        "sample_fps": 4.0,
        "room_bounds": [-4.0, -3.0, 4.0, 3.0],
        "cameras": {
            "C1": {
                "label": "CAM 1",
                "width": 960,
                "height": 540,
                "K": [[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]],
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                "R_world_to_camera": [[1.0, 0.0, 0.0], [0.0, -0.5, -0.866], [0.0, 0.866, -0.5]],
                "t_world_to_camera": [0.0, 0.5, 4.5],
            },
            "C2": {
                "label": "CAM 2",
                "width": 960,
                "height": 540,
                "K": [[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]],
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                "R_world_to_camera": [[-1.0, 0.0, 0.0], [0.0, 0.5, -0.866], [0.0, -0.866, -0.5]],
                "t_world_to_camera": [0.0, 0.5, 4.5],
            },
        },
    }
}


def demo_list_sites() -> list[dict[str, Any]]:
    return [
        {
            "site_id": s["site_id"],
            "name": s["name"],
            "cameras": list(s["cameras"].keys()),
            "labels": {cid: c.get("label", cid) for cid, c in s["cameras"].items()},
        }
        for s in DEMO_SITES.values()
    ]


def demo_load_site(site_id: str) -> dict[str, Any]:
    if site_id not in DEMO_SITES:
        raise FileNotFoundError(f'Không có hồ sơ địa điểm "{site_id}"')
    return json.loads(json.dumps(DEMO_SITES[site_id]))


def _safe_key(raw_name: str) -> str:
    stem = Path(raw_name or "demo").stem
    cleaned = re.sub(r"[^\w.-]+", "_", stem).strip("_.")
    return (cleaned[:40] or "demo")


def _demo_duration(options: dict[str, Any]) -> float:
    raw = options.get("demo_duration_s")
    if raw is None:
        raw = os.environ.get("EF_DEMO_DURATION_S", "3.0")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = 3.0
    return max(0.15, min(60.0, val))


def _safe_rel_path(base: Path, rel_path: str) -> Path:
    if not rel_path or "\x00" in rel_path or rel_path.startswith(("/", "\\")):
        raise ValueError("Invalid path")
    if "\\" in rel_path:
        raise ValueError("Invalid path")
    parts = rel_path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("Path traversal rejected")
    base_resolved = base.resolve()
    target = (base_resolved / rel_path).resolve()
    if base_resolved not in target.parents:
        raise ValueError("Path traversal rejected")
    return target


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".json": "application/json",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
    }.get(ext, "application/octet-stream")


def _list_dir_artifacts(base: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not base.is_dir():
        return files
    base_res = base.resolve()
    for p in sorted(base_res.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base_res).as_posix()
        files.append(
            {
                "rel_path": rel,
                "content_type": _guess_content_type(p),
                "size": p.stat().st_size,
            }
        )
    return files


class _DemoPushBuffer:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def put(self, data: bytes) -> None:
        if not self.closed:
            self.frames.append(data)
            if len(self.frames) > 30:
                self.frames = self.frames[-30:]

    def close(self) -> None:
        self.closed = True


class DemoRuntime:
    """Pure-CPU demo worker runtime backing /internal/* endpoints."""

    def __init__(self, data_dir: Path, mode: str = "full", det_conf: Optional[float] = None) -> None:
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        (self.root / "site_jobs").mkdir(parents=True, exist_ok=True)
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        self.mode = mode or "full"
        self.det_conf = det_conf
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._site_jobs: dict[str, dict[str, Any]] = {}
        self._multi_jobs: dict[str, dict[str, Any]] = {}
        self._push_buffers: dict[str, _DemoPushBuffer] = {}
        self._active_job_id: Optional[str] = None

    def det_conf_state(self) -> dict[str, Any]:
        return {
            "det_conf": self.det_conf,
            "det_conf_default": 0.01,
            "det_conf_effective": 0.01 if self.det_conf is None else self.det_conf,
        }

    def set_det_conf(self, conf: Optional[float]) -> dict[str, Any]:
        self.det_conf = conf
        return {**self.det_conf_state(), "applied_now": True}

    def health_info(self) -> dict[str, Any]:
        with self._lock:
            active_id = self._active_job_id
            busy = False
            if active_id and active_id in self._jobs:
                busy = self._jobs[active_id]["status"] == "running"
            if not busy and any(s["status"] == "running" for s in self._site_jobs.values()):
                busy = True
            if not busy:
                active_id = None
        return {
            "models": {"state": "ready", "msg": "Demo mode ready (CPU)"},
            "vlm": {
                "state": "ready",
                "url": "demo://local",
                "model": "Qwen3-VL-8B-Instruct (demo)",
                "shared": True,
            },
            "busy": busy,
            "active_job_id": active_id,
            **self.det_conf_state(),
        }

    def tuning_schema(self) -> dict[str, Any]:
        return demo_tuning_schema()

    def list_sites(self) -> list[dict[str, Any]]:
        return demo_list_sites()

    def load_site(self, site_id: str) -> dict[str, Any]:
        return demo_load_site(site_id)

    def submit_job(
        self,
        job_id: str,
        video_path: Optional[Path],
        video_name: str,
        source: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()) or any(
                sj["status"] == "running" for sj in self._site_jobs.values()
            ):
                raise RuntimeError("GPU worker is busy running another job")

            key = _safe_key(video_name or source or job_id)
            job_dir = self.root / "jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            duration_s = _demo_duration(options)
            realtime = bool(options.get("realtime"))
            artifacts_meta = [
                {"id": 1, "label": "Bronze Statue", "box": [180.0, 120.0, 280.0, 310.0], "pedestal": True, "x": 0.25, "y": 0.55},
                {"id": 2, "label": "Ceramic Vase", "box": [460.0, 140.0, 560.0, 295.0], "pedestal": True, "x": 0.55, "y": 0.50},
                {"id": 3, "label": "Oil Painting", "box": [700.0, 110.0, 840.0, 330.0], "pedestal": False, "x": 0.80, "y": 0.60},
            ]
            # Write initial floorplan map.png immediately
            (job_dir / "map.png").write_bytes(_render_map_png(artifacts_meta))

            job = {
                "job_id": job_id,
                "status": "running",
                "stage": 0,
                "stages": list(STAGES),
                "progress": 0.0,
                "key": key,
                "error": None,
                "name": video_name or (Path(source).name if source else f"{key}.mp4"),
                "realtime": realtime,
                "log": ["[demo] Started pure-CPU demo job"],
                "job_dir": job_dir,
                "video_path": video_path,
                "source": source,
                "options": options,
                "duration_s": duration_s,
                "n_frames": 15,
                "fps": 5.0,
                "frame_idx": 0,
                "live_jpeg": None,
                "frame_seq": 0,
                "events": [],
                "people": [],
                "artifacts": [{"id": a["id"], "x": a["x"], "y": a["y"]} for a in artifacts_meta],
                "artifacts_meta": artifacts_meta,
                "timeline_frames": {},
                "map_version": 1,
                "cancel_ev": threading.Event(),
                "finish_ev": threading.Event(),
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id

        t = threading.Thread(target=self._run_demo_job, args=(job,), daemon=True, name=f"demo-{job_id[:8]}")
        t.start()
        return {"job_id": job_id, "status": "running"}

    def _run_demo_job(self, job: dict[str, Any]) -> None:
        job_dir: Path = job["job_dir"]
        key: str = job["key"]
        n_frames: int = job["n_frames"]
        fps: float = job["fps"]
        step_sleep = max(0.01, job["duration_s"] / n_frames)
        artifacts_meta = job["artifacts_meta"]
        artifacts_map = [{"id": a["id"], "x": a["x"], "y": a["y"]} for a in artifacts_meta]

        # Prepare live.mp4 right away if source/uploaded video exists, else minimal valid MP4
        live_mp4 = job_dir / "live.mp4"
        src_vp: Optional[Path] = job.get("video_path")
        if src_vp is not None and Path(src_vp).is_file():
            shutil.copyfile(src_vp, live_mp4)
        else:
            live_mp4.write_bytes(minimal_mp4_bytes())

        scripted_events = {
            3: {
                "seq": 1,
                "kind": "contact",
                "track": 1,
                "artifact": 1,
                "artifacts": [1],
                "decision": "CANDIDATE",
                "behavior": "touching",
                "confidence": 0.62,
                "evidence": "Hand entered contact zone near Bronze Statue",
                "label": "touching",
                "start_s": 0.4,
                "end_s": 0.8,
            },
            6: {
                "seq": 2,
                "kind": "vlm",
                "track": 1,
                "artifact": 1,
                "artifacts": [1],
                "decision": "ANOMALY",
                "behavior": "touching",
                "confidence": 0.88,
                "evidence": "Visitor touched Bronze Statue surface with right hand",
                "label": "touching",
                "start_s": 0.4,
                "end_s": 1.2,
                "task": "t001",
            },
            10: {
                "seq": 3,
                "kind": "vlm",
                "track": 2,
                "artifact": 2,
                "artifacts": [2],
                "decision": "NORMAL",
                "behavior": "normal",
                "confidence": 0.91,
                "evidence": "Visitor observing Ceramic Vase without physical contact",
                "label": "normal",
                "start_s": 1.4,
                "end_s": 2.0,
                "task": "t002",
            },
            12: {
                "seq": 4,
                "kind": "vlm",
                "track": 1,
                "artifact": 1,
                "artifacts": [1],
                "decision": "ANOMALY",
                "behavior": "stealing",
                "confidence": 0.93,
                "evidence": "Visitor attempting to lift Bronze Statue off pedestal",
                "label": "stealing",
                "start_s": 2.0,
                "end_s": 2.6,
                "task": "t003",
            },
        }

        try:
            for idx in range(n_frames):
                if job["cancel_ev"].is_set():
                    with self._lock:
                        job["status"] = "cancelled"
                        job["stage"] = 4
                        job["log"].append("[demo] Job cancelled")
                        if self._active_job_id == job["job_id"]:
                            self._active_job_id = None
                    return

                stage = 0 if idx == 0 else 1 if idx == 1 else 2 if idx == 2 else 3
                p1x = round(0.24 + 0.04 * math.sin(idx * 0.4), 4)
                p1y = round(0.64 + 0.03 * math.cos(idx * 0.3), 4)
                p2x = round(0.56 + 0.05 * math.cos(idx * 0.35), 4)
                p2y = round(0.68 + 0.03 * math.sin(idx * 0.4), 4)
                p1_state = "alert" if idx >= 6 else ("contact" if idx >= 3 else "idle")
                p2_state = "contact" if 9 <= idx <= 10 else "idle"
                ppl = [
                    {"id": 1, "x": p1x, "y": p1y, "state": p1_state},
                    {"id": 2, "x": p2x, "y": p2y, "state": p2_state},
                ]
                jpg = _render_frame_jpeg(idx, n_frames, ppl, artifacts_map)

                with self._lock:
                    job["stage"] = stage
                    job["frame_idx"] = idx + 1
                    job["progress"] = round((idx + 1) / n_frames, 3)
                    job["people"] = ppl
                    job["live_jpeg"] = jpg
                    job["frame_seq"] += 1
                    job["timeline_frames"][str(idx)] = [[1, p1x, p1y], [2, p2x, p2y]]
                    if idx in scripted_events:
                        ev = dict(scripted_events[idx])
                        job["events"].append(ev)
                        job["log"].append(f"[demo] event #{ev['seq']}: {ev['decision']} {ev['behavior']}")

                if not job["finish_ev"].is_set():
                    time.sleep(step_sleep)

            # Finalize stage 4 and write all artifacts BEFORE marking status='done'
            with self._lock:
                job["stage"] = 4
                job["log"].append("[demo] Writing artifacts...")

            tasks_spec = [
                ("t001", 1, [1], [2, 6], "ANOMALY", "touching", 0.88, "Visitor touched Bronze Statue surface with right hand"),
                ("t002", 2, [2], [7, 10], "NORMAL", "normal", 0.91, "Visitor observing Ceramic Vase without physical contact"),
                ("t003", 1, [1], [10, 13], "ANOMALY", "stealing", 0.93, "Visitor attempting to lift Bronze Statue off pedestal"),
            ]

            vlm_tasks = []
            events_out = [
                {
                    "event_id": "e0",
                    "track_id": 1,
                    "artifact_id": 1,
                    "hand": "right",
                    "start_frame": 2,
                    "end_frame": 4,
                    "start_time_s": 0.4,
                    "end_time_s": 0.8,
                    "decision": "CANDIDATE",
                    "behavior": "touching",
                    "vlm_confidence": 0.62,
                    "vlm_evidence": "Hand entered contact zone near Bronze Statue",
                    "task": None,
                    "alarm_spans": [],
                    "alarm_onset_source": None,
                }
            ]
            alerts_out = []

            for idx_t, (tname, tid, art_ids, win, dec, beh, conf, evid) in enumerate(tasks_spec, start=1):
                sheet_dir = job_dir / "files" / key / "vlm_inputs" / tname
                sheet_dir.mkdir(parents=True, exist_ok=True)
                (sheet_dir / "sheet.jpg").write_bytes(_render_evidence_sheet_jpeg(tname, beh, tid, art_ids[0]))

                vlm_tasks.append(
                    {
                        "task": tname,
                        "task_id": f"demo/{key}/{tname}",
                        "track_id": tid,
                        "artifact_ids": art_ids,
                        "window_frames": win,
                        "status": "ok",
                        "decision": dec,
                        "label": beh,
                        "confidence": conf,
                        "evidence": evid,
                        "n_images": 8,
                        "n_input_tokens": 512,
                        "latency_s": 0.12,
                        "infer_s": 0.1,
                        "prefill_s": 0.04,
                        "queue_wait_s": 0.01,
                        "error": None,
                    }
                )
                eid = f"e{idx_t}"
                sf, ef = win[0], win[1]
                st_s, et_s = round(sf / fps, 3), round(ef / fps, 3)
                spans = [{"start_frame": sf, "end_frame": ef, "start_time_s": st_s, "end_time_s": et_s}] if dec == "ANOMALY" else []
                ev_item = {
                    "event_id": eid,
                    "track_id": tid,
                    "artifact_id": art_ids[0],
                    "hand": "right",
                    "start_frame": sf,
                    "end_frame": ef,
                    "start_time_s": st_s,
                    "end_time_s": et_s,
                    "decision": dec,
                    "behavior": beh,
                    "vlm_confidence": conf,
                    "vlm_evidence": evid,
                    "task": tname,
                    "alarm_spans": spans,
                    "alarm_onset_source": "vlm" if dec == "ANOMALY" else None,
                }
                events_out.append(ev_item)
                if dec == "ANOMALY":
                    alerts_out.append(
                        {
                            "event_id": eid,
                            "track_id": tid,
                            "artifact_id": art_ids[0],
                            "behavior": beh,
                            "confidence": conf,
                            "evidence": evid,
                            "alarm_start_frame": sf,
                            "alarm_start_time_s": st_s,
                            "alarm_end_frame": ef,
                            "alarm_end_time_s": et_s,
                            "onset_source": "vlm",
                            "task": tname,
                        }
                    )

            with self._lock:
                people_snap = list(job["people"])
                frames_snap = dict(job["timeline_frames"])

            (job_dir / "map.png").write_bytes(_render_map_png(artifacts_meta, people_snap))
            map_data = {
                "has_map": True,
                "version": 1,
                "fps": fps,
                "people": people_snap,
                "artifacts": artifacts_map,
                "frames": frames_snap,
            }
            (job_dir / "map_data.json").write_text(json.dumps(map_data), encoding="utf-8")

            alerts_by_beh: dict[str, int] = {}
            for a in alerts_out:
                alerts_by_beh[a["behavior"]] = alerts_by_beh.get(a["behavior"], 0) + 1

            result_obj = {
                "schema_version": "1.0",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "video": {
                    "path": str(live_mp4),
                    "name": job["name"],
                    "key": key,
                    "fps": fps,
                    "n_frames": n_frames,
                    "width": 960,
                    "height": 540,
                },
                "summary": {
                    "n_people_tracks": 2,
                    "n_events": len(events_out),
                    "n_alerts": len(alerts_out),
                    "alerts_by_behavior": alerts_by_beh,
                    "n_vlm_tasks": len(vlm_tasks),
                    "has_anomaly": bool(alerts_out),
                },
                "alerts": alerts_out,
                "events": events_out,
                "vlm_tasks": vlm_tasks,
                "track_summary": {
                    "1": {"behavior": "stealing", "n_tasks": 2},
                    "2": {"behavior": "normal", "n_tasks": 1},
                },
                "artifacts": [
                    {
                        "id": a["id"],
                        "label": a["label"],
                        "box": a["box"],
                        "pedestal": a["pedestal"],
                    }
                    for a in artifacts_meta
                ],
                "scenes": [{"start_frame": 0, "segment_frame": 0, "n_artifacts": len(artifacts_meta)}],
                "performance": {
                    "pipeline_fps": 25.0,
                    "wall_s": round(job["duration_s"], 2),
                    "wall_with_vlm_s": round(job["duration_s"], 2),
                    "frames_processed": n_frames,
                    "frames_dropped": 0,
                    "e2e_ms": {"mean": 40.0, "p95": 45.0},
                    "stages": {"demo": {"mean_ms": 20.0, "p95_ms": 25.0, "frames": n_frames}},
                    "vlm": {"n_tasks": len(vlm_tasks), "n_ok": len(vlm_tasks), "infer_mean_s": 0.1},
                    "gpu_peak_gb": 0.0,
                },
                "warnings": [],
                "config": {
                    "version": "ef-gpu-1",
                    "mode": "demo",
                    "models": [{"role": "Demo", "name": "CPU Demo", "detail": "pure-python"}],
                    "environment": {"python": platform.python_version(), "gpus": []},
                },
            }
            (job_dir / "result.json").write_text(json.dumps(result_obj, indent=2), encoding="utf-8")

            with self._lock:
                if job["cancel_ev"].is_set():
                    job["status"] = "cancelled"
                else:
                    job["status"] = "done"
                    job["progress"] = 1.0
                    job["stage"] = 4
                    job["log"].append("[demo] Completed")
                if self._active_job_id == job["job_id"]:
                    self._active_job_id = None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["log"].append(f"[demo] Error: {job['error']}")
                if self._active_job_id == job["job_id"]:
                    self._active_job_id = None

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            return {
                "job_id": j["job_id"],
                "status": j["status"],
                "stage": j["stage"],
                "stages": list(j["stages"]),
                "progress": j["progress"],
                "key": j["key"],
                "error": j["error"],
                "name": j["name"],
                "realtime": j["realtime"],
                "log": list(j["log"][-40:]),
            }

    def cancel_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            if j["status"] in ("done", "error", "cancelled"):
                return {"status": j["status"]}
            j["cancel_ev"].set()
            return {"status": "cancelling"}

    def finish_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            if j["status"] in ("done", "error", "cancelled"):
                return {"status": j["status"]}
            j["finish_ev"].set()
            return {"status": "finishing"}

    def get_live_frame(self, job_id: str) -> Optional[bytes]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            return j["live_jpeg"]

    def get_live_frame_with_seq(self, job_id: str) -> tuple[int, Optional[bytes], str]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return -1, None, "error"
            return j["frame_seq"], j["live_jpeg"], j["status"]

    def get_live_events(self, job_id: str, after: int = 0) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            fps = j["fps"]
            n_frames = j["n_frames"]
            fi = j["frame_idx"]
            st = j["status"]
            arts = [
                {"id": a["id"], "label": a["label"], "pedestal": a["pedestal"], "box": a["box"]}
                for a in j["artifacts_meta"]
            ]
            meta = {"artifacts": arts, "n_frames": n_frames, "fps": fps, "width": 960, "height": 540}
            status_obj = {
                "frame": fi,
                "n_frames": n_frames,
                "t_s": round(fi / fps, 2),
                "live_fps": fps,
                "dropped": 0,
                "vlm_pending": 0,
            }
            evs = [dict(e) for e in j["events"] if e["seq"] > after]
            return {"meta": meta, "status": status_obj, "events": evs, "job": st}

    def get_map_image(self, job_id: str) -> Optional[Path]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            p = j["job_dir"] / "map.png"
            return p if p.is_file() else None

    def get_map_data(self, job_id: str, full: int = 0) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            mp = j["job_dir"] / "map.png"
            out: dict[str, Any] = {
                "has_map": mp.is_file(),
                "version": j["map_version"],
                "fps": j["fps"],
                "people": list(j["people"]),
                "artifacts": list(j["artifacts"]),
            }
            if full:
                out["frames"] = dict(j["timeline_frames"])
            return out

    def list_job_artifacts(self, job_id: str) -> Optional[tuple[str, Path, list[dict[str, Any]]]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            key = j["key"]
            job_dir = j["job_dir"]
        return key, job_dir, _list_dir_artifacts(job_dir)

    def resolve_job_artifact(self, job_id: str, rel_path: str) -> Optional[Path]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            job_dir = j["job_dir"]
        target = _safe_rel_path(job_dir, rel_path)
        return target if target.is_file() else None

    # ---- Calibrated Site Jobs ----
    def submit_site_job(
        self,
        job_id: str,
        site_id: str,
        site_def: dict[str, Any],
        videos: dict[str, Path],
        events: bool,
    ) -> dict[str, Any]:
        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()) or any(
                sj["status"] == "running" for sj in self._site_jobs.values()
            ):
                raise RuntimeError("GPU worker is busy running another job")
            job_dir = self.root / "site_jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            stages = list(SITE_STAGES_EV if events else SITE_STAGES)
            sjob = {
                "job_id": job_id,
                "site": site_id,
                "site_def": site_def,
                "videos": videos,
                "events": events,
                "status": "running",
                "stage": 0,
                "stages": stages,
                "error": None,
                "log": [f"[demo-site] Started calibrated site job {site_id}"],
                "t0": time.time(),
                "job_dir": job_dir,
            }
            self._site_jobs[job_id] = sjob
            self._active_job_id = job_id

        t = threading.Thread(target=self._run_demo_site_job, args=(sjob,), daemon=True, name=f"demo-site-{job_id[:8]}")
        t.start()
        return {"job_id": job_id, "cameras": list(videos.keys()), "events": events}

    def _run_demo_site_job(self, sjob: dict[str, Any]) -> None:
        job_dir: Path = sjob["job_dir"]
        site_def: dict[str, Any] = sjob["site_def"]
        cams = list(sjob["videos"].keys())
        try:
            dur = _demo_duration({})
            step = max(0.02, dur / max(1, len(sjob["stages"])))
            videos_dir = job_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            for cam, src in sjob["videos"].items():
                dst = videos_dir / f"{cam}.mp4"
                if Path(src).is_file():
                    shutil.copyfile(src, dst)
                else:
                    dst.write_bytes(minimal_mp4_bytes())

            for st_idx in range(len(sjob["stages"]) - 1):
                with self._lock:
                    sjob["stage"] = st_idx
                    sjob["log"].append(f"[demo-site] Stage: {sjob['stages'][st_idx]}")
                time.sleep(step)

            artifacts = [
                {"id": 1, "label": "Bronze Statue", "x": -1.5, "y": 0.5},
                {"id": 2, "label": "Ceramic Vase", "x": 1.5, "y": -0.5},
            ]
            map_artifacts = [
                {"id": 1, "label": "Bronze Statue", "x": 0.3, "y": 0.4},
                {"id": 2, "label": "Ceramic Vase", "x": 0.7, "y": 0.6},
            ]
            (job_dir / "global_2d_map.png").write_bytes(_render_map_png(map_artifacts))

            cam_entries = []
            for idx, cid in enumerate(cams):
                cdef = (site_def.get("cameras") or {}).get(cid, {})
                cx = -2.5 + idx * 5.0
                cam_entries.append(
                    {
                        "id": cid,
                        "label": cdef.get("label", cid),
                        "center": [cx, -2.5],
                        "fov": [[cx - 1.5, -2.5], [cx + 2.0, 2.5], [cx - 2.0, 2.5]],
                        "H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                )

            frames = []
            for fi in range(12):
                t_val = round(fi * 0.25, 3)
                frames.append(
                    {
                        "t": t_val,
                        "people": [
                            {"id": 1, "x": round(-1.5 + 0.1 * math.sin(fi * 0.4), 3), "y": round(0.4 + 0.08 * math.cos(fi * 0.3), 3), "cams": cams[:1]},
                            {"id": 2, "x": round(1.2 + 0.15 * math.cos(fi * 0.3), 3), "y": round(-0.4 + 0.1 * math.sin(fi * 0.4), 3), "cams": cams},
                        ],
                    }
                )

            events_list = []
            if sjob["events"]:
                events_list.append(
                    {
                        "id": "inc1",
                        "behavior": "touching",
                        "conf": 0.89,
                        "t0": 0.5,
                        "t1": 1.75,
                        "x": -1.48,
                        "y": 0.42,
                        "person_id": 1,
                        "cams": cams,
                        "evidence": "Visitor touched Bronze Statue across calibrated camera view",
                        "artifact": 1,
                        "path": [[0.5, -1.5, 0.4], [1.0, -1.48, 0.42], [1.75, -1.45, 0.44]],
                        "per_cam": [{"cam": c, "track": 1, "t0": 0.5, "t1": 1.75} for c in cams],
                    }
                )

            site_result = {
                "room_id": site_def.get("room_id") or sjob["site"],
                "bounds": site_def.get("room_bounds") or [-4.0, -3.0, 4.0, 3.0],
                "cameras": cam_entries,
                "fps": 4.0,
                "frames": frames,
                "events": events_list,
                "artifacts": artifacts,
                "exhibits": [
                    {"id": 1, "label": "Bronze Statue", "pedestal": False, "x": -1.5, "y": 0.5},
                    {"id": 2, "label": "Ceramic Vase", "pedestal": False, "x": 1.5, "y": -0.5},
                ],
                "map_source": "demo",
                "map": {"file": "global_2d_map.png", "T": [[100.0, 0.0, 400.0], [0.0, -100.0, 300.0], [0.0, 0.0, 1.0]], "canvas": [800, 600], "metric": "demo"},
            }
            (job_dir / "site_result.json").write_text(json.dumps(site_result, indent=2), encoding="utf-8")

            for cid in cams:
                cam_out = job_dir / "cams" / cid
                (cam_out / "vlm_inputs" / "t001").mkdir(parents=True, exist_ok=True)
                (cam_out / "vlm_inputs" / "t001" / "sheet.jpg").write_bytes(
                    _render_evidence_sheet_jpeg("t001", "touching", 1, 1)
                )
                cam_res = {
                    "version": "ef-gpu-1",
                    "artifact_global": True,
                    "video": {"key": f"site_{cid}", "name": f"{cid}.mp4", "fps": 4.0, "n_frames": 12, "width": 640, "height": 360},
                    "artifacts": [
                        {"id": 1, "label": "Bronze Statue", "box": [120, 90, 220, 240], "pedestal": False},
                        {"id": 2, "label": "Ceramic Vase", "box": [400, 110, 490, 250], "pedestal": False},
                    ],
                    "events": [
                        {
                            "event_id": 1,
                            "track_id": 1,
                            "artifact_id": 1,
                            "start_frame": 2,
                            "end_frame": 7,
                            "start_time_s": 0.5,
                            "end_time_s": 1.75,
                            "decision": "ANOMALY",
                            "behavior": "touching",
                            "vlm_confidence": 0.89,
                            "vlm_evidence": "Visitor touched Bronze Statue across calibrated camera view",
                            "task": "t001",
                            "alarm_spans": [{"start_frame": 2, "end_frame": 7, "start_time_s": 0.5, "end_time_s": 1.75}],
                        }
                    ] if sjob["events"] else [],
                    "alerts": [
                        {
                            "event_id": 1,
                            "track_id": 1,
                            "artifact_id": 1,
                            "behavior": "touching",
                            "confidence": 0.89,
                            "evidence": "Visitor touched Bronze Statue across calibrated camera view",
                            "alarm_start_frame": 2,
                            "alarm_end_frame": 7,
                            "alarm_start_time_s": 0.5,
                            "alarm_end_time_s": 1.75,
                        }
                    ] if sjob["events"] else [],
                }
                (cam_out / "result.json").write_text(json.dumps(cam_res, indent=2), encoding="utf-8")

            with self._lock:
                sjob["stage"] = len(sjob["stages"]) - 1
                sjob["status"] = "done"
                sjob["log"].append("[demo-site] Completed")
                if self._active_job_id == sjob["job_id"]:
                    self._active_job_id = None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                sjob["status"] = "error"
                sjob["error"] = f"{type(exc).__name__}: {exc}"
                sjob["log"].append(f"[demo-site] Error: {sjob['error']}")
                if self._active_job_id == sjob["job_id"]:
                    self._active_job_id = None

    def get_site_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            j = self._site_jobs.get(job_id)
            if j is None:
                return None
            return {
                "job_id": j["job_id"],
                "status": j["status"],
                "stage": j["stage"],
                "stages": list(j["stages"]),
                "error": j["error"],
                "cameras": list(j["videos"].keys()),
                "log": list(j["log"][-20:]),
                "elapsed": round(time.time() - j["t0"]),
                "progress": 1.0 if j["status"] == "done" else 0.5,
            }

    def list_site_artifacts(self, job_id: str) -> Optional[tuple[str, Path, list[dict[str, Any]]]]:
        with self._lock:
            j = self._site_jobs.get(job_id)
            if j is None:
                return None
            job_dir = j["job_dir"]
            site_id = j["site"]
        return site_id, job_dir, _list_dir_artifacts(job_dir)

    def resolve_site_artifact(self, job_id: str, rel_path: str) -> Optional[Path]:
        with self._lock:
            j = self._site_jobs.get(job_id)
            if j is None:
                return None
            job_dir = j["job_dir"]
        target = _safe_rel_path(job_dir, rel_path)
        return target if target.is_file() else None

    # ---- Multi Jobs ----
    def submit_multi_job(self, cams: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        mid = uuid.uuid4().hex[:12]
        cam_records = []
        for idx, cam in enumerate(cams):
            jid = uuid.uuid4().hex[:12]
            cam_records.append(
                {
                    "idx": idx,
                    "name": cam["name"],
                    "path": cam["path"],
                    "job_id": jid,
                    "status": "queued",
                    "error": None,
                }
            )
        mr = {
            "multi_id": mid,
            "cams": cam_records,
            "options": options,
            "status": "running",
            "cur": 0,
            "finish_ev": threading.Event(),
            "parallel": 1,
        }
        with self._lock:
            self._multi_jobs[mid] = mr
        threading.Thread(target=self._run_multi, args=(mr,), daemon=True, name=f"demo-multi-{mid[:6]}").start()
        return {"multi_id": mid, "cameras": [c["name"] for c in cam_records], "parallel": 1}

    def _run_multi(self, mr: dict[str, Any]) -> None:
        for idx, c in enumerate(mr["cams"]):
            if mr["finish_ev"].is_set():
                c["status"] = "skipped"
                continue
            mr["cur"] = idx
            c["status"] = "running"
            # Wait if single job active
            while any(j["status"] == "running" for j in self._jobs.values()):
                time.sleep(0.05)
            self.submit_job(c["job_id"], c["path"], c["name"], "", mr["options"])
            while True:
                j = self.get_job(c["job_id"])
                if j is None or j["status"] in ("done", "error", "cancelled"):
                    c["status"] = j["status"] if j else "error"
                    c["error"] = j["error"] if j else None
                    break
                if mr["finish_ev"].is_set():
                    self.finish_job(c["job_id"])
                time.sleep(0.05)
        mr["status"] = "done"

    def get_multi_job(self, mid: str) -> Optional[dict[str, Any]]:
        mr = self._multi_jobs.get(mid)
        if mr is None:
            return None
        alerts = []
        for c in mr["cams"]:
            j = self._jobs.get(c["job_id"])
            if j:
                for ev in j["events"]:
                    if ev.get("decision") == "ANOMALY":
                        alerts.append({**ev, "camera": c["name"]})
        return {
            "multi_id": mr["multi_id"],
            "status": mr["status"],
            "cur": mr["cur"],
            "cams": [
                {"idx": c["idx"], "name": c["name"], "job_id": c["job_id"], "status": c["status"], "error": c["error"]}
                for c in mr["cams"]
            ],
            "alerts": alerts,
        }

    def get_multi_results(self, mid: str) -> Optional[list[dict[str, Any]]]:
        mr = self._multi_jobs.get(mid)
        if mr is None:
            return None
        out = []
        for c in mr["cams"]:
            if c["status"] != "done":
                continue
            j = self._jobs.get(c["job_id"])
            if not j:
                continue
            rp = j["job_dir"] / "result.json"
            if rp.is_file():
                out.append(
                    {
                        "idx": c["idx"],
                        "name": c["name"],
                        "job_id": c["job_id"],
                        "key": j["key"],
                        "result": json.loads(rp.read_text(encoding="utf-8")),
                    }
                )
        return out

    def finish_multi_job(self, mid: str) -> Optional[dict[str, Any]]:
        mr = self._multi_jobs.get(mid)
        if mr is None:
            return None
        mr["finish_ev"].set()
        return {"status": "finishing"}

    # ---- Camera Push Sessions ----
    def create_camera_session(self) -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._push_buffers[sid] = _DemoPushBuffer()
        return {"sid": sid, "ws": f"/internal/camera/{sid}/push"}

    def get_push_buffer(self, sid: str) -> Optional[_DemoPushBuffer]:
        with self._lock:
            return self._push_buffers.get(sid)

    def pop_push_buffer(self, sid: str) -> None:
        with self._lock:
            self._push_buffers.pop(sid, None)

    def close(self) -> None:
        with self._lock:
            for j in self._jobs.values():
                j["cancel_ev"].set()
