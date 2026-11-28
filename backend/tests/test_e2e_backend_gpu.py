"""End-to-end integration test: real GPU_server (--demo) + real CPU backend + offline replay."""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _minimal_mp4_bytes() -> bytes:
    ftyp_body = b"isom" + struct.pack(">I", 512) + b"isomiso2mp41"
    ftyp = struct.pack(">I", 8 + len(ftyp_body)) + b"ftyp" + ftyp_body
    mvhd_body = struct.pack(">IIII", 0, 0, 0, 1000) + struct.pack(">IIHH", 1000, 0x00010000, 0x0100, 0)
    mvhd_body += b"\x00" * 8
    mvhd_body += struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    mvhd_body += b"\x00" * 24 + struct.pack(">I", 2)
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    return ftyp + moov + (b"\x00" * 128)


def _wait_http(
    url: str,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    proc: subprocess.Popen | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
            raise AssertionError(f"Process exited early ({proc.returncode}) at {url}:\n{out}")
        try:
            r = httpx.get(url, headers=headers or {}, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.15)
    out = ""
    if proc is not None and proc.poll() is not None and proc.stdout:
        out = proc.stdout.read().decode("utf-8", errors="replace")
    raise AssertionError(f"Service did not become ready at {url}: {last_err}\n{out}")


def _wait_job_done(base: str, job_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = httpx.get(f"{base}/api/jobs/{job_id}", timeout=5.0)
        assert r.status_code == 200
        data = r.json()
        if data["status"] == "done":
            return data
        if data["status"] in ("error", "cancelled"):
            raise AssertionError(f"Job {job_id} failed: {data}")
        time.sleep(0.15)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def test_e2e_backend_gpu_queue_streaming_and_offline_replay(tmp_path: Path) -> None:
    gpu_port = _free_port()
    be_port = _free_port()
    gpu_url = f"http://127.0.0.1:{gpu_port}"
    be_url = f"http://127.0.0.1:{be_port}"

    gpu_env = os.environ.copy()
    gpu_env["PYTHONPATH"] = str(REPO_ROOT / "GPU_server")
    gpu_env["EF_DEMO_DURATION_S"] = "1.2"

    be_env = os.environ.copy()
    be_env["PYTHONPATH"] = str(REPO_ROOT)
    be_env["DATA_DIR"] = str(tmp_path / "backend_data")
    be_env["GPU_WORKER_URL"] = gpu_url
    be_env["GPU_TOKEN"] = "test-secret"
    be_env["POLL_INTERVAL_S"] = "0.15"
    be_env["HEALTH_INTERVAL_S"] = "0.3"
    be_env["ARTIFACT_STORE"] = "local"
    be_env["REDIS_URL"] = ""

    gpu_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "worker.app",
            "--host",
            "127.0.0.1",
            "--port",
            str(gpu_port),
            "--data-dir",
            str(tmp_path / "gpu_data"),
            "--demo",
            "--token",
            "test-secret",
        ],
        cwd=str(REPO_ROOT / "GPU_server"),
        env=gpu_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    be_proc = None
    try:
        _wait_http(f"{gpu_url}/internal/health", headers={"X-GPU-Token": "test-secret"}, proc=gpu_proc)

        # 1. Verify X-GPU-Token enforcement on GPU_server
        r_no_auth = httpx.get(f"{gpu_url}/internal/health", timeout=3.0)
        assert r_no_auth.status_code == 401
        r_bad_auth = httpx.get(
            f"{gpu_url}/internal/health", headers={"X-GPU-Token": "wrong"}, timeout=3.0
        )
        assert r_bad_auth.status_code == 401

        be_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(be_port),
                "--log-level",
                "warning",
            ],
            cwd=str(REPO_ROOT),
            env=be_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_http(f"{be_url}/api/health", proc=be_proc)

        # 2. Check /api/health, /api/config/gpu (token masked), /api/sites, /api/tuning/schema
        h = httpx.get(f"{be_url}/api/health", timeout=5.0).json()
        assert h["ok"] is True
        assert h["gpu"]["online"] is True
        assert h["gpu"]["demo"] is True

        cfg_r = httpx.get(f"{be_url}/api/config/gpu", timeout=5.0).json()
        assert cfg_r == {"url": gpu_url, "has_token": True}
        assert "test-secret" not in str(cfg_r)

        sites = httpx.get(f"{be_url}/api/sites", timeout=5.0).json()
        assert any(s["site_id"] == "demo_gallery" for s in sites)
        schema = httpx.get(f"{be_url}/api/tuning/schema", timeout=5.0).json()
        assert isinstance(schema, dict)

        # Live detector config proxy (/api/config -> /internal/config)
        live_cfg = httpx.post(f"{be_url}/api/config", json={"det_conf": 0.33}, timeout=5.0)
        assert live_cfg.status_code == 200
        assert live_cfg.json()["applied_now"] is True
        assert live_cfg.json()["det_conf_effective"] == 0.33
        assert httpx.post(f"{be_url}/api/config", json={"det_conf": None}, timeout=5.0).json()["det_conf_effective"] == 0.01

        # 3. Enqueue 2 single-cam jobs back-to-back without 409
        mp4 = _minimal_mp4_bytes()
        r1 = httpx.post(
            f"{be_url}/api/jobs",
            files={"video": ("cam_north.mp4", mp4, "video/mp4")},
            data={"options": '{"demo_duration_s": 1.2, "realtime": true}'},
            timeout=10.0,
        )
        assert r1.status_code == 200
        j1 = r1.json()
        assert j1["status"] == "queued"

        r2 = httpx.post(
            f"{be_url}/api/jobs",
            files={"video": ("cam_south.mp4", mp4, "video/mp4")},
            data={"options": '{"demo_duration_s": 0.6}'},
            timeout=10.0,
        )
        assert r2.status_code == 200
        j2 = r2.json()
        assert j2["status"] == "queued"
        jid1, jid2 = j1["job_id"], j2["job_id"]

        # 4. Stream MJPEG + per-job SSE from job 1 while running
        seen_mjpeg = b""
        with httpx.stream("GET", f"{be_url}/api/jobs/{jid1}/live/stream", timeout=10.0) as resp:
            assert resp.status_code == 200
            for chunk in resp.iter_bytes():
                seen_mjpeg += chunk
                if b"--frame" in seen_mjpeg and b"\xff\xd8" in seen_mjpeg:
                    break
        assert b"--frame" in seen_mjpeg and b"\xff\xd8" in seen_mjpeg

        sse_lines: list[str] = []
        with httpx.stream("GET", f"{be_url}/api/jobs/{jid1}/events", timeout=15.0) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line:
                    sse_lines.append(line)
                if "event: done" in sse_lines:
                    break
        assert any("event: status" in l for l in sse_lines)
        assert any("event: done" in l for l in sse_lines)

        done1 = _wait_job_done(be_url, jid1)
        done2 = _wait_job_done(be_url, jid2)
        assert done1["artifacts_synced"] is True
        assert done2["artifacts_synced"] is True
        assert done1["finished_at"] <= done2["finished_at"]

        # 5. Submit calibrated site-job (demo_gallery)
        sr = httpx.post(
            f"{be_url}/api/site-jobs",
            data={"site": "demo_gallery", "events": "1"},
            files={
                "C1": ("c1.mp4", mp4, "video/mp4"),
                "C2": ("c2.mp4", mp4, "video/mp4"),
            },
            timeout=10.0,
        )
        assert sr.status_code == 200
        site_jid = sr.json()["job_id"]
        done_site = _wait_job_done(be_url, site_jid, timeout_s=25.0)
        assert done_site["artifacts_synced"] is True
        assert done_site["job_type"] == "site"

        # 6. SHUT DOWN GPU_server completely!
        gpu_proc.terminate()
        gpu_proc.wait(timeout=10)

        # Wait for backend health monitor to detect GPU offline
        deadline = time.monotonic() + 5.0
        offline_health = None
        while time.monotonic() < deadline:
            offline_health = httpx.get(f"{be_url}/api/health", timeout=5.0).json()
            if offline_health["gpu"]["online"] is False:
                break
            time.sleep(0.2)
        assert offline_health is not None
        assert offline_health["ok"] is True
        assert offline_health["gpu"]["online"] is False

        # 7. Verify full offline replay with GPU_server dead!
        jobs_list = httpx.get(f"{be_url}/api/jobs", timeout=5.0).json()
        assert jobs_list["total"] >= 3

        res1 = httpx.get(f"{be_url}/api/jobs/{jid1}/result", timeout=5.0)
        assert res1.status_code == 200
        res1_json = res1.json()
        assert len(res1_json["events"]) > 0

        # Video full + HTTP Range 206
        vid_full = httpx.get(f"{be_url}/api/jobs/{jid1}/video", timeout=5.0)
        assert vid_full.status_code == 200
        vid_range = httpx.get(
            f"{be_url}/api/jobs/{jid1}/video",
            headers={"Range": "bytes=0-63"},
            timeout=5.0,
        )
        assert vid_range.status_code == 206
        assert vid_range.headers.get("content-range", "").startswith("bytes 0-63/")
        assert len(vid_range.content) == 64

        # Map + full map data
        mp = httpx.get(f"{be_url}/api/jobs/{jid1}/map", timeout=5.0)
        assert mp.status_code == 200 and mp.content.startswith(b"\x89PNG")

        mdata = httpx.get(f"{be_url}/api/jobs/{jid1}/map/data?full=1", timeout=5.0).json()
        assert mdata["has_map"] is True
        assert len(mdata.get("frames") or {}) > 0

        # Evidence JPEG via /api/jobs/{id}/files/...
        key = res1_json["video"]["key"]
        task_ev = next(e for e in res1_json["events"] if e.get("task"))
        sheet = httpx.get(
            f"{be_url}/api/jobs/{jid1}/files/{key}/vlm_inputs/{task_ev['task']}/sheet.jpg",
            timeout=5.0,
        )
        assert sheet.status_code == 200 and sheet.content.startswith(b"\xff\xd8")

        # Alerts query, PATCH review, CSV export
        alerts_resp = httpx.get(f"{be_url}/api/alerts?job_id={jid1}", timeout=5.0)
        assert alerts_resp.status_code == 200
        al_data = alerts_resp.json()
        assert al_data["total"] == len(res1_json["events"])
        first_alert = al_data["items"][0]
        if first_alert.get("img_url"):
            img_check = httpx.get(f"{be_url}{first_alert['img_url']}", timeout=5.0)
            assert img_check.status_code == 200

        aid = first_alert["alert_id"]
        patch_r = httpx.patch(
            f"{be_url}/api/alerts/{aid}", json={"review": "ok"}, timeout=5.0
        )
        assert patch_r.status_code == 200
        assert patch_r.json()["review"] == "ok"

        ok_alerts = httpx.get(f"{be_url}/api/alerts?review=ok", timeout=5.0).json()
        assert any(a["alert_id"] == aid for a in ok_alerts["items"])

        csv_r = httpx.get(f"{be_url}/api/alerts/export.csv", timeout=5.0)
        assert csv_r.status_code == 200
        assert "text/csv" in csv_r.headers.get("content-type", "")
        assert aid in csv_r.text and ",ok," in csv_r.text

        # Calibrated site-job offline replay
        sres = httpx.get(f"{be_url}/api/site-jobs/{site_jid}/result", timeout=5.0)
        assert sres.status_code == 200
        sres_json = sres.json()
        assert "bounds" in sres_json and "cameras" in sres_json and "frames" in sres_json

        smap = httpx.get(f"{be_url}/api/site-jobs/{site_jid}/map", timeout=5.0)
        assert smap.status_code == 200 and smap.content.startswith(b"\x89PNG")

        svid = httpx.get(
            f"{be_url}/api/site-jobs/{site_jid}/video/C1",
            headers={"Range": "bytes=0-31"},
            timeout=5.0,
        )
        assert svid.status_code == 206

        scam_res = httpx.get(f"{be_url}/api/site-jobs/{site_jid}/result/C1", timeout=5.0)
        assert scam_res.status_code == 200
        assert scam_res.json()["artifact_global"] is True
        scam_sheet = httpx.get(
            f"{be_url}/api/site-jobs/{site_jid}/files/C1/vlm_inputs/t001/sheet.jpg",
            timeout=5.0,
        )
        assert scam_sheet.status_code == 200 and scam_sheet.content.startswith(b"\xff\xd8")

        # Detector config proxy (/api/config -> /internal/config while GPU is offline too)
        cfg_resp = httpx.post(f"{be_url}/api/config", json={"det_conf": 0.22}, timeout=5.0)
        assert cfg_resp.status_code == 200
        assert cfg_resp.json()["det_conf_effective"] == 0.22
        assert httpx.get(f"{be_url}/api/health", timeout=5.0).json()["det_conf_effective"] == 0.22
        assert httpx.post(f"{be_url}/api/config", json={"det_conf": None}, timeout=5.0).json()["det_conf_effective"] == 0.01

        # Delete job 2
        del_r = httpx.delete(f"{be_url}/api/jobs/{jid2}", timeout=5.0)
        assert del_r.status_code == 200
        assert httpx.get(f"{be_url}/api/jobs/{jid2}", timeout=5.0).status_code == 404
    finally:
        if gpu_proc.poll() is None:
            gpu_proc.terminate()
            gpu_proc.wait(timeout=5)
        if be_proc is not None and be_proc.poll() is None:
            be_proc.terminate()
            be_proc.wait(timeout=5)
