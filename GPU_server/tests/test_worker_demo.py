"""CPU-only unit and E2E tests for GPU_server/worker in --demo mode."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worker.app import create_worker_app
from worker.demo import minimal_mp4_bytes

TOKEN = "test-gpu-secret"
HEADERS = {"X-GPU-Token": TOKEN}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("EF_DEMO_DURATION_S", "0.45")
    app = create_worker_app(data_dir=tmp_path / "gpu_data", demo=True, preload=True, token=TOKEN)
    with TestClient(app) as c:
        yield c


def _wait_job_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/internal/jobs/{job_id}", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def _wait_site_job_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/internal/site-jobs/{job_id}", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for site job {job_id}")


def test_demo_has_zero_heavy_imports(tmp_path: Path) -> None:
    import os
    import subprocess

    gpu_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "from worker.app import create_worker_app\n"
        "app = create_worker_app(data_dir=Path(sys.argv[1]), demo=True, preload=True, token='t')\n"
        "with TestClient(app) as c:\n"
        "    r = c.get('/internal/health', headers={'X-GPU-Token': 't'})\n"
        "    assert r.status_code == 200\n"
        "forbidden = ('torch', 'cv2', 'numpy', 'ultralytics', 'exhibitflow_pipeline', 'dashboard')\n"
        "found = [m for m in forbidden if m in sys.modules]\n"
        "assert not found, f'Forbidden modules imported in --demo mode: {found}'\n"
    )
    env = dict(os.environ, PYTHONPATH=gpu_root)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "gpu_data")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_auth_enforced_on_internal_routes(client: TestClient) -> None:
    for path in ("/internal/health", "/internal/system", "/internal/tuning/schema", "/internal/sites"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-GPU-Token": "wrong"}).status_code == 401
        assert client.get(path, headers=HEADERS).status_code == 200

    assert client.post("/internal/jobs", data={"options": "{}"}).status_code == 401
    assert client.post("/internal/camera/session").status_code == 401


def test_health_system_schema_and_sites(client: TestClient) -> None:
    h = client.get("/internal/health", headers=HEADERS).json()
    assert h["ok"] is True
    assert h["version"] == "ef-gpu-1"
    assert h["demo"] is True
    assert h["models"]["state"] == "ready"
    assert h["vlm"]["shared"] is True
    assert h["busy"] is False
    assert h["active_job_id"] is None
    assert h["capabilities"] == {
        "single_cam": True,
        "multi_cam": True,
        "site_calibrated": True,
        "floorplan": True,
    }

    sys_info = client.get("/internal/system", headers=HEADERS).json()
    assert "cores" in sys_info and sys_info["cores"]["logical"] >= 1
    assert isinstance(sys_info["gpus"], list)

    schema = client.get("/internal/tuning/schema", headers=HEADERS).json()
    assert "groups" in schema and "params" in schema and len(schema["params"]) > 0

    sites = client.get("/internal/sites", headers=HEADERS).json()
    demo_site = next((s for s in sites if s["site_id"] == "demo_gallery"), None)
    assert demo_site is not None
    assert demo_site["cameras"] == ["C1", "C2"]
    assert demo_site["labels"] == {"C1": "CAM 1", "C2": "CAM 2"}


def test_single_job_e2e_live_artifacts_and_traversal(client: TestClient) -> None:
    mp4 = minimal_mp4_bytes()
    r = client.post(
        "/internal/jobs",
        headers=HEADERS,
        files={"video": ("gallery_clip.mp4", mp4, "video/mp4")},
        data={"options": json.dumps({"demo_duration_s": 0.6})},
    )
    assert r.status_code == 200
    created = r.json()
    job_id = created["job_id"]
    assert created["status"] == "running"

    # Second concurrent job must be rejected with 409 while busy
    r_busy = client.post(
        "/internal/jobs",
        headers=HEADERS,
        files={"video": ("other.mp4", mp4, "video/mp4")},
        data={"options": "{}"},
    )
    assert r_busy.status_code == 409

    # Wait briefly for first live frame
    live_ok = False
    for _ in range(30):
        rl = client.get(f"/internal/jobs/{job_id}/live", headers=HEADERS)
        if rl.status_code == 200:
            assert rl.headers["content-type"].startswith("image/jpeg")
            assert rl.content.startswith(b"\xff\xd8")
            live_ok = True
            break
        time.sleep(0.03)
    assert live_ok, "Never received live JPEG frame"

    # MJPEG stream returns valid multipart frame(s)
    with client.stream("GET", f"/internal/jobs/{job_id}/live/stream", headers=HEADERS) as stream_resp:
        assert stream_resp.status_code == 200
        assert "multipart/x-mixed-replace" in stream_resp.headers["content-type"]
        collected = b""
        for chunk in stream_resp.iter_bytes():
            collected += chunk
            if b"--frame" in collected and b"\xff\xd8" in collected and b"\xff\xd9" in collected:
                break
        assert b"--frame" in collected and b"\xff\xd8" in collected

    # Map image & map data while/after running
    rm = client.get(f"/internal/jobs/{job_id}/map", headers=HEADERS)
    assert rm.status_code == 200
    assert rm.content.startswith(b"\x89PNG")

    final = _wait_job_done(client, job_id)
    assert final["status"] == "done"
    assert final["progress"] == 1.0
    key = final["key"]
    assert key == "gallery_clip"

    # Live events
    revs = client.get(f"/internal/jobs/{job_id}/live/events?after=0", headers=HEADERS)
    assert revs.status_code == 200
    ev_payload = revs.json()
    assert ev_payload["job"] == "done"
    assert len(ev_payload["events"]) >= 3
    kinds = {e["kind"] for e in ev_payload["events"]}
    assert "contact" in kinds and "vlm" in kinds

    # Full map data
    md = client.get(f"/internal/jobs/{job_id}/map/data?full=1", headers=HEADERS).json()
    assert md["has_map"] is True
    assert "frames" in md and len(md["frames"]) >= 10

    # Artifacts listing + downloads
    art_list = client.get(f"/internal/jobs/{job_id}/artifacts", headers=HEADERS).json()
    rels = {f["rel_path"] for f in art_list["files"]}
    assert "result.json" in rels
    assert "live.mp4" in rels
    assert "map.png" in rels
    assert "map_data.json" in rels
    assert f"files/{key}/vlm_inputs/t001/sheet.jpg" in rels

    res_json = client.get(f"/internal/jobs/{job_id}/artifacts/result.json", headers=HEADERS).json()
    assert res_json["summary"]["has_anomaly"] is True
    decisions = {e["decision"] for e in res_json["events"]}
    assert {"ANOMALY", "CANDIDATE", "NORMAL"}.issubset(decisions)

    sheet = client.get(
        f"/internal/jobs/{job_id}/artifacts/files/{key}/vlm_inputs/t001/sheet.jpg",
        headers=HEADERS,
    )
    assert sheet.status_code == 200
    assert sheet.content.startswith(b"\xff\xd8")

    # Path traversal rejection
    for bad in ("..%2f..%2fetc/passwd", "files/../../secret.txt", "../result.json"):
        r_bad = client.get(f"/internal/jobs/{job_id}/artifacts/{bad}", headers=HEADERS)
        assert r_bad.status_code in (400, 404)


def test_cancel_job(client: TestClient) -> None:
    r = client.post(
        "/internal/jobs",
        headers=HEADERS,
        data={"source": "demo", "options": json.dumps({"demo_duration_s": 4.0})},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    rc = client.post(f"/internal/jobs/{job_id}/cancel", headers=HEADERS)
    assert rc.status_code == 200
    assert rc.json()["status"] in ("cancelling", "cancelled")

    final = _wait_job_done(client, job_id)
    assert final["status"] == "cancelled"


def test_calibrated_site_job_e2e(client: TestClient) -> None:
    mp4 = minimal_mp4_bytes()
    r = client.post(
        "/internal/site-jobs",
        headers=HEADERS,
        data={"site": "demo_gallery", "events": "1"},
        files={
            "C1": ("cam1.mp4", mp4, "video/mp4"),
            "C2": ("cam2.mp4", mp4, "video/mp4"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    job_id = body["job_id"]
    assert body["cameras"] == ["C1", "C2"]
    assert body["events"] is True

    final = _wait_site_job_done(client, job_id)
    assert final["status"] == "done"

    arts = client.get(f"/internal/site-jobs/{job_id}/artifacts", headers=HEADERS).json()
    rels = {f["rel_path"] for f in arts["files"]}
    assert "site_result.json" in rels
    assert "global_2d_map.png" in rels
    assert "videos/C1.mp4" in rels
    assert "videos/C2.mp4" in rels

    sr = client.get(f"/internal/site-jobs/{job_id}/artifacts/site_result.json", headers=HEADERS).json()
    assert sr["map_source"] == "demo"
    assert len(sr["cameras"]) == 2
    assert len(sr["frames"]) > 0
    assert len(sr["events"]) >= 1

    png = client.get(f"/internal/site-jobs/{job_id}/artifacts/global_2d_map.png", headers=HEADERS)
    assert png.status_code == 200
    assert png.content.startswith(b"\x89PNG")

    assert client.get(f"/internal/site-jobs/{job_id}/artifacts/..%2fetc/passwd", headers=HEADERS).status_code in (400, 404)


def test_camera_session_ws_auth_and_push(client: TestClient) -> None:
    sess = client.post("/internal/camera/session", headers=HEADERS).json()
    sid = sess["sid"]

    # Rejected without token
    with pytest.raises(Exception):
        with client.websocket_connect(f"/internal/camera/{sid}/push"):
            pass

    # Accepted with ?token=
    with client.websocket_connect(f"/internal/camera/{sid}/push?token={TOKEN}") as ws:
        ws.send_bytes(b"\xff\xd8\xff\xd9")


def test_multi_job_e2e(client: TestClient) -> None:
    mp4 = minimal_mp4_bytes()
    r = client.post(
        "/internal/multi-jobs",
        headers=HEADERS,
        data={
            "options": json.dumps({"demo_duration_s": 0.2}),
            "cameras": json.dumps([{"name": "CamA"}, {"name": "CamB"}]),
        },
        files={
            "video_0": ("a.mp4", mp4, "video/mp4"),
            "video_1": ("b.mp4", mp4, "video/mp4"),
        },
    )
    assert r.status_code == 200
    mid = r.json()["multi_id"]

    deadline = time.time() + 10.0
    snap = None
    while time.time() < deadline:
        snap = client.get(f"/internal/multi-jobs/{mid}", headers=HEADERS).json()
        if snap["status"] == "done":
            break
        time.sleep(0.05)
    assert snap is not None and snap["status"] == "done"

    results = client.get(f"/internal/multi-jobs/{mid}/results", headers=HEADERS).json()
    assert len(results) == 2
