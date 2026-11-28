"""Authentication, stream-ticket, gpu.ready, and CORS security tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.gpu_client import compute_gpu_ready
from backend.app.main import create_app


def _make_secured_app(
    tmp_path: Path,
    *,
    api_key: str = "be-secret-123",
    gpu_token: str = "gpu-secret-999",
    cors_origins: str = "https://exhibitflow.vercel.app",
    ttl_s: int = 600,
):
    cfg = Settings(
        data_dir=tmp_path / "be_data",
        backend_api_key=api_key,
        gpu_token=gpu_token,
        gpu_worker_url="",
        cors_origins=cors_origins,
        stream_token_ttl_s=ttl_s,
    )
    app = create_app(cfg)
    return app


def _seed_synced_jobs_and_alerts(app) -> tuple[str, str, str]:
    store = app.state.store
    artifacts = app.state.artifacts
    now = time.time()

    # Single-cam synced job
    jid = "job_sec_1"
    store.create_job(
        {
            "job_id": jid,
            "job_type": "single",
            "status": "done",
            "name": "sec_cam.mp4",
            "key": "sec_cam",
            "artifacts_synced": True,
            "created_at": now,
            "finished_at": now + 1,
        }
    )
    artifacts.put_bytes(
        f"jobs/{jid}/result.json",
        json.dumps({"video": {"key": "sec_cam"}, "events": []}).encode("utf-8"),
        content_type="application/json",
    )
    artifacts.put_bytes(f"jobs/{jid}/live.mp4", b"0000ftypisom" + b"a" * 64, content_type="video/mp4")
    artifacts.put_bytes(f"jobs/{jid}/map.png", b"\x89PNG\r\n\x1a\n", content_type="image/png")
    artifacts.put_bytes(
        f"jobs/{jid}/map_data.json",
        b'{"has_map":true,"version":1,"fps":5.0,"people":[],"artifacts":[]}',
        content_type="application/json",
    )
    artifacts.put_bytes(
        f"jobs/{jid}/files/sec_cam/vlm_inputs/t1/sheet.jpg",
        b"\xff\xd8\xff\xd9",
        content_type="image/jpeg",
    )
    alerts = store.save_alerts(
        jid,
        [
            {
                "event_id": "e1",
                "job_name": "sec_cam.mp4",
                "track_id": 1,
                "artifact_id": 1,
                "decision": "ANOMALY",
                "behavior": "touching",
                "confidence": 0.9,
                "evidence": "Hand on exhibit",
                "start_s": 1.0,
                "end_s": 3.0,
            }
        ],
    )
    aid = alerts[0]["alert_id"]

    # Site synced job
    sjid = "site_sec_1"
    store.create_job(
        {
            "job_id": sjid,
            "job_type": "site",
            "status": "done",
            "name": "Site Demo",
            "site_id": "demo_gallery",
            "cameras": ["C1"],
            "artifacts_synced": True,
            "created_at": now,
            "finished_at": now + 1,
        }
    )
    artifacts.put_bytes(
        f"site_jobs/{sjid}/site_result.json",
        b'{"bounds":[-4,-3,4,3],"cameras":[],"frames":[],"events":[]}',
        content_type="application/json",
    )
    artifacts.put_bytes(
        f"site_jobs/{sjid}/global_2d_map.png",
        b"\x89PNG\r\n\x1a\n",
        content_type="image/png",
    )
    artifacts.put_bytes(
        f"site_jobs/{sjid}/videos/C1.mp4",
        b"0000ftypisom" + b"b" * 64,
        content_type="video/mp4",
    )
    return jid, sjid, aid


def test_compute_gpu_ready_contract() -> None:
    assert compute_gpu_ready({"online": False, "demo": True}) is False
    assert compute_gpu_ready({"online": True, "demo": True, "models": {"state": "loading"}}) is True
    assert compute_gpu_ready({"online": True, "demo": False, "models": {"state": "loading"}, "vlm": {"state": "ready"}}) is False
    assert compute_gpu_ready({"online": True, "demo": False, "models": {"state": "ready"}, "vlm": {"state": "error"}}) is False
    assert compute_gpu_ready({"online": True, "demo": False, "models": {"state": "ready"}, "vlm": {"state": "ready"}}) is True
    assert compute_gpu_ready({"online": True, "demo": False, "models": {"state": "ready"}, "vlm": {"state": "disabled"}}) is True


def test_health_privacy_and_gpu_token_isolation(tmp_path: Path) -> None:
    app = _make_secured_app(tmp_path)
    with TestClient(app) as client:
        # Unauthenticated /api/health hides gpu and queue_length
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["auth_required"] is True
        assert body["authenticated"] is False
        assert "gpu" not in body
        assert "queue_length" not in body

        # X-GPU-Token must NEVER authenticate against backend
        r_gpu_hdr = client.get("/api/health", headers={"X-GPU-Token": "gpu-secret-999"})
        assert r_gpu_hdr.json()["authenticated"] is False
        r_gpu_bearer = client.get(
            "/api/health", headers={"Authorization": "Bearer gpu-secret-999"}
        )
        assert r_gpu_bearer.json()["authenticated"] is False

        # Valid Backend key via Bearer and X-Backend-Key
        for hdrs in (
            {"Authorization": "Bearer be-secret-123"},
            {"X-Backend-Key": "be-secret-123"},
        ):
            authed = client.get("/api/health", headers=hdrs)
            assert authed.status_code == 200
            data = authed.json()
            assert data["authenticated"] is True
            assert data["auth_required"] is True
            assert "gpu" in data
            assert "ready" in data["gpu"]
            assert "queue_length" in data


def test_401_enforcement_and_stream_ticket_scopes(tmp_path: Path) -> None:
    app = _make_secured_app(tmp_path)
    jid, sjid, aid = _seed_synced_jobs_and_alerts(app)
    hdr = {"Authorization": "Bearer be-secret-123"}

    with TestClient(app) as client:
        # Sensitive header-only & media endpoints reject missing auth and X-GPU-Token
        sensitive_calls = [
            ("POST", "/api/auth/verify"),
            ("POST", "/api/auth/stream-token"),
            ("GET", "/api/system"),
            ("GET", "/api/config/gpu"),
            ("PUT", "/api/config/gpu"),
            ("GET", "/api/tuning/schema"),
            ("GET", "/api/sites"),
            ("POST", "/api/jobs"),
            ("GET", "/api/jobs"),
            ("GET", f"/api/jobs/{jid}"),
            ("DELETE", f"/api/jobs/{jid}"),
            ("POST", f"/api/jobs/{jid}/cancel"),
            ("POST", f"/api/jobs/{jid}/finish"),
            ("GET", f"/api/jobs/{jid}/live/events"),
            ("GET", f"/api/jobs/{jid}/result"),
            ("GET", f"/api/jobs/{jid}/map/data"),
            ("GET", "/api/alerts"),
            ("PATCH", f"/api/alerts/{aid}"),
            ("POST", "/api/site-jobs"),
            ("GET", f"/api/site-jobs/{sjid}"),
            ("GET", f"/api/site-jobs/{sjid}/result"),
            ("GET", "/api/events"),
            ("GET", f"/api/jobs/{jid}/events"),
            ("GET", f"/api/jobs/{jid}/live"),
            ("GET", f"/api/jobs/{jid}/live/stream"),
            ("GET", f"/api/jobs/{jid}/video"),
            ("GET", f"/api/jobs/{jid}/map"),
            ("GET", f"/api/jobs/{jid}/files/sec_cam/vlm_inputs/t1/sheet.jpg"),
            ("GET", f"/api/site-jobs/{sjid}/map"),
            ("GET", f"/api/site-jobs/{sjid}/video/C1"),
            ("GET", "/api/alerts/export.csv"),
        ]
        for method, path in sensitive_calls:
            r1 = client.request(method, path)
            assert r1.status_code == 401, f"{method} {path} expected 401, got {r1.status_code}"
            r2 = client.request(method, path, headers={"X-GPU-Token": "gpu-secret-999"})
            assert r2.status_code == 401, f"{method} {path} accepted GPU token!"

        # Verify /api/auth/verify & /api/auth/stream-token
        vr = client.post("/api/auth/verify", headers=hdr)
        assert vr.status_code == 200
        assert vr.json() == {"ok": True, "auth_required": True, "authenticated": True}

        tr = client.post("/api/auth/stream-token", headers=hdr)
        assert tr.status_code == 200
        tjson = tr.json()
        ticket = tjson["ticket"]
        assert "." in ticket and tjson["auth_required"] is True
        assert tjson["expires_in"] == 600

        # Stream ticket WORKS on media/SSE/CSV endpoints
        assert client.get(f"/api/jobs/{jid}/video?st={ticket}").status_code == 200
        assert client.get(f"/api/jobs/{jid}/map?st={ticket}").status_code == 200
        assert (
            client.get(
                f"/api/jobs/{jid}/files/sec_cam/vlm_inputs/t1/sheet.jpg?st={ticket}"
            ).status_code
            == 200
        )
        assert client.get(f"/api/site-jobs/{sjid}/map?st={ticket}").status_code == 200
        assert client.get(f"/api/site-jobs/{sjid}/video/C1?st={ticket}").status_code == 200
        assert client.get(f"/api/alerts/export.csv?st={ticket}").status_code == 200

        # SSE endpoints accept ?st= ticket
        with client.stream("GET", f"/api/jobs/{jid}/events?st={ticket}") as sresp:
            assert sresp.status_code == 200
            lines = []
            for line in sresp.iter_lines():
                if line:
                    lines.append(line)
                if any("event: done" in l for l in lines):
                    break
            assert any("event: status" in l for l in lines)

        with client.stream("GET", f"/api/events?st={ticket}") as gresp:
            assert gresp.status_code == 200
            first_chunk = next(gresp.iter_lines())
            assert first_chunk is not None

        # Stream ticket MUST BE REJECTED (401) on header-only endpoints!
        header_only_with_st = [
            ("POST", f"/api/jobs?st={ticket}"),
            ("DELETE", f"/api/jobs/{jid}?st={ticket}"),
            ("GET", f"/api/config/gpu?st={ticket}"),
            ("PUT", f"/api/config/gpu?st={ticket}"),
            ("PATCH", f"/api/alerts/{aid}?st={ticket}"),
            ("POST", f"/api/site-jobs?st={ticket}"),
            ("GET", f"/api/alerts?st={ticket}"),
            ("GET", f"/api/jobs/{jid}/result?st={ticket}"),
        ]
        for method, path in header_only_with_st:
            r = client.request(method, path)
            assert r.status_code == 401, f"{method} {path} should reject ?st= with 401"

        # Tampered or expired ?st= ticket MUST BE REJECTED (401)
        exp_part, sig_part = ticket.split(".", 1)
        tampered = f"{exp_part}.{'0' * len(sig_part)}"
        assert client.get(f"/api/jobs/{jid}/video?st={tampered}").status_code == 401
        assert client.get(f"/api/jobs/{jid}/video?st=1.{sig_part}").status_code == 401
        assert client.get(f"/api/jobs/{jid}/video?st=not-a-ticket").status_code == 401


def test_cors_preflight_for_vercel_origin(tmp_path: Path) -> None:
    app = _make_secured_app(tmp_path, cors_origins="https://exhibitflow.vercel.app")
    with TestClient(app) as client:
        r = client.options(
            "/api/jobs",
            headers={
                "Origin": "https://exhibitflow.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, X-Backend-Key, Content-Type",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://exhibitflow.vercel.app"
        allow_hdrs = (r.headers.get("access-control-allow-headers") or "").lower()
        assert "authorization" in allow_hdrs
        assert "x-backend-key" in allow_hdrs

        actual = client.get(
            "/api/health",
            headers={"Origin": "https://exhibitflow.vercel.app"},
        )
        assert actual.status_code == 200
        assert actual.headers.get("access-control-allow-origin") == "https://exhibitflow.vercel.app"
        expose_hdrs = (actual.headers.get("access-control-expose-headers") or "").lower()
        assert "content-range" in expose_hdrs
        assert "content-disposition" in expose_hdrs
