"""Tests for Prometheus /metrics endpoint, HTTP middleware, and auth controls."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def test_metrics_exposes_prometheus_format_and_gauges(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path / "be_data", gpu_worker_url="", backend_api_key="")
    app = create_app(cfg)
    with TestClient(app) as client:
        store = app.state.store
        now = time.time()
        store.create_job(
            {
                "job_id": "j_done",
                "job_type": "single",
                "status": "done",
                "name": "cam1.mp4",
                "created_at": now,
            }
        )
        store.save_alerts(
            "j_done",
            [
                {
                    "event_id": "e1",
                    "decision": "ANOMALY",
                    "behavior": "touching",
                    "confidence": 0.9,
                }
            ],
        )
        store.set_setting(
            "gpu_system",
            {"gpus": [{"name": "RTX 4090", "util": 42.5, "mem_used_gb": 8.2, "mem_total_gb": 24.0}]},
        )

        assert client.get("/api/health").status_code == 200
        assert client.get("/api/jobs").status_code == 200

        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        body = r.text
        assert "# TYPE exhibitflow_http_requests_total counter" in body
        assert 'path="/api/health",status="200"' in body
        assert 'exhibitflow_http_request_duration_seconds_bucket{method="GET",path="/api/jobs",le="+Inf"}' in body
        assert 'exhibitflow_jobs_total{status="done"} 1' in body
        assert 'exhibitflow_alerts_total{decision="ANOMALY"} 1' in body
        assert "exhibitflow_backend_cpu_percent " in body
        assert "exhibitflow_backend_ram_used_bytes " in body
        assert 'exhibitflow_gpu_vram_used_gb{gpu_index="0",gpu_name="RTX 4090"} 8.200' in body


def test_metrics_respects_backend_auth_and_metrics_public_flag(tmp_path: Path) -> None:
    cfg_priv = Settings(
        data_dir=tmp_path / "be_priv",
        gpu_worker_url="",
        backend_api_key="be-secret",
        metrics_public=False,
    )
    with TestClient(create_app(cfg_priv)) as c_priv:
        assert c_priv.get("/metrics").status_code == 401
        ok = c_priv.get("/metrics", headers={"Authorization": "Bearer be-secret"})
        assert ok.status_code == 200
        assert "exhibitflow_queue_depth" in ok.text

    cfg_pub = Settings(
        data_dir=tmp_path / "be_pub",
        gpu_worker_url="",
        backend_api_key="be-secret",
        metrics_public=True,
    )
    with TestClient(create_app(cfg_pub)) as c_pub:
        assert c_pub.get("/metrics").status_code == 200
