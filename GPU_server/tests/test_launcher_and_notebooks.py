"""Unit and subprocess tests for GPU_server/launch.py and Kaggle notebook."""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import launch as gpu_launch

GPU_SERVER_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = GPU_SERVER_DIR / "notebooks" / "exhibitflow_kaggle_worker.ipynb"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_evaluate_readiness_strict_gates() -> None:
    # Loading models -> NOT ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True, "models": {"state": "loading", "msg": "downloading"}, "vlm": {"state": "stopped"}},
        demo=False,
        wait_models_flag=True,
    )
    assert ok is False
    assert any("loading" in r for r in reasons)

    # Error models -> NOT ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True, "models": {"state": "error", "msg": "missing weights"}, "vlm": {"state": "ready"}},
        demo=False,
        wait_models_flag=True,
    )
    assert ok is False
    assert any("error" in r for r in reasons)

    # VLM error -> NOT ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True, "models": {"state": "ready"}, "vlm": {"state": "error"}},
        demo=False,
        wait_models_flag=True,
    )
    assert ok is False
    assert any("vlm.state" in r for r in reasons)

    # Demo mode -> ready when ok=True
    ok, reasons = gpu_launch.evaluate_readiness({"ok": True}, demo=True, wait_models_flag=True)
    assert ok is True and reasons == []

    # Real GPU ready + verified external tunnel -> ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True, "models": {"state": "ready"}, "vlm": {"state": "ready"}},
        demo=False,
        wait_models_flag=True,
        tunnel_requested=True,
        public_url="https://demo.trycloudflare.com",
        external_ok=True,
        external_auth_ok=True,
    )
    assert ok is True and reasons == []

    # Tunnel requested but external probe failed -> NOT ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True},
        demo=True,
        tunnel_requested=True,
        public_url="https://demo.trycloudflare.com",
        external_ok=False,
        external_auth_ok=None,
    )
    assert ok is False
    assert any("External health probe failed" in r for r in reasons)

    # Tunnel requested but unauthenticated probe was not rejected with 401 -> NOT ready
    ok, reasons = gpu_launch.evaluate_readiness(
        {"ok": True},
        demo=True,
        tunnel_requested=True,
        public_url="https://demo.trycloudflare.com",
        external_ok=True,
        external_auth_ok=False,
    )
    assert ok is False
    assert any("401" in r for r in reasons)


def test_probe_external_health_checks_auth_and_401() -> None:
    expected = "probe-secret-tok"

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/internal/health":
                self.send_response(404)
                self.end_headers()
                return
            if self.headers.get("X-GPU-Token") != expected:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"detail":"Unauthorized"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        auth_ok, unauth_401, _msg = gpu_launch.probe_external_health(
            f"http://127.0.0.1:{port}",
            expected,
            timeout_s=3.0,
            interval_s=0.1,
        )
        assert auth_ok is True
        assert unauth_401 is True
    finally:
        srv.shutdown()
        srv.server_close()


def test_named_tunnel_cmd_and_vast_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = gpu_launch.build_named_tunnel_cmd("/usr/local/bin/cloudflared", "cf-tok-abc")
    assert cmd == ["/usr/local/bin/cloudflared", "tunnel", "--no-autoupdate", "run", "--token", "cf-tok-abc"]

    with pytest.raises(ValueError):
        gpu_launch.build_named_tunnel_cmd("/usr/local/bin/cloudflared", "")

    # CLI validation: --tunnel named without token/public-url exits non-zero
    res = subprocess.run(
        [sys.executable, "launch.py", "--tunnel", "named", "--demo"],
        cwd=GPU_SERVER_DIR,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert res.returncode != 0
    assert "CLOUDFLARE_TUNNEL_TOKEN" in res.stderr

    monkeypatch.setenv("PUBLIC_IPADDR", "203.0.113.10")
    monkeypatch.setenv("VAST_TCP_PORT_8001", "41234")
    assert gpu_launch.detect_vast_direct_url(8001) == "http://203.0.113.10:41234"


def test_launcher_demo_no_token_leak_in_log_file(tmp_path: Path) -> None:
    port = _free_port()
    log_file = tmp_path / "worker.log"
    secret = "sec-tok-123"
    proc = subprocess.Popen(
        [
            sys.executable,
            "launch.py",
            "--demo",
            "--no-tunnel",
            "--port",
            str(port),
            "--data-dir",
            str(tmp_path / "gpu_data"),
            "--log-file",
            str(log_file),
            "--token",
            secret,
        ],
        cwd=GPU_SERVER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        lines: list[str] = []
        ready_seen = False
        token_seen = False
        deadline = time.time() + 35.0
        while time.time() < deadline:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            lines.append(line)
            if "STATUS: READY" in line:
                ready_seen = True
            if re.search(rf"X-GPU-Token\s*:\s*{re.escape(secret)}", line):
                token_seen = True
            if ready_seen and token_seen and "Worker Log" in "".join(lines):
                break
        output = "".join(lines)
        assert ready_seen, f"STATUS: READY missing in launcher output:\n{output}"
        assert token_seen, f"X-GPU-Token missing in launcher banner:\n{output}"
        assert log_file.is_file()
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
        assert secret not in log_text, "Secret GPU_TOKEN leaked into worker log file!"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_kaggle_notebook_valid_and_documents_limits() -> None:
    assert NOTEBOOK_PATH.is_file()
    raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 4

    cell_types = {c["cell_type"] for c in nb["cells"]}
    assert "markdown" in cell_types and "code" in cell_types

    md_text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for required in ("T4 x2", "12 hour", "20-minute", "30 hours/week", "20 GB", "Vast.ai"):
        assert required in md_text, f"Notebook markdown missing documented Kaggle limit: {required}"

    # Must not contain hardcoded secrets
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", raw)
    assert not re.search(r'GPU_TOKEN\s*=\s*["\'][^"\']+["\']', raw)


def test_wait_tunnel_dns_doh_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-trycloudflare URL skips DoH and returns True immediately
    assert gpu_launch.wait_tunnel_dns("https://gpu.example.com", timeout_s=1.0) is True

    calls = {"n": 0}

    class _FakeResp:
        def __init__(self, payload: dict) -> None:
            self._data = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=3.0):
        calls["n"] += 1
        assert "1.1.1.1/dns-query" in req.full_url
        assert req.headers.get("Accept") == "application/dns-json"
        if calls["n"] == 1:
            return _FakeResp({"Status": 3, "Answer": []})
        return _FakeResp({"Status": 0, "Answer": [{"data": "104.16.230.132"}]})

    monkeypatch.setattr(gpu_launch.urllib.request, "urlopen", _fake_urlopen)
    assert gpu_launch.wait_tunnel_dns("https://abc-xyz.trycloudflare.com", timeout_s=5.0, interval_s=0.01) is True
    assert calls["n"] == 2

    # Fallback when DoH itself fails
    slept: list[float] = []
    monkeypatch.setattr(gpu_launch.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(gpu_launch.time, "sleep", lambda s: slept.append(s))
    assert gpu_launch.wait_tunnel_dns("https://abc-xyz.trycloudflare.com", timeout_s=5.0) is False
    assert slept == [3.5]

