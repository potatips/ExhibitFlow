"""Unit and subprocess tests for backend/launch.py."""

from __future__ import annotations

import io
import os
import re
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
import pytest

from backend import launch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_resolve_helpers_and_named_tunnel_validation() -> None:
    assert launch.resolve_tunnel_mode("quick", False) == "quick"
    assert launch.resolve_tunnel_mode("quick", True) == "none"
    assert launch.resolve_tunnel_mode("named", False) == "named"
    with pytest.raises(ValueError):
        launch.resolve_tunnel_mode("invalid", False)

    # Named tunnel requires both token and https/http public URL
    with pytest.raises(ValueError):
        launch.build_tunnel_command("named", 8000, "cloudflared", tunnel_token="", public_url="https://api.example.com")
    with pytest.raises(ValueError):
        launch.build_tunnel_command("named", 8000, "cloudflared", tunnel_token="tok-abc", public_url="")
    with pytest.raises(ValueError):
        launch.build_tunnel_command("named", 8000, "cloudflared", tunnel_token="tok-abc", public_url="not-a-url")

    cmd_named = launch.build_tunnel_command(
        "named",
        8000,
        "/usr/bin/cloudflared",
        tunnel_token="tok-abc",
        public_url="https://api.example.com",
    )
    assert cmd_named == [
        "/usr/bin/cloudflared",
        "tunnel",
        "--no-autoupdate",
        "run",
        "--token",
        "tok-abc",
    ]

    cmd_quick = launch.build_tunnel_command("quick", 8010, "/usr/bin/cloudflared")
    assert cmd_quick == [
        "/usr/bin/cloudflared",
        "tunnel",
        "--no-autoupdate",
        "--url",
        "http://127.0.0.1:8010",
    ]


def test_resolve_api_key_behavior() -> None:
    assert launch.resolve_api_key("my-key", False, {}) == ("my-key", False)
    assert launch.resolve_api_key("my-key", True, {}) == ("", False)
    assert launch.resolve_api_key(None, False, {"BACKEND_API_KEY": "env-key"}) == ("env-key", False)
    gen, auto = launch.resolve_api_key(None, False, {})
    assert auto is True and len(gen) >= 20


def test_macos_tgz_extraction(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = b"#!/bin/sh\necho cloudflared\n"
        info = tarfile.TarInfo(name="cloudflared")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    dst = tmp_path / "bin" / "cloudflared"
    assert launch._extract_tgz_binary(buf.getvalue(), dst) is True
    assert dst.is_file()
    assert dst.read_bytes() == payload


def test_wait_dns_propagation_doh_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _FakeResp:
        def __init__(self, payload: dict) -> None:
            self._data = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            import json

            return json.dumps(self._data).encode("utf-8")

    def fake_urlopen(req, timeout=3.0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp({"Status": 3})  # NXDOMAIN initially
        return _FakeResp({"Status": 0, "Answer": [{"data": "104.16.230.132"}]})

    monkeypatch.setattr(launch.urllib.request, "urlopen", fake_urlopen)
    assert launch.wait_dns_propagation("demo.trycloudflare.com", timeout_s=2.0, interval_s=0.01) is True
    assert calls["n"] == 2

    def fail_urlopen(req, timeout=3.0):
        raise OSError("network disabled")

    monkeypatch.setattr(launch.urllib.request, "urlopen", fail_urlopen)
    assert (
        launch.wait_dns_propagation(
            "demo.trycloudflare.com",
            timeout_s=0.05,
            interval_s=0.01,
            fallback_sleep_s=0.01,
        )
        is False
    )


def test_launch_no_tunnel_auto_key_and_no_secret_leak_in_log(tmp_path: Path) -> None:
    port = _free_port()
    data_dir = tmp_path / "be_data"
    log_file = tmp_path / "backend.log"

    env = os.environ.copy()
    env.pop("BACKEND_API_KEY", None)
    env["GPU_TOKEN"] = "gpu-super-secret-xyz"
    env["CLOUDFLARE_TUNNEL_TOKEN"] = "cf-super-secret-xyz"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(REPO_ROOT / "backend" / "launch.py"),
            "--no-tunnel",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
            "--log-file",
            str(log_file),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines: list[str] = []
    auto_key = ""
    try:
        assert proc.stdout is not None
        deadline = time.monotonic() + 45.0
        ready = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            lines.append(line)
            m = re.search(r"Backend Access Key\s*:\s*(\S+)", line)
            if m:
                auto_key = m.group(1)
            if "STATUS: READY" in line:
                ready = True
                break
        assert ready, f"launch.py did not print READY. Output:\n{''.join(lines)}"
        assert auto_key and auto_key != "DISABLED"

        # Probe local server with and without auto-generated key
        unauth = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=5.0).json()
        assert unauth["ok"] is True
        assert unauth["auth_required"] is True
        assert unauth["authenticated"] is False
        assert "gpu" not in unauth

        authed = httpx.get(
            f"http://127.0.0.1:{port}/api/health",
            headers={"Authorization": f"Bearer {auto_key}"},
            timeout=5.0,
        ).json()
        assert authed["authenticated"] is True
        assert "gpu" in authed
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)

    assert log_file.is_file()
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    assert auto_key not in log_text
    assert "gpu-super-secret-xyz" not in log_text
    assert "cf-super-secret-xyz" not in log_text
