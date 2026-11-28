"""Unit tests for SharedVLMServer singleton and LlamaCppVLM shared server mode."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from exhibitflow_gpu.vlm_server import SharedVLMServer


class _MockLlamaHandler(BaseHTTPRequestHandler):
    requests_seen: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        self.requests_seen.append(f"GET {self.path}")
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(length)
        self.requests_seen.append(f"POST {self.path}")
        if self.path == "/v1/chat/completions":
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "label": "touching",
                                    "confidence": 0.91,
                                    "evidence": "Hand contacted exhibit",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18},
                "timings": {"prompt_ms": 25.0, "predicted_per_second": 42.0},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


@pytest.fixture()
def mock_vlm_server():
    _MockLlamaHandler.requests_seen = []
    srv = HTTPServer(("127.0.0.1", 0), _MockLlamaHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", _MockLlamaHandler.requests_seen
    finally:
        srv.shutdown()
        srv.server_close()


def test_shared_vlm_server_singleton_and_status(mock_vlm_server) -> None:
    url, seen = mock_vlm_server
    s1 = SharedVLMServer.instance()
    s2 = SharedVLMServer.instance()
    assert s1 is s2

    s1.close()
    try:
        resolved = s1.ensure_started({"server_url": url, "repo_id": "test/Qwen3-VL-8B-Instruct-GGUF"})
        assert resolved == url
        st = s1.status()
        assert st["state"] == "ready"
        assert st["url"] == url
        assert st["shared"] is True
        assert "GET /health" in seen
    finally:
        s1.close()
        assert s1.status()["state"] == "stopped"


def test_llama_cpp_vlm_reuses_shared_server(mock_vlm_server, tmp_path: Path) -> None:
    from PIL import Image
    from exhibitflow_gpu.vlm import LlamaCppVLM

    url, seen = mock_vlm_server
    img_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), (100, 120, 140)).save(img_path, format="JPEG")

    vlm = LlamaCppVLM(server_url=url, startup_timeout=5.0)
    assert vlm.shared_server is True
    assert vlm.proc is None
    assert vlm.server_url == url
    assert "GET /health" in seen

    text, n_tok, meta = vlm.infer(
        {
            "images": [img_path],
            "times": [0.5],
            "prompt": "Classify interaction.",
            "system": "Be concise.",
        }
    )
    assert "POST /v1/chat/completions" in seen
    assert "touching" in text
    assert n_tok == 120
    assert meta["n_output_tokens"] == 18

    # close() must be a no-op when using shared server
    vlm.close()
    assert vlm.server_url == url
