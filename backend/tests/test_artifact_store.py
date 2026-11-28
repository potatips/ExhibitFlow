"""Unit tests for LocalArtifactStore, VercelBlobArtifactStore, and key validation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest

from backend.app.artifacts import (
    LocalArtifactStore,
    VercelBlobArtifactStore,
    create_artifact_store,
    validate_key,
)
from backend.app.config import Settings


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "../secret.txt",
        "jobs/../../etc/passwd",
        "/absolute/path.mp4",
        "jobs/./file.json",
        "jobs\\win\\file.mp4",
        "jobs/bad\x00key.png",
        "jobs//double.png",
    ],
)
def test_validate_key_rejects_traversal(bad_key: str) -> None:
    with pytest.raises(ValueError):
        validate_key(bad_key)


def test_local_artifact_store_crud_and_streaming(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    k1 = store.put_bytes("jobs/j1/result.json", b'{"ok":true}', content_type="application/json")
    assert k1 == "jobs/j1/result.json"
    assert store.exists(k1)
    assert store.get_bytes(k1) == b'{"ok":true}'
    assert store.get_public_url(k1) is None

    st = store.stat(k1)
    assert st is not None
    assert st.size == len(b'{"ok":true}')
    assert st.content_type == "application/json"

    lp = store.local_path(k1)
    assert lp is not None and lp.is_file()

    # put_file
    src = tmp_path / "sample.mp4"
    src.write_bytes(b"0000ftypisom" + b"x" * 2048)
    store.put_file("jobs/j1/live.mp4", src, content_type="video/mp4")
    assert b"".join(store.open_stream("jobs/j1/live.mp4", chunk_size=512)) == src.read_bytes()

    # put_stream with file-like & iterable
    store.put_stream("jobs/j1/files/k/sheet.jpg", io.BytesIO(b"\xff\xd8\xff\xd9"), content_type="image/jpeg")
    store.put_stream("jobs/j2/map.png", [b"\x89PNG", b"\r\n\x1a\n"], content_type="image/png")

    # list_keys excludes internal .meta directory
    all_keys = store.list_keys()
    assert all_keys == [
        "jobs/j1/files/k/sheet.jpg",
        "jobs/j1/live.mp4",
        "jobs/j1/result.json",
        "jobs/j2/map.png",
    ]
    assert store.list_keys("jobs/j1/") == [
        "jobs/j1/files/k/sheet.jpg",
        "jobs/j1/live.mp4",
        "jobs/j1/result.json",
    ]

    # path-traversal rejected by store methods too
    with pytest.raises(ValueError):
        store.put_bytes("../escape.txt", b"no")
    with pytest.raises(ValueError):
        store.local_path("/etc/passwd")

    # delete_prefix
    deleted = store.delete_prefix("jobs/j1/")
    assert deleted == 3
    assert store.list_keys() == ["jobs/j2/map.png"]
    assert not store.exists("jobs/j1/result.json")


def test_vercel_blob_artifact_store_with_mock_transport(tmp_path: Path) -> None:
    blobs: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.lstrip("/")
        if request.method == "PUT":
            assert request.headers.get("authorization") == "Bearer test-rw-token"
            ct = request.headers.get("x-content-type", "application/octet-stream")
            url = f"https://cdn.vercel-storage.test/{path}"
            blobs[path] = {"pathname": path, "url": url, "size": len(request.content), "contentType": ct, "body": request.content}
            return httpx.Response(200, json={"url": url, "pathname": path, "contentType": ct})
        if request.method == "GET" and request.url.host == "blob.vercel-storage.com":
            prefix = request.url.params.get("prefix", "")
            matched = [
                {k: v for k, v in b.items() if k != "body"}
                for p, b in sorted(blobs.items())
                if p.startswith(prefix)
            ]
            return httpx.Response(200, json={"blobs": matched, "hasMore": False})
        if request.method == "GET" and request.url.host == "cdn.vercel-storage.test":
            b = blobs.get(path)
            if not b:
                return httpx.Response(404)
            return httpx.Response(200, content=b["body"], headers={"content-type": b["contentType"]})
        if request.method == "POST" and path == "delete":
            body = json.loads(request.content.decode("utf-8"))
            urls = set(body.get("urls", []))
            to_del = [k for k, v in blobs.items() if v["url"] in urls]
            for k in to_del:
                del blobs[k]
            return httpx.Response(200, json={"deleted": len(to_del)})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VercelBlobArtifactStore(token="test-rw-token", client=client)
    try:
        store.put_bytes("jobs/j1/result.json", b'{"demo":1}', content_type="application/json")
        assert store.local_path("jobs/j1/result.json") is None
        assert store.exists("jobs/j1/result.json")
        assert store.get_bytes("jobs/j1/result.json") == b'{"demo":1}'
        assert b"".join(store.open_stream("jobs/j1/result.json")) == b'{"demo":1}'
        assert store.get_public_url("jobs/j1/result.json") == "https://cdn.vercel-storage.test/jobs/j1/result.json"

        st = store.stat("jobs/j1/result.json")
        assert st is not None and st.size == 10 and st.content_type == "application/json"

        src = tmp_path / "vid.mp4"
        src.write_bytes(b"mp4-bytes")
        store.put_file("jobs/j1/live.mp4", src, content_type="video/mp4")
        assert store.list_keys("jobs/j1/") == ["jobs/j1/live.mp4", "jobs/j1/result.json"]

        with pytest.raises(ValueError):
            store.put_bytes("../bad", b"x")

        assert store.delete_prefix("jobs/j1/") == 2
        assert store.list_keys() == []
    finally:
        client.close()


def test_create_artifact_store_factory(tmp_path: Path) -> None:
    s_local = Settings(data_dir=tmp_path, artifact_store="local")
    assert isinstance(create_artifact_store(s_local), LocalArtifactStore)

    s_vb = Settings(
        data_dir=tmp_path,
        artifact_store="vercel_blob",
        blob_read_write_token="tok-123",
    )
    assert isinstance(create_artifact_store(s_vb), VercelBlobArtifactStore)

    with pytest.raises(ValueError):
        create_artifact_store(Settings(data_dir=tmp_path, artifact_store="unknown"))
