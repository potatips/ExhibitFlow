"""Pluggable Vercel Blob ArtifactStore adapter (with S3/R2 extension guide).

Extension Guide for S3 / Cloudflare R2:
    To add an S3-compatible or Cloudflare R2 artifact backend, subclass
    `ArtifactStore` in a sibling module (e.g. `backend/app/artifacts/s3.py`)
    implementing the same 11 abstract methods using `boto3` or signed HTTP
    requests against the S3 bucket endpoint:
      - `put_bytes` / `put_file` / `put_stream` -> `put_object` / multipart upload
      - `get_bytes` / `open_stream` -> `get_object` streaming `Body`
      - `local_path` -> return `None` (backend streams or redirects via presigned URL)
      - `exists` / `stat` -> `head_object`
      - `list_keys` -> `list_objects_v2`
      - `delete_prefix` -> `delete_objects`
      - `get_public_url` -> `generate_presigned_url` or CDN public URL
    Then register the new `ARTIFACT_STORE` name in `create_artifact_store()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

import httpx

from .base import (
    ArtifactStat,
    ArtifactStore,
    guess_content_type,
    validate_key,
    validate_prefix,
)


class VercelBlobArtifactStore(ArtifactStore):
    """ArtifactStore adapter for the Vercel Blob HTTP REST API."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://blob.vercel-storage.com",
        prefix: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        if not token:
            raise ValueError("BLOB_READ_WRITE_TOKEN is required for VercelBlobArtifactStore")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.prefix = validate_prefix(prefix) if prefix else ""
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=60.0)
        self._url_cache: dict[str, str] = {}

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}"}
        if extra:
            h.update(extra)
        return h

    def _remote_key(self, key: str) -> str:
        norm = validate_key(key)
        return f"{self.prefix}{norm}" if self.prefix else norm

    def _strip_prefix(self, pathname: str) -> str:
        if self.prefix and pathname.startswith(self.prefix):
            return pathname[len(self.prefix) :]
        return pathname

    def _list_blobs(self, prefix: str = "") -> list[dict[str, Any]]:
        norm_p = validate_prefix(prefix)
        full_prefix = f"{self.prefix}{norm_p}" if self.prefix else norm_p
        blobs: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, str] = {}
            if full_prefix:
                params["prefix"] = full_prefix
            if cursor:
                params["cursor"] = cursor
            resp = self._client.get(
                f"{self.base_url}/",
                params=params,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("blobs", []):
                pathname = str(item.get("pathname") or "")
                if pathname and item.get("url"):
                    self._url_cache[pathname] = str(item["url"])
                blobs.append(item)
            if not data.get("hasMore") or not data.get("cursor"):
                break
            cursor = str(data["cursor"])
        return blobs

    def _resolve_blob(self, key: str) -> dict[str, Any] | None:
        rkey = self._remote_key(key)
        for item in self._list_blobs(validate_key(key)):
            if item.get("pathname") == rkey:
                return item
        return None

    def put_bytes(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> str:
        norm = validate_key(key)
        rkey = self._remote_key(norm)
        ct = guess_content_type(norm, content_type)
        resp = self._client.put(
            f"{self.base_url}/{rkey}",
            content=data,
            headers=self._headers(
                {
                    "x-content-type": ct,
                    "x-add-random-suffix": "0",
                    "content-type": ct,
                }
            ),
        )
        resp.raise_for_status()
        payload = resp.json()
        if url := payload.get("url"):
            self._url_cache[rkey] = str(url)
        return norm

    def put_file(
        self, key: str, src_path: str | Path, content_type: str | None = None
    ) -> str:
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        with src.open("rb") as fh:
            return self.put_stream(key, fh, content_type=content_type)

    def put_stream(
        self,
        key: str,
        stream: BinaryIO | Iterable[bytes],
        content_type: str | None = None,
    ) -> str:
        if hasattr(stream, "read"):
            chunks: list[bytes] = []
            while chunk := stream.read(1 << 20):  # type: ignore[union-attr]
                chunks.append(chunk)
            data = b"".join(chunks)
        else:
            data = b"".join(stream)
        return self.put_bytes(key, data, content_type=content_type)

    def _download_url(self, key: str) -> str:
        rkey = self._remote_key(key)
        if rkey in self._url_cache:
            return self._url_cache[rkey]
        blob = self._resolve_blob(key)
        if blob and blob.get("url"):
            return str(blob["url"])
        raise FileNotFoundError(key)

    def get_bytes(self, key: str) -> bytes:
        url = self._download_url(key)
        resp = self._client.get(url)
        if resp.status_code == 404:
            raise FileNotFoundError(key)
        resp.raise_for_status()
        return resp.content

    def open_stream(self, key: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        url = self._download_url(key)

        def _gen() -> Iterator[bytes]:
            with self._client.stream("GET", url) as resp:
                if resp.status_code == 404:
                    raise FileNotFoundError(key)
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    yield chunk

        return _gen()

    def local_path(self, key: str) -> Path | None:
        validate_key(key)
        return None

    def exists(self, key: str) -> bool:
        return self._resolve_blob(key) is not None

    def stat(self, key: str) -> ArtifactStat | None:
        norm = validate_key(key)
        blob = self._resolve_blob(norm)
        if blob is None:
            return None
        return ArtifactStat(
            key=norm,
            size=int(blob.get("size") or 0),
            content_type=str(blob.get("contentType") or guess_content_type(norm)),
            modified_at=None,
            url=blob.get("url"),
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        blobs = self._list_blobs(prefix)
        keys = [
            self._strip_prefix(str(b.get("pathname") or ""))
            for b in blobs
            if b.get("pathname")
        ]
        keys.sort()
        return keys

    def delete_prefix(self, prefix: str) -> int:
        blobs = self._list_blobs(prefix)
        urls = [str(b["url"]) for b in blobs if b.get("url")]
        if not urls:
            return 0
        resp = self._client.post(
            f"{self.base_url}/delete",
            json={"urls": urls},
            headers=self._headers(),
        )
        resp.raise_for_status()
        for b in blobs:
            if pathname := b.get("pathname"):
                self._url_cache.pop(str(pathname), None)
        return len(urls)

    def get_public_url(self, key: str) -> str | None:
        rkey = self._remote_key(key)
        if rkey in self._url_cache:
            return self._url_cache[rkey]
        blob = self._resolve_blob(key)
        return str(blob["url"]) if blob and blob.get("url") else None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
