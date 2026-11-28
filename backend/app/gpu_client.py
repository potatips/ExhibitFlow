"""Async HTTP client for communicating with the remote GPU_server worker."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

import httpx

from .artifacts.base import ArtifactStore


class GPUClientError(RuntimeError):
    """Structured error raised when GPU_server returns an error response or is unreachable."""

    def __init__(self, status_code: int, detail: str, retryable: bool = False) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retryable = retryable


def compute_gpu_ready(health: dict[str, Any]) -> bool:
    """Compute honest readiness: online AND (demo OR (models.state == 'ready' and vlm.state != 'error') OR active GPU job)."""
    online = bool(health.get("online"))
    demo = bool(health.get("demo"))
    busy = bool(health.get("busy"))
    models_state = str((health.get("models") or {}).get("state") or "").lower()
    vlm_state = str((health.get("vlm") or {}).get("state") or "").lower()
    return bool(
        online
        and (
            demo
            or (models_state == "ready" and vlm_state != "error")
            or (busy and models_state != "error" and vlm_state != "error")
        )
    )


def _extract_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        return json.dumps(data, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return resp.text or f"HTTP {resp.status_code}"


class GPUClient:
    """Async httpx client wrapping all /internal/* routes on GPU_server."""

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.token = token or ""
        self._timeout = timeout
        self._transport = transport
        self._client = self._build_client()

    def _build_client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(
                connect=5.0, read=self._timeout, write=self._timeout, pool=5.0
            ),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def configure(self, base_url: str, token: str | None = None) -> None:
        """Update target GPU_server URL and optional token at runtime."""
        self.base_url = (base_url or "").strip().rstrip("/")
        if token is not None:
            self.token = token

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.token:
            h["X-GPU-Token"] = self.token
        if extra:
            h.update(extra)
        return h

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise GPUClientError(503, "GPU Worker URL chưa được cấu hình", retryable=True)
        return f"{self.base_url}{path}"

    async def _get_json(self, path: str, timeout: float | None = None) -> Any:
        try:
            resp = await self._client.get(
                self._url(path),
                headers=self._headers(),
                timeout=timeout or self._timeout,
            )
            if resp.status_code >= 400:
                raise GPUClientError(
                    resp.status_code,
                    _extract_error(resp),
                    retryable=resp.status_code in (409, 429, 502, 503, 504),
                )
            return resp.json()
        except GPUClientError:
            raise
        except httpx.HTTPError as exc:
            raise GPUClientError(503, f"GPU offline: {exc}", retryable=True) from exc

    async def _get_bytes(self, path: str, timeout: float | None = None) -> bytes | None:
        try:
            resp = await self._client.get(
                self._url(path),
                headers=self._headers(),
                timeout=timeout or self._timeout,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code >= 400:
                raise GPUClientError(resp.status_code, _extract_error(resp))
            return resp.content
        except GPUClientError:
            raise
        except httpx.HTTPError:
            return None

    # ---- Health, System, Schema, Sites ----
    async def health(self) -> dict[str, Any]:
        """Probe GPU_server health safely; never raises when offline."""
        if not self.base_url:
            return {
                "online": False,
                "ready": False,
                "busy": False,
                "demo": False,
                "models": {"state": "unknown", "msg": ""},
                "vlm": {"state": "unknown"},
                "error": "GPU Worker URL chưa được cấu hình",
                "url": "",
            }
        try:
            resp = await self._client.get(
                self._url("/internal/health"),
                headers=self._headers(),
                timeout=4.0,
            )
            if resp.status_code >= 400:
                return {
                    "online": False,
                    "ready": False,
                    "busy": False,
                    "demo": False,
                    "models": {"state": "error", "msg": ""},
                    "vlm": {"state": "unknown"},
                    "error": _extract_error(resp),
                    "url": self.base_url,
                }
            data = resp.json()
            out = {
                "online": True,
                "version": data.get("version"),
                "demo": bool(data.get("demo")),
                "models": data.get("models") or {"state": "ready", "msg": ""},
                "vlm": data.get("vlm") or {"state": "unknown"},
                "busy": bool(data.get("busy")),
                "active_job_id": data.get("active_job_id"),
                "det_conf": data.get("det_conf"),
                "det_conf_default": data.get("det_conf_default", 0.01),
                "det_conf_effective": data.get(
                    "det_conf_effective",
                    0.01 if data.get("det_conf") is None else data.get("det_conf"),
                ),
                "capabilities": data.get("capabilities") or {},
                "error": None,
                "url": self.base_url,
            }
            out["ready"] = compute_gpu_ready(out)
            return out
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "online": False,
                "ready": False,
                "busy": False,
                "demo": False,
                "models": {"state": "unknown", "msg": ""},
                "vlm": {"state": "unknown"},
                "error": str(exc) or "Connection failed",
                "url": self.base_url,
            }

    async def system(self) -> dict[str, Any] | None:
        try:
            return await self._get_json("/internal/system", timeout=5.0)
        except GPUClientError:
            return None

    async def tuning_schema(self) -> dict[str, Any] | None:
        try:
            return await self._get_json("/internal/tuning/schema", timeout=5.0)
        except GPUClientError:
            return None

    async def sites(self) -> list[dict[str, Any]] | None:
        try:
            data = await self._get_json("/internal/sites", timeout=5.0)
            return data if isinstance(data, list) else None
        except GPUClientError:
            return None

    async def set_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                self._url("/internal/config"),
                json=payload,
                headers=self._headers(),
                timeout=8.0,
            )
            if resp.status_code >= 400:
                raise GPUClientError(
                    resp.status_code,
                    _extract_error(resp),
                    retryable=resp.status_code in (409, 429, 502, 503, 504),
                )
            return resp.json()
        except GPUClientError:
            raise
        except httpx.HTTPError as exc:
            raise GPUClientError(503, f"GPU offline: {exc}", retryable=True) from exc

    # ---- Single-cam Jobs ----
    async def submit_job(
        self,
        job_id: str,
        video_path: Path | None = None,
        video_name: str | None = None,
        source: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {
            "job_id": job_id,
            "options": json.dumps(options or {}, ensure_ascii=False),
        }
        if source:
            data["source"] = source

        fh = None
        try:
            files = None
            if video_path is not None:
                p = Path(video_path)
                fh = p.open("rb")
                files = {"video": (video_name or p.name, fh, "application/octet-stream")}
            resp = await self._client.post(
                self._url("/internal/jobs"),
                data=data,
                files=files,
                headers=self._headers(),
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=300.0, pool=10.0),
            )
            if resp.status_code >= 400:
                raise GPUClientError(
                    resp.status_code,
                    _extract_error(resp),
                    retryable=resp.status_code in (409, 429, 502, 503, 504),
                )
            return resp.json()
        except GPUClientError:
            raise
        except httpx.HTTPError as exc:
            raise GPUClientError(503, f"Submit job failed: {exc}", retryable=True) from exc
        finally:
            if fh is not None:
                fh.close()

    async def job_status(self, job_id: str) -> dict[str, Any] | None:
        try:
            return await self._get_json(f"/internal/jobs/{job_id}", timeout=6.0)
        except GPUClientError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            resp = await self._client.post(
                self._url(f"/internal/jobs/{job_id}/cancel"),
                headers=self._headers(),
                timeout=6.0,
            )
            if resp.status_code >= 400:
                return None
            return resp.json()
        except (GPUClientError, httpx.HTTPError):
            return None

    async def finish_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            resp = await self._client.post(
                self._url(f"/internal/jobs/{job_id}/finish"),
                headers=self._headers(),
                timeout=6.0,
            )
            if resp.status_code >= 400:
                return None
            return resp.json()
        except (GPUClientError, httpx.HTTPError):
            return None

    # ---- Live Frame, MJPEG, Events, Map ----
    async def live_frame(self, job_id: str) -> bytes | None:
        return await self._get_bytes(f"/internal/jobs/{job_id}/live", timeout=5.0)

    async def stream_mjpeg(self, job_id: str) -> AsyncIterator[bytes]:
        url = self._url(f"/internal/jobs/{job_id}/live/stream")
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        try:
            async with self._client.stream(
                "GET", url, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status_code >= 400:
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except (GPUClientError, httpx.HTTPError):
            return

    async def live_events(self, job_id: str, after: int = 0) -> dict[str, Any] | None:
        try:
            return await self._get_json(
                f"/internal/jobs/{job_id}/live/events?after={int(after)}", timeout=5.0
            )
        except GPUClientError:
            return None

    async def live_map(self, job_id: str) -> bytes | None:
        return await self._get_bytes(f"/internal/jobs/{job_id}/map", timeout=5.0)

    async def live_map_data(self, job_id: str, full: int = 0) -> dict[str, Any] | None:
        try:
            return await self._get_json(
                f"/internal/jobs/{job_id}/map/data?full={1 if full else 0}", timeout=5.0
            )
        except GPUClientError:
            return None

    # ---- Artifacts Sync ----
    async def list_artifacts(
        self, job_id: str, is_site: bool = False
    ) -> dict[str, Any] | None:
        kind = "site-jobs" if is_site else "jobs"
        try:
            return await self._get_json(
                f"/internal/{kind}/{job_id}/artifacts", timeout=15.0
            )
        except GPUClientError:
            return None

    async def download_artifact_to_store(
        self,
        job_id: str,
        rel_path: str,
        store: ArtifactStore,
        dst_key: str,
        is_site: bool = False,
        content_type: str | None = None,
    ) -> bool:
        kind = "site-jobs" if is_site else "jobs"
        encoded = "/".join(quote(seg, safe="") for seg in rel_path.split("/"))
        url = self._url(f"/internal/{kind}/{job_id}/artifacts/{encoded}")
        tmp_path: Path | None = None
        try:
            timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
            async with self._client.stream(
                "GET", url, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status_code == 404:
                    return False
                if resp.status_code >= 400:
                    raise GPUClientError(resp.status_code, _extract_error(resp))
                ct = content_type or resp.headers.get("content-type")
                fd, tmp_name = tempfile.mkstemp(prefix="ef_art_", suffix=".tmp")
                tmp_path = Path(tmp_name)
                with os.fdopen(fd, "wb") as fh:
                    async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            await asyncio.to_thread(store.put_file, dst_key, tmp_path, ct)
            return True
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    async def fetch_artifact_bytes(
        self, job_id: str, rel_path: str, is_site: bool = False
    ) -> bytes | None:
        kind = "site-jobs" if is_site else "jobs"
        encoded = "/".join(quote(seg, safe="") for seg in rel_path.split("/"))
        return await self._get_bytes(f"/internal/{kind}/{job_id}/artifacts/{encoded}")

    # ---- Calibrated Site Jobs ----
    async def submit_site_job(
        self,
        job_id: str,
        form_fields: dict[str, str],
        files_map: dict[str, tuple[str, Path, str]],
    ) -> dict[str, Any]:
        data = dict(form_fields)
        data["job_id"] = job_id
        opened = []
        files_payload: list[tuple[str, tuple[str, Any, str]]] = []
        try:
            for field_name, (filename, path, content_type) in files_map.items():
                fh = Path(path).open("rb")
                opened.append(fh)
                files_payload.append((field_name, (filename, fh, content_type)))
            resp = await self._client.post(
                self._url("/internal/site-jobs"),
                data=data,
                files=files_payload,
                headers=self._headers(),
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=300.0, pool=10.0),
            )
            if resp.status_code >= 400:
                raise GPUClientError(
                    resp.status_code,
                    _extract_error(resp),
                    retryable=resp.status_code in (409, 429, 502, 503, 504),
                )
            return resp.json()
        except GPUClientError:
            raise
        except httpx.HTTPError as exc:
            raise GPUClientError(503, f"Submit site-job failed: {exc}", retryable=True) from exc
        finally:
            for fh in opened:
                fh.close()

    async def site_job_status(self, job_id: str) -> dict[str, Any] | None:
        try:
            return await self._get_json(f"/internal/site-jobs/{job_id}", timeout=6.0)
        except GPUClientError as exc:
            if exc.status_code == 404:
                return None
            raise
