"""Pydantic schemas and shared constants for ExhibitFlow CPU backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_STAGES = [
    "Khởi động & nạp model",
    "Tách hiện vật",
    "Nạp VLM",
    "Phân tích video",
    "Ghi kết quả",
]

SITE_STAGES = [
    "Cắt khung đã khử méo",
    "Phát hiện người + ghép các camera",
    "Hoàn tất",
]

SITE_STAGES_EV = [
    "Cắt khung đã khử méo",
    "Phát hiện người + ghép các camera",
    "Phát hiện hành vi bất thường (từng camera)",
    "Chiếu hành vi bất thường lên map",
    "Hoàn tất",
]


class GpuConfigUpdate(BaseModel):
    """Payload for updating GPU worker connection settings at runtime."""

    url: str = ""
    token: str | None = None
    clear_token: bool = False


class GpuConfigResponse(BaseModel):
    """Public GPU configuration view (never exposes raw secret token)."""

    url: str
    has_token: bool


class AlertReviewUpdate(BaseModel):
    """Payload for reviewing an alert (ok, no, or null to reset)."""

    review: Literal["ok", "no"] | None = None


class JobSubmitResponse(BaseModel):
    """Response returned immediately when a job is queued."""

    job_id: str
    status: str = "queued"
    queue_position: int


class GpuHealthSnapshot(BaseModel):
    """Normalized GPU health block included in /api/health."""

    online: bool = False
    busy: bool = False
    demo: bool = False
    models: dict[str, Any] = Field(default_factory=lambda: {"state": "unknown", "msg": ""})
    vlm: dict[str, Any] = Field(default_factory=lambda: {"state": "unknown"})
    error: str | None = None
    url: str = ""
