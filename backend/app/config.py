"""Configuration settings for ExhibitFlow CPU backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_web_dir() -> Path | None:
    env_val = os.environ.get("WEB_DIR", "").strip()
    if env_val:
        p = Path(env_val).expanduser().resolve()
        return p if p.is_dir() else None
    candidate = Path(__file__).resolve().parents[2] / "web"
    return candidate if candidate.is_dir() else None


def _default_data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR") or "./backend_data").expanduser()


@dataclass
class Settings:
    """Runtime settings loaded from environment variables or explicit overrides."""

    data_dir: Path = field(default_factory=_default_data_dir)
    sqlite_path: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["SQLITE_PATH"]).expanduser()
            if os.environ.get("SQLITE_PATH", "").strip()
            else None
        )
    )
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "").strip())
    gpu_worker_url: str = field(
        default_factory=lambda: os.environ.get("GPU_WORKER_URL", "").strip().rstrip("/")
    )
    gpu_token: str = field(default_factory=lambda: os.environ.get("GPU_TOKEN", ""))
    backend_api_key: str = field(
        default_factory=lambda: os.environ.get("BACKEND_API_KEY", "").strip()
    )
    stream_token_ttl_s: int = field(
        default_factory=lambda: int(os.environ.get("STREAM_TOKEN_TTL_S") or "600")
    )
    cors_origins: str = field(
        default_factory=lambda: os.environ.get("CORS_ORIGINS", "*").strip()
    )
    artifact_store: str = field(
        default_factory=lambda: (os.environ.get("ARTIFACT_STORE") or "local").strip().lower()
    )
    blob_read_write_token: str = field(
        default_factory=lambda: os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
    )
    web_dir: Path | None = field(default_factory=_detect_web_dir)
    max_upload_mb: int = field(
        default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB") or "600")
    )
    poll_interval_s: float = field(
        default_factory=lambda: float(os.environ.get("POLL_INTERVAL_S") or "0.5")
    )
    health_interval_s: float = field(
        default_factory=lambda: float(os.environ.get("HEALTH_INTERVAL_S") or "2.0")
    )
    port: int = field(default_factory=lambda: int(os.environ.get("PORT") or "8000"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.sqlite_path is None:
            self.sqlite_path = self.data_dir / "state.db"
        else:
            self.sqlite_path = Path(self.sqlite_path)
        self.gpu_worker_url = (self.gpu_worker_url or "").strip().rstrip("/")
        self.redis_url = (self.redis_url or "").strip()
        self.backend_api_key = (self.backend_api_key or "").strip()
        self.stream_token_ttl_s = max(10, int(self.stream_token_ttl_s))
        self.cors_origins = (self.cors_origins or "*").strip()
        self.artifact_store = (self.artifact_store or "local").strip().lower()
        if isinstance(self.web_dir, str):
            p = Path(self.web_dir).expanduser()
            self.web_dir = p if p.is_dir() else None

    @property
    def max_upload_bytes(self) -> int:
        return max(1, int(self.max_upload_mb)) * 1024 * 1024

    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "*").strip()
        if not raw or raw == "*":
            return ["*"]
        origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
        return origins or ["*"]


def load_settings() -> Settings:
    """Create a Settings instance from the current environment."""
    return Settings()
