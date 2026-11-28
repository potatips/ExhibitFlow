"""ArtifactStore factory and exports."""

from __future__ import annotations

from ..config import Settings
from .base import ArtifactStat, ArtifactStore, guess_content_type, validate_key, validate_prefix
from .local import LocalArtifactStore
from .vercel_blob import VercelBlobArtifactStore


def create_artifact_store(settings: Settings) -> ArtifactStore:
    """Create the configured ArtifactStore backend ('local' or 'vercel_blob')."""
    kind = (settings.artifact_store or "local").strip().lower()
    if kind == "local":
        return LocalArtifactStore(settings.data_dir / "artifacts")
    if kind == "vercel_blob":
        return VercelBlobArtifactStore(token=settings.blob_read_write_token)
    raise ValueError(f"Unsupported ARTIFACT_STORE backend: {settings.artifact_store!r}")


__all__ = [
    "ArtifactStat",
    "ArtifactStore",
    "LocalArtifactStore",
    "VercelBlobArtifactStore",
    "create_artifact_store",
    "guess_content_type",
    "validate_key",
    "validate_prefix",
]
