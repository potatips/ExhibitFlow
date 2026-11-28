"""Abstract ArtifactStore interface, key validation, and content-type helpers."""

from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Iterator


@dataclass(frozen=True)
class ArtifactStat:
    """Metadata describing a stored artifact object."""

    key: str
    size: int
    content_type: str
    modified_at: float | None = None
    url: str | None = None


def guess_content_type(key: str, explicit: str | None = None) -> str:
    """Infer MIME content-type from key extension unless explicitly provided."""
    if explicit:
        return explicit
    guessed, _ = mimetypes.guess_type(key)
    return guessed or "application/octet-stream"


def validate_key(key: str) -> str:
    """Validate and normalize a relative POSIX artifact key.

    Rejects empty strings, leading slashes, null bytes, backslashes, and '.' / '..'
    path segments to guarantee path-traversal safety across all store backends.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("Artifact key must be a non-empty string")
    if "\x00" in key:
        raise ValueError("Artifact key must not contain null bytes")
    if key.startswith("/"):
        raise ValueError("Artifact key must not start with '/'")
    if "\\" in key:
        raise ValueError("Artifact key must not contain backslashes")
    parts = key.split("/")
    for seg in parts:
        if seg in ("", ".", ".."):
            raise ValueError(f"Invalid path segment in artifact key: {key!r}")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"Invalid artifact key: {key!r}")
    return normalized


def validate_prefix(prefix: str) -> str:
    """Validate a key prefix (may be empty string or end with '/')."""
    if prefix == "":
        return ""
    if not isinstance(prefix, str):
        raise ValueError("Prefix must be a string")
    if "\x00" in prefix:
        raise ValueError("Prefix must not contain null bytes")
    if prefix.startswith("/"):
        raise ValueError("Prefix must not start with '/'")
    if "\\" in prefix:
        raise ValueError("Prefix must not contain backslashes")
    trailing = prefix.endswith("/")
    stripped = prefix.rstrip("/")
    if not stripped:
        raise ValueError("Invalid prefix")
    norm = validate_key(stripped)
    return f"{norm}/" if trailing else norm


class ArtifactStore(ABC):
    """Pluggable binary artifact storage interface (local disk, Vercel Blob, S3/R2)."""

    @abstractmethod
    def put_bytes(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> str:
        """Store raw bytes at key and return the normalized key."""

    @abstractmethod
    def put_file(
        self, key: str, src_path: str | Path, content_type: str | None = None
    ) -> str:
        """Store contents of a local file at key and return the normalized key."""

    @abstractmethod
    def put_stream(
        self,
        key: str,
        stream: BinaryIO | Iterable[bytes],
        content_type: str | None = None,
    ) -> str:
        """Store a binary stream or byte-chunk iterable at key."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read entire artifact payload as bytes. Raises FileNotFoundError if absent."""

    @abstractmethod
    def open_stream(self, key: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        """Yield chunks of bytes from key. Raises FileNotFoundError if absent."""

    @abstractmethod
    def local_path(self, key: str) -> Path | None:
        """Return a local filesystem Path if directly accessible (for FileResponse), else None."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if key exists in the store."""

    @abstractmethod
    def stat(self, key: str) -> ArtifactStat | None:
        """Return ArtifactStat for key, or None if it does not exist."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all stored keys matching prefix in sorted order."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete all artifacts matching prefix and return the number deleted."""

    @abstractmethod
    def get_public_url(self, key: str) -> str | None:
        """Return a direct public/CDN URL if supported by the backend, else None."""
