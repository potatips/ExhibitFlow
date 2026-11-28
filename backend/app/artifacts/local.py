"""Local filesystem ArtifactStore with atomic writes and FileResponse support."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

from .base import (
    ArtifactStat,
    ArtifactStore,
    guess_content_type,
    validate_key,
    validate_prefix,
)

META_DIRNAME = ".meta"


class LocalArtifactStore(ArtifactStore):
    """Stores artifacts under a local root directory with atomic temp-file writes."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_root = self.root / META_DIRNAME

    def _resolve(self, key: str) -> tuple[str, Path]:
        norm = validate_key(key)
        target = (self.root / norm).resolve()
        if self.root not in target.parents or target == self._meta_root or self._meta_root in target.parents:
            raise ValueError(f"Key resolves outside artifact root: {key!r}")
        return norm, target

    def _meta_path(self, norm_key: str) -> Path:
        return (self._meta_root / f"{norm_key}.json").resolve()

    def _write_meta(self, norm_key: str, content_type: str) -> None:
        mp = self._meta_path(norm_key)
        mp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=mp.parent, prefix=".meta_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"content_type": content_type}, fh)
            os.replace(tmp, mp)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_content_type(self, norm_key: str) -> str:
        mp = self._meta_path(norm_key)
        if mp.is_file():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
                if ct := data.get("content_type"):
                    return str(ct)
            except (OSError, json.JSONDecodeError):
                pass
        return guess_content_type(norm_key)

    def _atomic_write_chunks(
        self, norm_key: str, target: Path, chunks: Iterable[bytes], content_type: str
    ) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".art_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in chunks:
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._write_meta(norm_key, content_type)
        return norm_key

    def put_bytes(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> str:
        norm, target = self._resolve(key)
        ct = guess_content_type(norm, content_type)
        return self._atomic_write_chunks(norm, target, (data,), ct)

    def put_file(
        self, key: str, src_path: str | Path, content_type: str | None = None
    ) -> str:
        norm, target = self._resolve(key)
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        ct = guess_content_type(norm, content_type)

        def _file_chunks() -> Iterator[bytes]:
            with src.open("rb") as fh:
                while chunk := fh.read(1 << 20):
                    yield chunk

        return self._atomic_write_chunks(norm, target, _file_chunks(), ct)

    def put_stream(
        self,
        key: str,
        stream: BinaryIO | Iterable[bytes],
        content_type: str | None = None,
    ) -> str:
        norm, target = self._resolve(key)
        ct = guess_content_type(norm, content_type)
        if hasattr(stream, "read"):
            def _reader() -> Iterator[bytes]:
                while chunk := stream.read(1 << 20):  # type: ignore[union-attr]
                    yield chunk

            chunks: Iterable[bytes] = _reader()
        else:
            chunks = stream
        return self._atomic_write_chunks(norm, target, chunks, ct)

    def get_bytes(self, key: str) -> bytes:
        _, target = self._resolve(key)
        if not target.is_file():
            raise FileNotFoundError(key)
        return target.read_bytes()

    def open_stream(self, key: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        _, target = self._resolve(key)
        if not target.is_file():
            raise FileNotFoundError(key)

        def _gen() -> Iterator[bytes]:
            with target.open("rb") as fh:
                while chunk := fh.read(chunk_size):
                    yield chunk

        return _gen()

    def local_path(self, key: str) -> Path | None:
        _, target = self._resolve(key)
        return target if target.is_file() else None

    def exists(self, key: str) -> bool:
        _, target = self._resolve(key)
        return target.is_file()

    def stat(self, key: str) -> ArtifactStat | None:
        norm, target = self._resolve(key)
        if not target.is_file():
            return None
        st = target.stat()
        return ArtifactStat(
            key=norm,
            size=st.st_size,
            content_type=self._read_content_type(norm),
            modified_at=st.st_mtime,
            url=None,
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        norm_prefix = validate_prefix(prefix)
        out: list[str] = []
        if not self.root.exists():
            return out
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path == self._meta_root or self._meta_root in path.parents:
                continue
            rel = path.relative_to(self.root).as_posix()
            if not norm_prefix or rel.startswith(norm_prefix):
                out.append(rel)
        out.sort()
        return out

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_keys(prefix)
        deleted = 0
        for key in keys:
            _, target = self._resolve(key)
            if target.is_file():
                target.unlink()
                deleted += 1
            mp = self._meta_path(key)
            if mp.is_file():
                mp.unlink()
            self._prune_empty_dirs(target.parent, self.root)
            self._prune_empty_dirs(mp.parent, self._meta_root)
        return deleted

    @staticmethod
    def _prune_empty_dirs(directory: Path, stop_at: Path) -> None:
        cur = directory
        while cur != stop_at and stop_at in cur.parents:
            try:
                cur.rmdir()
            except OSError:
                break
            cur = cur.parent

    def get_public_url(self, key: str) -> str | None:
        validate_key(key)
        return None
