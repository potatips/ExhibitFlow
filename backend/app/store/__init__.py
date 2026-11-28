"""StateStore factory and exports."""

from __future__ import annotations

from ..config import Settings
from .base import StateStore
from .redis_store import RedisStore
from .sqlite_store import SQLiteStore


def create_state_store(settings: Settings) -> StateStore:
    """Instantiate RedisStore when REDIS_URL is set, otherwise SQLiteStore."""
    if settings.redis_url:
        return RedisStore(url=settings.redis_url)
    assert settings.sqlite_path is not None
    return SQLiteStore(settings.sqlite_path)


__all__ = [
    "StateStore",
    "SQLiteStore",
    "RedisStore",
    "create_state_store",
]
