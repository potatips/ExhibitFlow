"""Server-Sent Events (SSE) multi-subscriber pub/sub broker."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, AsyncIterator


def format_sse(event: str, data: Any) -> str:
    """Format an event name and JSON-serializable payload as an SSE frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


class EventBroker:
    """In-process pub/sub broker supporting 'global' and 'job:{id}' SSE topics."""

    def __init__(self, max_queue_size: int = 256) -> None:
        self._max_queue_size = max_queue_size
        self._topics: dict[str, set[asyncio.Queue[tuple[str, Any]]]] = defaultdict(set)

    def publish(self, topic: str, event: str, data: Any) -> None:
        """Broadcast an event to all active subscribers on topic without blocking."""
        queues = list(self._topics.get(topic, ()))
        for q in queues:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                pass

    def subscriber_count(self, topic: str) -> int:
        """Return number of currently connected subscribers on topic."""
        return len(self._topics.get(topic, ()))

    async def subscribe(
        self,
        topic: str,
        initial: list[tuple[str, Any]] | None = None,
        keepalive_s: float = 15.0,
    ) -> AsyncIterator[str]:
        """Yield formatted SSE frames for topic with periodic keepalive comments."""
        q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=self._max_queue_size)
        self._topics[topic].add(q)
        try:
            if initial:
                for event, data in initial:
                    yield format_sse(event, data)
            while True:
                try:
                    event, data = await asyncio.wait_for(q.get(), timeout=keepalive_s)
                    yield format_sse(event, data)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            subs = self._topics.get(topic)
            if subs is not None:
                subs.discard(q)
                if not subs:
                    self._topics.pop(topic, None)
