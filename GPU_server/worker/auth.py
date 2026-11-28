"""X-GPU-Token authentication helpers for internal GPU worker endpoints."""
import hmac
import os
from typing import Optional

from fastapi import HTTPException, Request, WebSocket


def resolve_token(explicit_token: Optional[str] = None) -> str:
    """Resolve configured GPU token (explicit argument takes precedence over GPU_TOKEN env)."""
    if explicit_token is not None:
        return str(explicit_token)
    return os.environ.get("GPU_TOKEN", "")


def verify_token_value(provided: Optional[str], expected: str) -> bool:
    """Constant-time comparison using hmac.compare_digest. Empty expected disables auth."""
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


async def require_gpu_token(request: Request) -> None:
    """FastAPI dependency enforcing X-GPU-Token on all /internal/* HTTP routes."""
    expected = getattr(request.app.state, "gpu_token", "") or ""
    if not expected:
        return
    provided = request.headers.get("X-GPU-Token") or request.headers.get("x-gpu-token")
    if not verify_token_value(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-GPU-Token")


async def verify_ws_token(ws: WebSocket) -> bool:
    """Verify X-GPU-Token header or ?token= query param on WebSocket connections."""
    expected = getattr(ws.app.state, "gpu_token", "") or ""
    if not expected:
        return True
    provided = (
        ws.headers.get("X-GPU-Token")
        or ws.headers.get("x-gpu-token")
        or ws.query_params.get("token")
    )
    if not verify_token_value(provided, expected):
        await ws.close(code=4401, reason="Unauthorized")
        return False
    return True
