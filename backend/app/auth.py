"""Authentication and short-lived HMAC stream-ticket manager for ExhibitFlow backend."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from fastapi import HTTPException, Query, Request


class AuthManager:
    """Constant-time API key verification and process-scoped HMAC stream tickets.

    - Header authentication accepts ONLY ``Authorization: Bearer <key>`` or
      ``X-Backend-Key: <key>`` verified against ``BACKEND_API_KEY``.
    - Explicitly NEVER accepts or falls back to ``GPU_TOKEN`` / ``X-GPU-Token``.
    - Stream tickets (``?st=<exp_int>.<sig_hex>``) are signed with
      ``sha256((backend_api_key + ":ef-stream:" + process_nonce).encode())``
      so the master ``BACKEND_API_KEY`` never appears in query strings or URLs.
    """

    def __init__(self, api_key: str = "", stream_ttl_s: int = 600) -> None:
        self.api_key = (api_key or "").strip()
        self.stream_ttl_s = max(1, int(stream_ttl_s))
        self._process_nonce = secrets.token_hex(16)
        self._stream_secret = hashlib.sha256(
            f"{self.api_key}:ef-stream:{self._process_nonce}".encode("utf-8")
        ).digest()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def extract_header_key(request: Request) -> str | None:
        auth_hdr = request.headers.get("authorization") or ""
        if auth_hdr.lower().startswith("bearer "):
            tok = auth_hdr[7:].strip()
            if tok:
                return tok
        x_key = (request.headers.get("x-backend-key") or "").strip()
        return x_key or None

    def verify_header_token(self, token: str | None) -> bool:
        if not self.enabled:
            return True
        if not token:
            return False
        return hmac.compare_digest(token.encode("utf-8"), self.api_key.encode("utf-8"))

    def verify_request_header(self, request: Request) -> bool:
        return self.verify_header_token(self.extract_header_key(request))

    def issue_stream_ticket(self, ttl_s: int | None = None) -> dict[str, Any]:
        ttl = self.stream_ttl_s if ttl_s is None else max(1, int(ttl_s))
        expires_at = int(time.time()) + ttl
        if not self.enabled:
            return {
                "ticket": "",
                "expires_at": float(expires_at),
                "expires_in": ttl,
                "auth_required": False,
            }
        exp_str = str(expires_at)
        sig_hex = hmac.new(
            self._stream_secret, exp_str.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return {
            "ticket": f"{exp_str}.{sig_hex}",
            "expires_at": float(expires_at),
            "expires_in": ttl,
            "auth_required": True,
        }

    def verify_stream_ticket(self, ticket: str | None) -> bool:
        if not self.enabled:
            return True
        if not ticket or "." not in ticket:
            return False
        exp_str, sig_hex = ticket.split(".", 1)
        if not exp_str.isdigit() or not sig_hex:
            return False
        if time.time() > int(exp_str):
            return False
        expected = hmac.new(
            self._stream_secret, exp_str.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(sig_hex.lower(), expected)


async def require_header_auth(request: Request) -> None:
    """FastAPI dependency enforcing header-only Backend API key authentication."""
    auth: AuthManager = request.app.state.auth
    if not auth.enabled:
        return
    if not auth.verify_request_header(request):
        raise HTTPException(
            status_code=401,
            detail="Cần xác thực Backend (Authorization: Bearer hoặc X-Backend-Key)",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_header_or_stream_ticket(
    request: Request,
    st: str | None = Query(None),
) -> None:
    """FastAPI dependency allowing header auth OR a valid short-lived ?st= stream ticket."""
    auth: AuthManager = request.app.state.auth
    if not auth.enabled:
        return
    if auth.verify_request_header(request):
        return
    if st and auth.verify_stream_ticket(st):
        return
    raise HTTPException(
        status_code=401,
        detail="Cần xác thực Backend hoặc stream ticket (?st=) hợp lệ",
        headers={"WWW-Authenticate": "Bearer"},
    )
