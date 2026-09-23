"""
auth.py

API key enforcement. One job: verify the caller is allowed in.
Used as a FastAPI dependency on every route that shouldn't be public.

Accepts either the custom ``X-API-Key`` header or the standard
``Authorization: Bearer <key>`` header so that OpenAI-compatible
clients work out-of-the-box.
"""

from fastapi import Header, HTTPException, WebSocket
from .config import settings


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> None:
    key = _extract_bearer_token(authorization) or x_api_key
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def verify_api_key_ws(api_key: str) -> bool:
    """Verify API key for WebSocket connections. Returns True if valid."""
    return api_key == settings.api_key
