"""
auth.py

API key enforcement. One job: verify the caller is allowed in.
Used as a FastAPI dependency on every route that shouldn't be public.
"""

from fastapi import Header, HTTPException
from .config import settings


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
