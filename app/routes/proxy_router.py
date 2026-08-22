"""
proxy_router.py

Forwards /v1/* requests to the native llama-server process running
internally on 127.0.0.1 (never LAN-exposed — it has no auth of its own).
This is the only reason a proxy exists: external clients still hit one
public port with one API key, exactly like before switching to the
native server — they don't need to know a second process exists.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
import requests

from ..auth import verify_api_key


def build_proxy_router(base_url: str) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        body = await request.body()
        upstream = requests.request(
            method=request.method,
            url=f"{base_url}/v1/{path}",
            params=request.query_params,
            data=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
            timeout=300,
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return router