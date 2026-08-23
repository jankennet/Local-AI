"""
proxy_router.py

Forwards /v1/* requests to the native llama-server process running
internally on 127.0.0.1 (never LAN-exposed — it has no auth of its own).
This is the only reason a proxy exists: external clients still hit one
public port with one API key, exactly like before switching to the
native server — they don't need to know a second process exists.
"""

import json
import logging
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
import requests

from ..auth import verify_api_key

logger = logging.getLogger(__name__)


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
        
        if upstream.status_code >= 400:
            try:
                body_str = body.decode("utf-8")
                body_json = json.loads(body_str) if body_str else {}
                logger.warning(
                    "Upstream request failed",
                    extra={
                        "path": path,
                        "method": request.method,
                        "status_code": upstream.status_code,
                        "request_body": body_json,
                        "response_body": upstream.text,
                    },
                )
            except Exception as e:
                logger.warning(
                    "Upstream request failed (could not parse body)",
                    extra={
                        "path": path,
                        "method": request.method,
                        "status_code": upstream.status_code,
                        "raw_body": body[:500] if body else None,
                        "response_body": upstream.text,
                        "parse_error": str(e),
                    },
                )
        
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return router