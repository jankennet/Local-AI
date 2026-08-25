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
import time
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
import requests

from ..auth import verify_api_key

logger = logging.getLogger(__name__)


def build_proxy_router(base_url: str) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start_time = time.time()

        body = await request.body()
        client_host = request.client.host if request.client else "unknown"

        logger.info(
            "Proxy request started",
            extra={
                "request_id": request_id,
                "path": path,
                "method": request.method,
                "client": client_host,
                "body_size": len(body),
            },
        )

        try:
            upstream = requests.request(
                method=request.method,
                url=f"{base_url}/v1/{path}",
                params=request.query_params,
                data=body,
                headers={"content-type": request.headers.get("content-type", "application/json")},
                timeout=300,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if upstream.status_code >= 400:
                try:
                    body_str = body.decode("utf-8")
                    body_json = json.loads(body_str) if body_str else {}
                    logger.warning(
                        "Upstream request failed",
                        extra={
                            "request_id": request_id,
                            "path": path,
                            "method": request.method,
                            "status_code": upstream.status_code,
                            "duration_ms": duration_ms,
                            "request_body": body_json,
                            "response_body": upstream.text[:500],
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Upstream request failed (could not parse body)",
                        extra={
                            "request_id": request_id,
                            "path": path,
                            "method": request.method,
                            "status_code": upstream.status_code,
                            "duration_ms": duration_ms,
                            "raw_body": body[:500] if body else None,
                            "response_body": upstream.text[:500],
                            "parse_error": str(e),
                        },
                    )
            else:
                logger.info(
                    "Proxy request completed",
                    extra={
                        "request_id": request_id,
                        "path": path,
                        "method": request.method,
                        "status_code": upstream.status_code,
                        "duration_ms": duration_ms,
                        "response_size": len(upstream.content),
                    },
                )

            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                headers={"X-Request-ID": request_id},
            )

        except requests.exceptions.Timeout:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Upstream request timeout",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": request.method,
                    "duration_ms": duration_ms,
                },
            )
            return Response(
                content=json.dumps({"error": "Upstream timeout"}),
                status_code=504,
                media_type="application/json",
                headers={"X-Request-ID": request_id},
            )
        except requests.exceptions.ConnectionError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Upstream connection error",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": request.method,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
            )
            return Response(
                content=json.dumps({"error": "Upstream unavailable"}),
                status_code=503,
                media_type="application/json",
                headers={"X-Request-ID": request_id},
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(
                "Proxy request error",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": request.method,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
            )
            return Response(
                content=json.dumps({"error": "Internal proxy error"}),
                status_code=500,
                media_type="application/json",
                headers={"X-Request-ID": request_id},
            )

    return router