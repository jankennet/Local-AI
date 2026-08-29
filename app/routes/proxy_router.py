"""
proxy_router.py

Forwards /v1/* requests to the native llama-server process running
internally on 127.0.0.1 (never LAN-exposed — it has no auth of its own).
This is the only reason a proxy exists: external clients still hit one
public port with one API key, exactly like before switching to the
native server — they don't need to know a second process exists.

Validates and transforms requests for llama.cpp compatibility before forwarding.
"""

import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, JSONResponse
import requests

from ..auth import verify_api_key
from ..llm.log_buffer import get_log_buffer

logger = logging.getLogger(__name__)

# Endpoints supported by llama.cpp's llama-server
# NOTE: embeddings is intentionally excluded — llama-server doesn't serve embeddings.
# Continue's @codebase will fall back to its local embedding provider.
SUPPORTED_ENDPOINTS = {
    "chat/completions",
    "completions",
    "models",
}

# Parameters llama.cpp doesn't support (OpenAI-specific)
UNSUPPORTED_CHAT_PARAMS = {
    "tool_choice",
    "parallel_tool_calls",
    "function_call",
    "functions",  # legacy
    "user",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "seed",
    "service_tier",
    "metadata",
    "store",
    "reasoning_effort",
}

# Condensed log format: single line per request
def _log_request_line(request_id: str, method: str, path: str, status_code: int, duration_ms: int, error: str = None) -> None:
    """Log a single condensed line for the request."""
    if error:
        logger.error(f"REQ {request_id} {method} /v1/{path} -> {status_code} ({duration_ms}ms) ERROR: {error}")
    elif status_code >= 400:
        logger.warning(f"REQ {request_id} {method} /v1/{path} -> {status_code} ({duration_ms}ms)")
    else:
        logger.info(f"REQ {request_id} {method} /v1/{path} -> {status_code} ({duration_ms}ms)")


def _dump_llama_logs_on_error(request_id: str, path: str, method: str, status_code: int, error: str = None) -> None:
    """Dump llama-server logs when request fails."""
    log_buffer = get_log_buffer()
    if not log_buffer:
        return
    
    dump = log_buffer.dump(
        prefix=f"LLAMA-SERVER LOGS FOR FAILED REQUEST {request_id}: {method} /v1/{path} -> {status_code}"
    )
    # Print directly to stderr for maximum visibility
    sys.stderr.write(dump)
    sys.stderr.flush()


def _sanitize_chat_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove OpenAI-specific parameters that llama.cpp doesn't support.
    Also normalize tool calling format for llama.cpp's --jinja template.
    """
    sanitized = {k: v for k, v in payload.items() if k not in UNSUPPORTED_CHAT_PARAMS}
    
    # Ensure tools format is compatible with llama.cpp
    if "tools" in sanitized and sanitized["tools"]:
        # llama.cpp expects tools array with type=function and function object
        # This should already be correct from OpenAI format, but ensure it's a list
        if not isinstance(sanitized["tools"], list):
            sanitized.pop("tools", None)
    
    return sanitized


def _is_supported_endpoint(path: str) -> bool:
    """Check if the endpoint is supported by llama.cpp."""
    # Remove query parameters if present
    path = path.split("?")[0]
    return path in SUPPORTED_ENDPOINTS


def build_proxy_router(base_url: str) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start_time = time.time()

        # Check if endpoint is supported
        if not _is_supported_endpoint(path):
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, path, 404, duration_ms, "UNSUPPORTED_ENDPOINT")
            return JSONResponse(
                content={"error": f"Endpoint /v1/{path} not supported. Supported: {', '.join(sorted(SUPPORTED_ENDPOINTS))}"},
                status_code=404,
                headers={"X-Request-ID": request_id},
            )

        body = await request.body()

        # Parse and sanitize JSON payload for chat/completions
        if request.method == "POST" and path in ("chat/completions", "completions") and body:
            try:
                payload = json.loads(body)
                payload = _sanitize_chat_payload(payload)
                body = json.dumps(payload).encode("utf-8")
            except json.JSONDecodeError:
                pass  # Forward as-is if not valid JSON

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
                _log_request_line(request_id, request.method, path, upstream.status_code, duration_ms)
                _dump_llama_logs_on_error(request_id, path, request.method, upstream.status_code)
            else:
                _log_request_line(request_id, request.method, path, upstream.status_code, duration_ms)

            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                headers={"X-Request-ID": request_id},
            )

        except requests.exceptions.Timeout:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, path, 504, duration_ms, "TIMEOUT")
            _dump_llama_logs_on_error(request_id, path, request.method, 504, "TIMEOUT")
            return JSONResponse(
                content={"error": "Upstream timeout"},
                status_code=504,
                headers={"X-Request-ID": request_id},
            )
        except requests.exceptions.ConnectionError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, path, 503, duration_ms, f"CONNECTION_ERROR: {e}")
            _dump_llama_logs_on_error(request_id, path, request.method, 503, f"CONNECTION_ERROR: {e}")
            return JSONResponse(
                content={"error": "Upstream unavailable"},
                status_code=503,
                headers={"X-Request-ID": request_id},
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, path, 500, duration_ms, f"EXCEPTION: {e}")
            _dump_llama_logs_on_error(request_id, path, request.method, 500, f"EXCEPTION: {e}")
            return JSONResponse(
                content={"error": "Internal proxy error"},
                status_code=500,
                headers={"X-Request-ID": request_id},
            )

    return router