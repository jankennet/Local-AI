"""
proxy_router.py

Forwards /v1/* requests to the native llama-server process running
internally on 127.0.0.1 (never LAN-exposed — it has no auth of its own).
This is the only reason a proxy exists: external clients still hit one
public port with one API key, exactly like before switching to the
native server — they don't need to know a second process exists.

Validates and transforms requests for llama.cpp compatibility before forwarding.
"""

import asyncio
import base64
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, JSONResponse
import httpx

from ..auth import verify_api_key
from ..config import settings
from ..llm.log_buffer import get_log_buffer

logger = logging.getLogger(__name__)

# Endpoints forwarded to llama.cpp's llama-server.
# NOTE: embeddings is served in-process from the local EmbeddingService
# (llama-server can't do it — it loads a chat GGUF, wrong dimensions).
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


def build_proxy_router(base_url: str, embedding_service: Optional[Any] = None) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])
    
    # Shared async HTTP client with connection pooling
    client = httpx.AsyncClient(
        base_url=f"{base_url}/v1",
        timeout=httpx.Timeout(300.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    # ---- /v1/embeddings (served in-process, never forwarded) -------------

    @router.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start_time = time.time()

        if embedding_service is None:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, "embeddings", 503, duration_ms, "EMBEDDINGS_UNAVAILABLE")
            return JSONResponse(
                content={"error": "Embeddings are not available on this server (no embedding model loaded)."},
                status_code=503,
                headers={"X-Request-ID": request_id},
            )
        if not settings.embeddings_enabled:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, "embeddings", 404, duration_ms, "EMBEDDINGS_DISABLED")
            return JSONResponse(
                content={"error": "Embeddings are disabled on this server (set LLM_EMBEDDINGS_ENABLED=true to enable)."},
                status_code=404,
                headers={"X-Request-ID": request_id},
            )

        try:
            body = await request.json()
        except Exception:
            body = None

        if not body or not isinstance(body, dict):
            return JSONResponse(
                content={"error": "Invalid request body — expected a JSON object."},
                status_code=400,
                headers={"X-Request-ID": request_id},
            )

        raw_input = body.get("input")
        if raw_input is None:
            return JSONResponse(
                content={"error": "Missing required field 'input'."},
                status_code=400,
                headers={"X-Request-ID": request_id},
            )

        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list) and all(isinstance(t, str) for t in raw_input):
            inputs = raw_input
        else:
            return JSONResponse(
                content={"error": "'input' must be a string or an array of strings (token-ID inputs are not supported)."},
                status_code=400,
                headers={"X-Request-ID": request_id},
            )

        if not inputs:
            return JSONResponse(
                content={"error": "'input' must not be an empty array."},
                status_code=400,
                headers={"X-Request-ID": request_id},
            )

        model = body.get("model") or embedding_service.model_name
        encoding_format = body.get("encoding_format", "float")

        try:
            # Embedding is CPU-bound — run it off the event loop.
            vectors = await asyncio.to_thread(embedding_service.embed, inputs)
        except Exception as e:
            logger.exception("Embeddings request failed")
            _log_request_line(request_id, request.method, "embeddings", 500, 0, "EMBED_FAILURE")
            return JSONResponse(
                content={"error": "Failed to compute embeddings."},
                status_code=500,
                headers={"X-Request-ID": request_id},
            )

        if encoding_format == "base64":
            import numpy as np
            encoded = [
                {"object": "embedding", "index": i, "embedding": base64.b64encode(np.asarray(vec).astype("float32").tobytes()).decode("ascii")}
                for i, vec in enumerate(vectors)
            ]
        else:
            encoded = [
                {"object": "embedding", "index": i, "embedding": [round(float(x), 6) for x in vec]}
                for i, vec in enumerate(vectors)
            ]

        prompt_tokens = sum(max(1, len(t) // 4) for t in inputs if t)
        payload = {
            "object": "list",
            "data": encoded,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

        duration_ms = int((time.time() - start_time) * 1000)
        _log_request_line(request_id, request.method, "embeddings", 200, duration_ms)
        return JSONResponse(content=payload, headers={"X-Request-ID": request_id})

    # ---- proxy to llama-server -------------------------------------------

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
            upstream = await client.request(
                method=request.method,
                url=f"/{path}",
                params=request.query_params,
                content=body,
                headers={"content-type": request.headers.get("content-type", "application/json")},
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

        except httpx.TimeoutException:
            duration_ms = int((time.time() - start_time) * 1000)
            _log_request_line(request_id, request.method, path, 504, duration_ms, "TIMEOUT")
            _dump_llama_logs_on_error(request_id, path, request.method, 504, "TIMEOUT")
            return JSONResponse(
                content={"error": "Upstream timeout"},
                status_code=504,
                headers={"X-Request-ID": request_id},
            )
        except httpx.ConnectError as e:
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
        
    # Cleanup on shutdown
    @router.on_event("shutdown")
    async def shutdown():
        await client.aclose()

    return router