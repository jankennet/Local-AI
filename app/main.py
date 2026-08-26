"""
main.py

Composition root. This is the ONLY file that knows about concrete
implementations (JSONSessionRepository, LlamaVocabTokenCounter,
SummarizeOldestStrategy, the tool registry, etc.) — everything else in
the app depends on interfaces/abstractions and gets its collaborators
handed to it here.

The model server is now the native llama-server binary, run as a
subprocess bound to 127.0.0.1 (see server_launcher.py for why: it reads
each GGUF's own chat template, so tool-calling generalizes across model
families instead of needing a hand-picked format per model). This app
is the only thing bound to 0.0.0.0 — /v1/* is proxied through to the
internal process so external clients see the same single port + API key
as before.
"""

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .config import settings
from .cleanup import periodic_cleanup
from .tokenizer import LlamaVocabTokenCounter
from .sessions.repository import JSONSessionRepository
from .sessions.eviction import SummarizeOldestStrategy
from .sessions.store import SessionStore
from .embeddings import EmbeddingService, create_vector_store
from .llm.gpu_detect import detect_gpu, get_vram_tier
from .llm.catalog import get_existing_models, download_model, MODEL_CATALOG
from .llm.server_launcher import launch_llama_server, terminate_process
from .llm.watchdog import watch_llama_server
from .llm.log_buffer import LogBuffer, set_log_buffer
from .llm.completion_client import LoopbackCompletionClient
from .llm.tools import TOOLS
from .routes.sessions_router import build_sessions_router
from .routes.proxy_router import build_proxy_router
from .routes.debug_router import build_debug_router
from .llm.watchdog import watch_llama_server
from .metrics import (
    init_server_info, update_llama_server_config, set_llama_health,
    set_active_sessions, record_http_request, record_session_created,
    record_session_deleted, record_session_expired, record_llama_restart,
)
from prometheus_client import make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_model_path() -> tuple[str, str]:
    """Returns (model_path, vram_tier). Uses an existing .gguf if present,
    otherwise detects hardware and downloads the top recommendation for
    that tier (non-interactive — this runs as a background service)."""
    vendor, gpu_name, vram_gb = detect_gpu()
    vram_tier = get_vram_tier(vram_gb)
    print(f"Detected: {vendor} {gpu_name} (~{vram_gb}GB) -> tier {vram_tier}")

    existing = get_existing_models(settings.models_dir)
    if existing:
        return existing[0], vram_tier

    first_choice = next(iter(MODEL_CATALOG[vram_tier]))
    print(f"No local model found — downloading default for {vram_tier}: {first_choice}")
    path = download_model(settings.models_dir, vram_tier, first_choice)
    return path, vram_tier


def create_app() -> FastAPI:
    model_path, vram_tier = resolve_model_path()
    
    # Create log buffer for llama-server output
    log_buffer = LogBuffer(max_lines=100)
    set_log_buffer(log_buffer)
    
    process, base_url, selected_config = launch_llama_server(
        settings.llama_server_bin, model_path, "127.0.0.1", settings.internal_port, vram_tier,
        log_buffer
    )
    kv_note = "q8_0 (quantized)" if selected_config["kv_quant"] else "f16 (default)"
    print(f"Native llama-server ready at {base_url} — n_ctx={selected_config['n_ctx']}, "
          f"n_batch={selected_config['n_batch']}, kv_cache={kv_note}")
    print("Llama-server logs captured to rolling buffer (last 100 lines). Errors will dump buffer.")

    counter = LlamaVocabTokenCounter(model_path)
    repository = JSONSessionRepository(settings.sessions_file)
    eviction = SummarizeOldestStrategy()
    embedding_service = EmbeddingService(settings.embedding_model_code)
    vector_store_factory = lambda es: create_vector_store(
        es,
        backend=settings.vector_backend,
        path=settings.vector_db_path,
        collection_name=settings.vector_collection,
    )
    store = SessionStore(
        counter=counter,
        repository=repository,
        eviction=eviction,
        n_ctx=selected_config["n_ctx"],
        reserve_for_response=settings.reserve_for_response,
        ttl_days=settings.session_ttl_days,
        embedding_service=embedding_service,
        vector_store_factory=vector_store_factory,
    )
    completion_client = LoopbackCompletionClient(base_url, model_path)

    # Mutable holders — the watchdog replaces these in place if it has to
    # relaunch the process, so completion_client (which already points at
    # a fixed host:port, unaffected by a relaunch) doesn't need to change.
    process_holder = {"process": process}
    config_holder = {"config": selected_config}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cleanup_task = asyncio.create_task(periodic_cleanup(store, settings.cleanup_interval_seconds))
        watchdog_task = asyncio.create_task(watch_llama_server(
            settings.llama_server_bin, model_path, "127.0.0.1", settings.internal_port,
            vram_tier, process_holder, config_holder,
        ))
        yield
        cleanup_task.cancel()
        watchdog_task.cancel()
        terminate_process(process_holder["process"])  # stop the native llama-server subprocess too

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_sessions_router(
        store,
        completion_client,
        TOOLS,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        tool_max_retries=settings.tool_max_retries,
    ))
    app.include_router(build_proxy_router(base_url))
    app.include_router(build_debug_router(
        store, embedding_service,
        settings.vector_backend, settings.vector_db_path, settings.vector_collection,
    ))

    # Prometheus metrics endpoint
    # Use a route instead of mount to avoid trailing slash issues
    from starlette.responses import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/stats", include_in_schema=False)
    async def stats():
        """Human-readable key metrics summary."""
        # Query Prometheus registry directly for current values
        def get_metric(name: str, labels: dict = None):
            labels = labels or {}
            for metric in REGISTRY.collect():
                for sample in metric.samples:
                    if sample.name == name and sample.labels == labels:
                        return sample.value
            return None

        def get_metric_by_prefix(prefix: str):
            results = {}
            for metric in REGISTRY.collect():
                for sample in metric.samples:
                    if sample.name.startswith(prefix):
                        key = sample.name
                        if sample.labels:
                            key += "{" + ",".join(f'{k}="{v}"' for k, v in sample.labels.items()) + "}"
                        results[key] = sample.value
            return results

        return {
            "server": {
                "status": "healthy" if get_metric("llm_llama_server_health") == 1 else "degraded",
                "model": model_path,
                "n_ctx": int(get_metric("llm_llama_server_n_ctx") or 0),
                "n_batch": int(get_metric("llm_llama_server_n_batch") or 0),
            },
            "sessions": {
                "active": int(get_metric("llm_active_sessions") or 0),
                "created_total": int(get_metric("llm_sessions_created_total") or 0),
                "expired_total": int(get_metric("llm_sessions_expired_total") or 0),
            },
            "http": {
                "requests_total": int(sum(v for k, v in get_metric_by_prefix("llm_http_requests_total").items())),
            },
            "tools": get_metric_by_prefix("llm_tool_calls_total"),
            "tokens": {
                "prompt_total": int(sum(v for k, v in get_metric_by_prefix("llm_completion_tokens_total").items() if 'type="prompt"' in k)),
                "completion_total": int(sum(v for k, v in get_metric_by_prefix("llm_completion_tokens_total").items() if 'type="completion"' in k)),
            },
        }

    # Initialize server info metrics
    init_server_info(model_path, selected_config["n_ctx"], selected_config["n_batch"],
                     "q8_0" if selected_config["kv_quant"] else "f16")
    update_llama_server_config(selected_config["n_ctx"], selected_config["n_batch"])

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        import time
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        record_http_request(
            request.method, request.url.path, response.status_code, duration
        )
        return response

    @app.get("/health")
    async def health():
        llama_healthy = False
        try:
            import requests
            resp = requests.get(f"{base_url}/health", timeout=2)
            llama_healthy = resp.status_code == 200
        except Exception:
            pass

        set_llama_health(llama_healthy)
        set_active_sessions(len(store.list_sessions()))

        return {
            "status": "healthy" if llama_healthy else "degraded",
            "llama_server": "healthy" if llama_healthy else "unhealthy",
            "model": model_path,
            "n_ctx": selected_config["n_ctx"],
            "n_batch": selected_config["n_batch"],
            "kv_cache": "q8_0" if selected_config["kv_quant"] else "f16",
            "active_sessions": len(store.list_sessions()),
        }

    return app


def run():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()