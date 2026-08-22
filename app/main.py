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
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .cleanup import periodic_cleanup
from .tokenizer import LlamaVocabTokenCounter
from .sessions.repository import JSONSessionRepository
from .sessions.eviction import SummarizeOldestStrategy
from .sessions.store import SessionStore
from .llm.gpu_detect import detect_gpu, get_vram_tier
from .llm.catalog import get_existing_models, download_model, MODEL_CATALOG
from .llm.server_launcher import launch_llama_server, terminate_process
from .llm.completion_client import LoopbackCompletionClient
from .llm.tools import TOOLS
from .routes.sessions_router import build_sessions_router
from .routes.proxy_router import build_proxy_router


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
    process, base_url, selected_config = launch_llama_server(
        settings.llama_server_bin, model_path, "127.0.0.1", settings.internal_port, vram_tier
    )
    kv_note = "q8_0 (quantized)" if selected_config["kv_quant"] else "f16 (default)"
    print(f"Native llama-server ready at {base_url} — n_ctx={selected_config['n_ctx']}, "
          f"n_batch={selected_config['n_batch']}, kv_cache={kv_note}")

    counter = LlamaVocabTokenCounter(model_path)
    repository = JSONSessionRepository(settings.sessions_file)
    eviction = SummarizeOldestStrategy()
    store = SessionStore(
        counter=counter,
        repository=repository,
        eviction=eviction,
        n_ctx=selected_config["n_ctx"],
        reserve_for_response=settings.reserve_for_response,
        ttl_days=settings.session_ttl_days,
    )
    completion_client = LoopbackCompletionClient(base_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cleanup_task = asyncio.create_task(periodic_cleanup(store, settings.cleanup_interval_seconds))
        yield
        cleanup_task.cancel()
        terminate_process(process)  # stop the native llama-server subprocess too

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_sessions_router(store, completion_client, TOOLS))
    app.include_router(build_proxy_router(base_url))
    return app


def run():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()