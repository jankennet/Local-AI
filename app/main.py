"""
main.py

Composition root. This is the ONLY file that knows about concrete
implementations (JSONSessionRepository, LlamaVocabTokenCounter,
SummarizeOldestStrategy, etc.) — everything else in the app depends on
interfaces/abstractions and gets its collaborators handed to it here.

If you want to swap JSON storage for SQLite, or summarization for plain
drop-oldest, this is the one place you touch.
"""

import asyncio
from contextlib import asynccontextmanager

from .config import settings
from .cleanup import periodic_cleanup
from .tokenizer import LlamaVocabTokenCounter
from .sessions.repository import JSONSessionRepository
from .sessions.eviction import SummarizeOldestStrategy
from .sessions.store import SessionStore
from .llm.gpu_detect import detect_gpu, get_vram_tier
from .llm.catalog import get_existing_models, download_model, MODEL_CATALOG
from .llm.server_launcher import build_llama_app
from .llm.completion_client import LoopbackCompletionClient
from .routes.sessions_router import build_sessions_router


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


def create_app():
    model_path, vram_tier = resolve_model_path()
    llama_app, selected_config = build_llama_app(model_path, vram_tier)
    print(f"Model server ready — n_ctx={selected_config['n_ctx']}, "
          f"n_batch={selected_config['n_batch']}")

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
    completion_client = LoopbackCompletionClient(settings.port)

    router = build_sessions_router(store, completion_client)
    llama_app.include_router(router)

    cleanup_task_holder = {}

    @llama_app.on_event("startup")
    async def start_cleanup():
        cleanup_task_holder["task"] = asyncio.create_task(
            periodic_cleanup(store, settings.cleanup_interval_seconds)
        )

    @llama_app.on_event("shutdown")
    async def stop_cleanup():
        task = cleanup_task_holder.get("task")
        if task:
            task.cancel()

    return llama_app


def run():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
