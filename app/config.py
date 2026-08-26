"""
config.py

Single source of truth for runtime settings, loaded from environment
variables. Nothing else in the app reads os.environ directly — if a
setting needs to change (e.g. swap the .env approach for a secrets
manager later), this is the only file that touches.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (created by setup.sh/setup.bat)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: str = "models"
    sessions_file: str = "sessions.json"
    reserve_for_response: int = 768
    reserve_for_response_min: int = 256      # Minimum reserve for short queries
    reserve_for_response_max: int = 2048     # Maximum reserve for complex queries
    session_ttl_days: int = 30
    cleanup_interval_seconds: int = 3600
    llama_server_bin: str = "llama.cpp/build/bin/llama-server"
    internal_port: int = 8081
    # Embedding models (CPU-bound). For old CPUs without AVX2 (e.g. Athlon X4 860k):
    # - sentence-transformers/all-MiniLM-L6-v2 (22M, fastest, no AVX2 needed)
    # - BAAI/bge-small-en-v1.5 (33M, good quality, no AVX2 needed)
    # - intfloat/multilingual-e5-small (33M, multilingual, no AVX2 needed)
    # Avoid: bge-base/large, e5-base/large, nomic-embed-text (need AVX2 or slow)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_code: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_backend: str = "qdrant"
    vector_db_path: str = "./qdrant_db"
    vector_collection: str = "conversations"
    # RAG settings
    rag_top_k: int = 5                    # Final results after reranking
    rag_initial_k: int = 20               # Initial vector search breadth
    rag_token_budget: int = 1024          # Max tokens for retrieved context
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight, no AVX2
    reranker_enabled: bool = True         # Disable if too slow on CPU
    tool_timeout_seconds: float = 30.0
    tool_max_retries: int = 2


def load_settings() -> Settings:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit(
            "LLM_API_KEY is not set. Since this server is reachable from your "
            "whole network, an API key is required.\n"
            "Set it before launching, e.g.:\n"
            "  export LLM_API_KEY=\"$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\"\n"
        )

    return Settings(
        api_key=api_key,
        host=os.environ.get("LLM_HOST", "0.0.0.0"),
        port=int(os.environ.get("LLM_PORT", "8000")),
        models_dir=os.environ.get("LLM_MODELS_DIR", "models"),
        sessions_file=os.environ.get("LLM_SESSIONS_FILE", "sessions.json"),
        reserve_for_response=int(os.environ.get("LLM_RESERVE_FOR_RESPONSE", "768")),
        reserve_for_response_min=int(os.environ.get("LLM_RESERVE_FOR_RESPONSE_MIN", "256")),
        reserve_for_response_max=int(os.environ.get("LLM_RESERVE_FOR_RESPONSE_MAX", "2048")),
        session_ttl_days=int(os.environ.get("LLM_SESSION_TTL_DAYS", "30")),
        cleanup_interval_seconds=int(os.environ.get("LLM_CLEANUP_INTERVAL_SECONDS", "3600")),
        llama_server_bin=os.environ.get("LLM_LLAMA_SERVER_BIN", "llama.cpp/build/bin/llama-server"),
        internal_port=int(os.environ.get("LLM_INTERNAL_PORT", "8081")),
        embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_model_code=os.environ.get("LLM_EMBEDDING_MODEL_CODE", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_backend=os.environ.get("LLM_VECTOR_BACKEND", "qdrant"),
        vector_db_path=os.environ.get("LLM_VECTOR_DB_PATH", "./qdrant_db"),
        vector_collection=os.environ.get("LLM_VECTOR_COLLECTION", "conversations"),
        rag_top_k=int(os.environ.get("LLM_RAG_TOP_K", "5")),
        rag_initial_k=int(os.environ.get("LLM_RAG_INITIAL_K", "20")),
        rag_token_budget=int(os.environ.get("LLM_RAG_TOKEN_BUDGET", "1024")),
        reranker_model=os.environ.get("LLM_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        reranker_enabled=os.environ.get("LLM_RERANKER_ENABLED", "true").lower() == "true",
        tool_timeout_seconds=float(os.environ.get("LLM_TOOL_TIMEOUT_SECONDS", "30.0")),
        tool_max_retries=int(os.environ.get("LLM_TOOL_MAX_RETRIES", "2")),
    )


settings = load_settings()