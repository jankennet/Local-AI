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


@dataclass(frozen=True)
class Settings:
    api_key: str
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: str = "models"
    sessions_file: str = "sessions.json"
    reserve_for_response: int = 768
    session_ttl_days: int = 30
    cleanup_interval_seconds: int = 3600
    llama_server_bin: str = "llama.cpp/build/bin/llama-server"
    internal_port: int = 8081
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model_code: str = "microsoft/codebert-base"
    vector_backend: str = "qdrant"
    vector_db_path: str = "./qdrant_db"
    vector_collection: str = "conversations"


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
        session_ttl_days=int(os.environ.get("LLM_SESSION_TTL_DAYS", "30")),
        cleanup_interval_seconds=int(os.environ.get("LLM_CLEANUP_INTERVAL_SECONDS", "3600")),
        llama_server_bin=os.environ.get("LLM_LLAMA_SERVER_BIN", "llama.cpp/build/bin/llama-server"),
        internal_port=int(os.environ.get("LLM_INTERNAL_PORT", "8081")),
        embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_model_code=os.environ.get("LLM_EMBEDDING_MODEL_CODE", "microsoft/codebert-base"),
        vector_backend=os.environ.get("LLM_VECTOR_BACKEND", "qdrant"),
        vector_db_path=os.environ.get("LLM_VECTOR_DB_PATH", "./qdrant_db"),
        vector_collection=os.environ.get("LLM_VECTOR_COLLECTION", "conversations"),
    )


settings = load_settings()