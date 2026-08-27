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
    # Tool output settings
    tool_output_max_tokens: int = 512     # Max tokens for tool results before summarization
    tool_output_summarize: bool = True    # Enable tool result summarization
    # Per-component token budgets (percentages of available budget after response reserve)
    budget_system_prompt_pct: float = 0.05   # 5% for system prompt
    budget_summary_pct: float = 0.10         # 10% for conversation summary
    budget_history_pct: float = 0.60         # 60% for recent history
    budget_rag_pct: float = 0.15             # 15% for RAG context
    budget_tools_pct: float = 0.10           # 10% for tool results
    tool_timeout_seconds: float = 30.0
    tool_max_retries: int = 2
    # Adaptive temperature settings
    tool_call_temperature: float = 0.1      # Low temp for deterministic tool calling
    final_response_temperature: float = 0.7  # Higher temp for natural final response
    # Semantic deduplication settings
    rag_dedup_threshold: float = 0.85   # Cosine similarity threshold for deduplication
    rag_dedup_enabled: bool = True      # Enable/disable semantic deduplication

    # Multi-agent orchestration settings
    orchestrator_enabled: bool = True           # Enable multi-agent orchestration
    orchestrator_planning: bool = True          # Enable automatic planning for complex tasks
    orchestrator_review: bool = True            # Enable output review for quality
    orchestrator_force_agent: str = ""          # Force specific agent: planner/coder/researcher/reviewer/general


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
        tool_output_max_tokens=int(os.environ.get("LLM_TOOL_OUTPUT_MAX_TOKENS", "512")),
        tool_output_summarize=os.environ.get("LLM_TOOL_OUTPUT_SUMMARIZE", "true").lower() == "true",
        budget_system_prompt_pct=float(os.environ.get("LLM_BUDGET_SYSTEM_PROMPT_PCT", "0.05")),
        budget_summary_pct=float(os.environ.get("LLM_BUDGET_SUMMARY_PCT", "0.10")),
        budget_history_pct=float(os.environ.get("LLM_BUDGET_HISTORY_PCT", "0.60")),
        budget_rag_pct=float(os.environ.get("LLM_BUDGET_RAG_PCT", "0.15")),
        budget_tools_pct=float(os.environ.get("LLM_BUDGET_TOOLS_PCT", "0.10")),
        tool_timeout_seconds=float(os.environ.get("LLM_TOOL_TIMEOUT_SECONDS", "30.0")),
        tool_max_retries=int(os.environ.get("LLM_TOOL_MAX_RETRIES", "2")),
        tool_call_temperature=float(os.environ.get("LLM_TOOL_CALL_TEMPERATURE", "0.1")),
        final_response_temperature=float(os.environ.get("LLM_FINAL_RESPONSE_TEMPERATURE", "0.7")),
        rag_dedup_threshold=float(os.environ.get("LLM_RAG_DEDUP_THRESHOLD", "0.85")),
        rag_dedup_enabled=os.environ.get("LLM_RAG_DEDUP_ENABLED", "true").lower() == "true",
        orchestrator_enabled=os.environ.get("LLM_ORCHESTRATOR_ENABLED", "true").lower() == "true",
        orchestrator_planning=os.environ.get("LLM_ORCHESTRATOR_PLANNING", "true").lower() == "true",
        orchestrator_review=os.environ.get("LLM_ORCHESTRATOR_REVIEW", "true").lower() == "true",
        orchestrator_force_agent=os.environ.get("LLM_ORCHESTRATOR_FORCE_AGENT", ""),
    )


settings = load_settings()