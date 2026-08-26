"""
Unit tests for config.py
"""

import pytest
import os
from app.config import Settings, load_settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(api_key="test-key")
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.models_dir == "models"
        assert settings.reserve_for_response == 768
        assert settings.session_ttl_days == 30
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.rag_top_k == 5
        assert settings.rag_initial_k == 20
        assert settings.rag_token_budget == 1024
        assert settings.reranker_enabled is True
        assert settings.reserve_for_response_min == 256
        assert settings.reserve_for_response_max == 2048
        assert settings.tool_output_max_tokens == 512
        assert settings.tool_output_summarize is True
        assert settings.budget_system_prompt_pct == 0.05
        assert settings.budget_summary_pct == 0.10
        assert settings.budget_history_pct == 0.60
        assert settings.budget_rag_pct == 0.15
        assert settings.budget_tools_pct == 0.10

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_HOST", "127.0.0.1")
        monkeypatch.setenv("LLM_PORT", "9000")
        monkeypatch.setenv("LLM_RAG_TOP_K", "10")
        monkeypatch.setenv("LLM_RAG_TOKEN_BUDGET", "2048")
        monkeypatch.setenv("LLM_RERANKER_ENABLED", "false")
        monkeypatch.setenv("LLM_TOOL_OUTPUT_MAX_TOKENS", "256")
        monkeypatch.setenv("LLM_TOOL_OUTPUT_SUMMARIZE", "false")
        monkeypatch.setenv("LLM_BUDGET_HISTORY_PCT", "0.70")

        settings = load_settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.rag_top_k == 10
        assert settings.rag_token_budget == 2048
        assert settings.reranker_enabled is False
        assert settings.tool_output_max_tokens == 256
        assert settings.tool_output_summarize is False
        assert settings.budget_history_pct == 0.70

    def test_frozen_dataclass(self):
        settings = Settings(api_key="test")
        with pytest.raises(AttributeError):
            settings.host = "127.0.0.1"