"""
Unit tests for auth.py, cleanup.py, metrics.py
"""

from unittest.mock import MagicMock, patch
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends

from app.auth import verify_api_key
from app.cleanup import periodic_cleanup
from app.metrics import (
    init_server_info, update_llama_server_config, set_llama_health,
    set_active_sessions, record_http_request, record_session_created,
    record_session_deleted, record_session_expired, record_llama_restart,
    record_agent_round, record_agent_turn, record_tool_call,
)


class TestAuth:
    def _setup_auth(self, monkeypatch, api_key: str):
        """Helper to set up auth with a specific API key."""
        monkeypatch.setenv("LLM_API_KEY", api_key)
        import importlib
        import app.config
        importlib.reload(app.config)
        import app.auth_keys
        importlib.reload(app.auth_keys)
        import app.auth
        importlib.reload(app.auth)
        return app.auth.verify_api_key

    def test_verify_api_key_valid(self, monkeypatch):
        self._setup_auth(monkeypatch, "expected-key")
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(api_key: str = Depends(verify_api_key)):
            return {"validated": True}
        
        client = TestClient(app)
        
        # Need to create a valid key in the store first
        from app.auth_keys import init_key_store, Scope
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            temp_path = f.name
        
        try:
            store = init_key_store(temp_path)
            plaintext, _ = store.create_key(
                name="test",
                scopes=[Scope.READ, Scope.WRITE],
            )
            response = client.get("/test", headers={"X-API-Key": plaintext})
            assert response.status_code == 200
            assert response.json()["validated"] is True
        finally:
            os.unlink(temp_path)

    def test_verify_api_key_invalid(self, monkeypatch):
        self._setup_auth(monkeypatch, "expected-key")
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(api_key: str = Depends(verify_api_key)):
            return {"validated": True}
        
        client = TestClient(app)
        
        # Invalid key format
        response = client.get("/test", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_verify_api_key_missing(self, monkeypatch):
        self._setup_auth(monkeypatch, "expected-key")
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(api_key: str = Depends(verify_api_key)):
            return {"validated": True}
        
        client = TestClient(app)
        
        response = client.get("/test")
        # Header(...) makes it required, so 422 for missing header
        assert response.status_code in (401, 422)


class TestCleanup:
    @pytest.mark.asyncio
    async def test_periodic_cleanup_calls_purge(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            ttl_days=0,  # Expire immediately
            embedding_service=mock_embedding_service,
        )
        
        # Create an expired session
        session = store.create_session("device-1")
        store._sessions[session.session_id].last_active = time.time() - 86400
        store._repo.save(store._sessions)
        
        # Run cleanup once (not periodic)
        purged = store.purge_expired()
        assert purged == 1

    @pytest.mark.asyncio
    async def test_periodic_cleanup_runs(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            ttl_days=0,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        store._sessions[session.session_id].last_active = time.time() - 86400
        store._repo.save(store._sessions)
        
        # Run periodic cleanup with short interval
        import asyncio
        task = asyncio.create_task(periodic_cleanup(store, 0.01))
        await asyncio.sleep(0.05)  # Let it run a few times
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Session should be purged
        assert store.get(session.session_id) is None


class TestMetrics:
    def test_metrics_functions_exist(self):
        """Verify all metric functions are callable."""
        init_server_info("model.gguf", 4096, 512, "q8_0")
        update_llama_server_config(4096, 512)
        set_llama_health(True)
        set_active_sessions(5)
        record_http_request("GET", "/test", 200, 0.1)
        record_session_created()
        record_session_deleted()
        record_session_expired()
        record_llama_restart("test")
        record_agent_round("session-1")
        record_agent_turn("session-1", "assistant")
        record_tool_call("read_file", 0.1, True, 0)

    def test_metrics_increment(self):
        """Test that counters increment."""
        from prometheus_client import REGISTRY
        
        # Get initial values
        def get_metric(name):
            for metric in REGISTRY.collect():
                for sample in metric.samples:
                    if sample.name == name:
                        return sample.value
            return 0
        
        initial_created = get_metric("llm_sessions_created_total")
        record_session_created()
        after_created = get_metric("llm_sessions_created_total")
        
        assert after_created == initial_created + 1