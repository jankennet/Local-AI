"""
Unit tests for routes (sessions_router.py, proxy_router.py, debug_router.py)
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routes.sessions_router import build_sessions_router
from app.routes.proxy_router import build_proxy_router
from app.routes.debug_router import build_debug_router
from app.auth import verify_api_key
from app.models.schemas import (
    NewSessionRequest, NewSessionResponse,
    ChatRequest, ChatResponse, SessionInfo,
)


class TestSessionsRouter:
    @pytest.fixture
    def app(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.completion_client import CompletionClient
        
        # Mock completion client - async now
        call_count = 0
        async def mock_complete_with_tools(messages, tool_schemas, max_tokens, temperature):
            nonlocal call_count
            call_count += 1
            return {
                "content": "Test response",
                "tool_calls": None,
            }
        
        mock_client = MagicMock(spec=CompletionClient)
        mock_client.complete_with_tools = mock_complete_with_tools
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        app = FastAPI()
        router = build_sessions_router(
            store=store,
            completion_client=mock_client,
            tools={},
            tool_timeout_seconds=30.0,
            tool_max_retries=2,
        )
        app.include_router(router)
        
        # Override auth dependency
        async def mock_verify():
            return "test-key"
        app.dependency_overrides[verify_api_key] = mock_verify
        
        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_create_session(self, client):
        response = client.post(
            "/sessions",
            json={"device_name": "test-device", "system_prompt": "Be helpful"},
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_create_session_requires_api_key(self, app):
        """Test that API key is required."""
        client = TestClient(app)
        # Remove the override
        app.dependency_overrides.clear()
        
        response = client.post(
            "/sessions",
            json={"device_name": "test-device"},
        )
        
        # Header(...) makes it required, so 422 for missing header
        assert response.status_code in (401, 422)

    def test_list_sessions(self, client):
        # Create a session first
        client.post("/sessions", json={"device_name": "device-1"}, headers={"X-API-Key": "test-key"})
        
        response = client.get("/sessions", headers={"X-API-Key": "test-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["device_name"] == "device-1"

    def test_delete_session(self, client):
        # Create a session
        create_resp = client.post("/sessions", json={"device_name": "device-1"}, headers={"X-API-Key": "test-key"})
        session_id = create_resp.json()["session_id"]
        
        # Delete it
        response = client.delete(f"/sessions/{session_id}", headers={"X-API-Key": "test-key"})
        
        assert response.status_code == 200
        assert response.json()["deleted"] == session_id
        
        # Verify it's gone
        list_resp = client.get("/sessions", headers={"X-API-Key": "test-key"})
        assert len(list_resp.json()) == 0

    def test_chat(self, client, monkeypatch):
        # Create a session
        create_resp = client.post("/sessions", json={"device_name": "device-1"}, headers={"X-API-Key": "test-key"})
        session_id = create_resp.json()["session_id"]
        
        # The completion_client in the router is a mock, but we need to ensure it returns proper response
        # The mock is already set up in the app fixture
        response = client.post(
            f"/sessions/{session_id}/chat",
            json={"message": "Hello", "max_tokens": 100, "temperature": 0.7},
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["reply"] == "Test response"
        assert "context_used" in data
        assert "context_limit" in data

    def test_chat_nonexistent_session(self, client):
        response = client.post(
            "/sessions/nonexistent/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 404


class TestProxyRouter:
    @pytest.fixture
    def app(self):
        app = FastAPI()
        router = build_proxy_router("http://localhost:8081")
        app.include_router(router)
        
        async def mock_verify():
            return "test-key"
        app.dependency_overrides[verify_api_key] = mock_verify
        
        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_models_endpoint(self, client):
        with patch("app.routes.proxy_router.requests.request") as mock_request:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "model-1"}]}
            mock_resp.content = b'{"data": [{"id": "model-1"}]}'
            mock_resp.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_resp
            
            response = client.get("/v1/models", headers={"X-API-Key": "test-key"})
            
            assert response.status_code == 200
            assert "data" in response.json()

    def test_chat_completions_endpoint(self, client):
        with patch("app.routes.proxy_router.requests.request") as mock_request:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_resp.content = b'{"choices": [{"message": {"content": "Hello!"}}]}'
            mock_resp.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_resp
            
            response = client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"X-API-Key": "test-key"},
            )
            
            assert response.status_code == 200


class TestDebugRouter:
    @pytest.fixture
    def app(self, mock_tokenizer, mock_embedding_service, temp_dir, monkeypatch):
        monkeypatch.setenv("LLM_ENABLE_DEBUG", "true")
        
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        app = FastAPI()
        router = build_debug_router(
            store=store,
            embedding_service=mock_embedding_service,
            vector_backend="simple",
            vector_db_path="./test_qdrant",
            vector_collection="test",
        )
        app.include_router(router)
        
        async def mock_verify():
            return "test-key"
        app.dependency_overrides[verify_api_key] = mock_verify
        
        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_vector_stats(self, client):
        response = client.get("/debug/vector/stats", headers={"X-API-Key": "test-key"})
        
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "vector_count" in data

    def test_vector_add(self, client):
        response = client.post(
            "/debug/vector/add",
            json={"text": "Test document", "metadata": {"source": "test"}},
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data

    def test_vector_search(self, client):
        # Add first
        client.post(
            "/debug/vector/add",
            json={"text": "Python programming", "metadata": {}},
            headers={"X-API-Key": "test-key"},
        )
        
        response = client.post(
            "/debug/vector/search",
            json={"query": "Python", "top_k": 5},
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 1


class TestSchemas:
    def test_new_session_request(self):
        req = NewSessionRequest(device_name="test", system_prompt="Be helpful")
        assert req.device_name == "test"
        assert req.system_prompt == "Be helpful"

    def test_new_session_request_defaults(self):
        req = NewSessionRequest()
        assert req.device_name == "unknown device"
        assert req.system_prompt is None

    def test_chat_request(self):
        req = ChatRequest(message="Hello", max_tokens=100, temperature=0.7)
        assert req.message == "Hello"
        assert req.max_tokens == 100
        assert req.temperature == 0.7

    def test_chat_request_defaults(self):
        req = ChatRequest(message="Hello")
        assert req.max_tokens == 512
        assert req.temperature == 0.7

    def test_session_info(self):
        info = SessionInfo(
            session_id="test-1",
            device_name="device-1",
            turns=5,
            last_active=1234567890.0,
            context_used=100,
            context_limit=4096,
        )
        assert info.session_id == "test-1"
        assert info.turns == 5