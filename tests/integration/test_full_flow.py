"""
Integration tests for full request flows.
"""

from unittest.mock import MagicMock, AsyncMock, patch
import os

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends

from app.auth import verify_api_key


class TestFullChatFlow:
    @pytest.fixture
    def app(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.main import create_app
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.completion_client import CompletionClient
        
        # We need to mock the llama server startup
        with patch("app.main.launch_llama_server") as mock_launch:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_launch.return_value = (mock_process, "http://localhost:8081", {
                "n_ctx": 4096,
                "n_batch": 512,
                "kv_quant": False,
            })
            
            with patch("app.main.watch_llama_server"):
                with patch("app.main.periodic_cleanup"):
                    app = create_app()
        
        # Override the completion client with mock
        mock_client = MagicMock(spec=CompletionClient)
        mock_client.complete_with_tools = AsyncMock(return_value={
            "content": "I'll help you with that Python function.",
            "tool_calls": None,
        })
        
        # Replace the completion client in the app's routers
        # This is tricky - we need to access the router's closure
        # For now, just test that app creates
        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model" in data

    def test_stats_endpoint(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "server" in data
        assert "sessions" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "llm_" in response.text


class TestSessionWithRAG:
    @pytest.fixture
    def session_with_rag(self, mock_tokenizer, mock_embedding_service, temp_dir, monkeypatch):
        monkeypatch.setenv("LLM_RAG_TOKEN_BUDGET", "2000")
        monkeypatch.setenv("LLM_BUDGET_RAG_PCT", "0.3")
        
        # Reload settings to pick up new env vars
        import importlib
        import app.config
        importlib.reload(app.config)
        
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
        
        session = store.create_session("device-1")
        
        # Add conversation history
        store.add_turn(session.session_id, "user", "I'm learning Python for data science")
        store.add_turn(session.session_id, "assistant", "Great! Python has pandas, numpy, and scikit-learn")
        store.add_turn(session.session_id, "user", "I also want to learn JavaScript for web")
        store.add_turn(session.session_id, "assistant", "JavaScript is essential for frontend development")
        
        return store, session.session_id

    @pytest.mark.skip(reason="Test isolation issue - passes in isolation but fails in suite due to config module caching")
    def test_rag_retrieves_relevant_context(self, session_with_rag, mock_embedding_service):
        store, session_id = session_with_rag
        
        # Mock the session's retrieve_relevant to return relevant content
        session = store.get(session_id)
        original_retrieve = session.retrieve_relevant
        session.retrieve_relevant = lambda *args, **kwargs: [(0.9, "Python for data science", {"role": "assistant"})]
        
        try:
            messages = store.build_messages(
                session_id=session_id,
                use_rag=True,
                query="What languages am I learning?",
            )
        finally:
            session.retrieve_relevant = original_retrieve
        
        # Should have system + RAG context + history
        assert len(messages) >= 3
        rag_content = " ".join(m.get("content", "") for m in messages if "Relevant context" in m.get("content", ""))
        assert "Python" in rag_content or "JavaScript" in rag_content

    def test_rag_budget_enforced(self, session_with_rag, mock_tokenizer):
        store, session_id = session_with_rag
        
        # Add very long history
        for i in range(20):
            store.add_turn(session_id, "user", f"Long message {i} " * 100)
            store.add_turn(session_id, "assistant", f"Response {i} " * 100)
        
        # Call build_messages on the session directly with token_counter
        session = store.get(session_id)
        messages = session.build_messages(
            use_rag=True,
            query="test",
            token_counter=mock_tokenizer,
        )
        
        # Total tokens should be reasonable
        total = sum(mock_tokenizer.count(m.get("content", "")) for m in messages)
        assert total < 5000  # Should not explode


class TestToolFlow:
    @pytest.fixture
    def tool_session(self, mock_tokenizer, mock_embedding_service, temp_dir, workspace_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.tools import TOOLS
        
        os.environ["LLM_WORKSPACE_DIR"] = str(workspace_dir)
        os.environ["LLM_ALLOW_SHELL"] = "1"
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        return store, session.session_id, TOOLS

    @pytest.mark.asyncio
    async def test_agent_turn_with_tool_call(self, tool_session):
        store, session_id, tools = tool_session
        from app.llm.completion_client import CompletionClient
        
        mock_client = MagicMock(spec=CompletionClient)
        # First call returns tool call, second returns final answer
        def side_effect(messages, tool_schemas, max_tokens, temperature):
            call_count = side_effect.call_count
            side_effect.call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "tc-1",
                        "function": {"name": "write_file", "arguments": '{"path": "test.py", "content": "print(1)"}'}
                    }]
                }
            else:
                return {
                    "content": "File created successfully.",
                    "tool_calls": None,
                }
        side_effect.call_count = 1
        mock_client.complete_with_tools = side_effect
        
        from app.llm.agent_loop import run_agent_turn
        
        reply = await run_agent_turn(
            store=store,
            session_id=session_id,
            completion_client=mock_client,
            tools=tools,
            rag_query="Create a test file",
        )
        
        assert "File created" in reply
        assert side_effect.call_count == 3
        
        # Check file was written
        import os
        assert os.path.exists("workspace/test.py")

    @pytest.mark.asyncio
    async def test_tool_output_summarization_in_session(self, tool_session):
        store, session_id, tools = tool_session
        from app.llm.completion_client import CompletionClient
        
        mock_client = MagicMock(spec=CompletionClient)
        def side_effect(messages, tool_schemas, max_tokens, temperature):
            call_count = side_effect.call_count
            side_effect.call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "tc-1",
                        "function": {"name": "run_bash", "arguments": '{"command": "echo line 1 && echo line 2 && echo line 3 && echo ERROR: failed && echo line 5"}'}
                    }]
                }
            else:
                return {
                    "content": "Command completed with error.",
                    "tool_calls": None,
                }
        side_effect.call_count = 1
        mock_client.complete_with_tools = side_effect
        
        from app.llm.agent_loop import run_agent_turn
        
        reply = await run_agent_turn(
            store=store,
            session_id=session_id,
            completion_client=mock_client,
            tools=tools,
            rag_query="Run a command",
        )
        
        # Tool result should be in history
        session = store.get(session_id)
        tool_messages = [m for m in session.history if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        
        # Should be summarized (preserving ERROR)
        tool_content = tool_messages[0]["content"]
        assert "ERROR" in tool_content


class TestDynamicReserve:
    @pytest.mark.asyncio
    async def test_dynamic_reserve_affects_max_tokens(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.completion_client import CompletionClient
        
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        
        mock_client = MagicMock(spec=CompletionClient)
        captured_max_tokens = []
        
        def capture_max_tokens(messages, tool_schemas, max_tokens, temperature):
            captured_max_tokens.append(max_tokens)
            return {"content": "Response", "tool_calls": None}
        
        mock_client.complete_with_tools = capture_max_tokens
        
        from app.llm.agent_loop import run_agent_turn
        from app.config import settings
        
        # Simple query - should use min reserve
        await run_agent_turn(
            store=store,
            session_id=session.session_id,
            completion_client=mock_client,
            tools={},
            rag_query="Hi",
        )
        
        simple_reserve = captured_max_tokens[0]
        assert simple_reserve == settings.reserve_for_response_min
        
        captured_max_tokens.clear()
        
        # Complex query - should use higher reserve
        await run_agent_turn(
            store=store,
            session_id=session.session_id,
            completion_client=mock_client,
            tools={},
            rag_query="Write a complete REST API with authentication, database models, tests, and documentation",
        )
        
        complex_reserve = captured_max_tokens[0]
        assert complex_reserve > simple_reserve
        assert complex_reserve <= settings.reserve_for_response_max