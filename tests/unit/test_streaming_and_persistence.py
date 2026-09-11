"""
Unit tests for Phase A/B/C behaviors:
- async (coroutine-based) completion clients
- batched persistence via SessionStore.flush()
- session-scoped vector retrieval (filter passed to vector store)
- AgentOrchestrator.execute_stream event contract
"""

import pytest

from app.llm.agents.base import AgentContext, AgentType
from app.llm.agents.classifier import ClassificationResult
from app.llm.agents.orchestrator import AgentOrchestrator


class FakeAsyncCompletionClient:
    """Implements CompletionClient protocol with plain `async def` methods."""

    def __init__(self, content=None, tool_calls=None):
        self._content = content or "Fake response"
        self._tool_calls = tool_calls

    async def complete(self, messages, max_tokens, temperature=0.7, stop=None):
        return self._content

    async def complete_with_tools(self, messages, tool_schemas, max_tokens, temperature=0.7, stop=None):
        return {"content": self._content, "tool_calls": self._tool_calls}


class TestAsyncCompletionClient:
    @pytest.mark.asyncio
    async def test_fake_async_client_drives_agent_turn(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.agent_loop import run_agent_turn

        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Hi")

        client = FakeAsyncCompletionClient(content="Hello from fake client")

        reply = await run_agent_turn(
            store=store,
            session_id=session.session_id,
            completion_client=client,
            tools={},
            rag_query="Hi",
        )

        assert reply == "Hello from fake client"
        assert store.get(session.session_id).history[-1]["role"] == "assistant"


class TestFlushBatching:
    @pytest.mark.asyncio
    async def test_flush_persists_pending_mutations(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy

        path = str(temp_dir / "sessions.json")
        repo = JSONSessionRepository(path)
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )

        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Hello")
        store.add_turn(session.session_id, "assistant", "Hi there!")

        # add_turn is batched -- session row exists but turns aren't yet on disk
        assert store._dirty is True
        before = JSONSessionRepository(path)
        assert len(before.load()[session.session_id].history) == 0

        await store.flush()

        assert store._dirty is False
        fresh_repo = JSONSessionRepository(path)
        loaded = fresh_repo.load()
        assert len(loaded[session.session_id].history) == 2


class TestSessionScopedRetrieval:
    def test_retrieve_relevant_passes_session_filter(self, mock_embedding_service):
        from app.sessions.session import Session

        recorded = {}

        class RecordingVectorStore:
            def add(self, content, metadata, embedding_service=None, store_factory=None):
                pass

            def size(self):
                return 1

            def search(self, query, top_k=5, filter=None):
                recorded["filter"] = filter
                return [(0.5, "candidate", {"session_id": "sess-123"})]

            def delete_by_filter(self, filter):
                pass

        session = Session(
            session_id="sess-123",
            device_name="dev",
            system_prompt="sys",
            created_at=0.0,
        )
        session._embedding_service = mock_embedding_service
        session._vector_store = RecordingVectorStore()

        results = session.retrieve_relevant("test query", use_reranker=False, use_dedup=False)

        assert recorded["filter"] == {"session_id": "sess-123"}
        assert results == [(0.5, "candidate", {"session_id": "sess-123"})]


class FakeStreamAgent:
    name = "fake-stream-agent"
    description = "fake"
    allowed_tools = set()

    async def _stream(self, context):
        yield {"type": "content_delta", "content": "partial output"}
        yield {"type": "done", "reply": "final answer"}


class FakeClassifier:
    def classify(self, query, session_context):
        return ClassificationResult(
            agent_type=AgentType.GENERAL,
            confidence=0.9,
            reasoning="test",
            requires_planning=False,
        )


class TestOrchestratorExecuteStream:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_content_delta_then_done(self, mock_tokenizer, mock_embedding_service, temp_dir):
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

        orch = AgentOrchestrator()
        orch.configure(enable_planning=False, enable_review=False)
        orch._agents[AgentType.GENERAL] = FakeStreamAgent()
        orch._classifier = FakeClassifier()

        context = AgentContext(
            session_id=session.session_id,
            query="Tell me a story",
            store=store,
            completion_client=object(),
            tools={},
            max_tokens=64,
            max_retries=0,
        )

        events = []
        async for event in orch.execute_stream(context):
            events.append(event)

        assert [e["type"] for e in events] == ["content_delta", "done"]
        assert events[0]["content"] == "partial output"
        assert events[-1]["reply"] == "final answer"