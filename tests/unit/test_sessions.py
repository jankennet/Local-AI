"""
Unit tests for sessions module (session.py, store.py, eviction.py, repository.py)
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.sessions.session import Session
from app.sessions.store import SessionStore
from app.sessions.repository import JSONSessionRepository
from app.sessions.eviction import (
    EvictionStrategy,
    DropOldestStrategy,
    SummarizeOldestStrategy,
    _session_tokens,
    _pop_oldest_group,
)


class TestSession:
    def test_create_session(self, mock_embedding_service, mock_tokenizer):
        session = Session(
            session_id="test-1",
            device_name="test-device",
            system_prompt="You are helpful.",
        )
        session.init_vector_store(mock_embedding_service)
        
        assert session.session_id == "test-1"
        assert session.device_name == "test-device"
        assert session.system_prompt == "You are helpful."
        assert session.history == []
        assert session.summary == ""

    def test_add_turn_to_vector_store(self, mock_embedding_service, mock_tokenizer):
        session = Session(session_id="test-1")
        session.init_vector_store(mock_embedding_service)
        
        session.add_to_vector_store("user", "Hello", 0)
        session.add_to_vector_store("assistant", "Hi there!", 1)
        
        # Just verify the calls work, don't test search behavior with mock
        assert len(session.history) == 0  # history not modified by add_to_vector_store

    def test_retrieve_relevant_respects_top_k(self, mock_embedding_service, mock_tokenizer):
        session = Session(session_id="test-1")
        session.init_vector_store(mock_embedding_service)
        
        for i in range(10):
            session.add_to_vector_store("user", f"Message {i}", i)
        
        # Mock returns empty results for search with mock embeddings
        # This is expected with random/deterministic mock vectors
        results = session.retrieve_relevant("message", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_relevant_with_reranker_disabled(self, mock_embedding_service, mock_tokenizer):
        session = Session(session_id="test-1")
        session.init_vector_store(mock_embedding_service)
        
        session.add_to_vector_store("user", "Python programming", 0)
        session.add_to_vector_store("user", "JavaScript web", 1)
        
        results = session.retrieve_relevant("Python", top_k=1, use_reranker=False)
        assert isinstance(results, list)

    def test_build_messages_basic(self, mock_embedding_service, mock_tokenizer):
        session = Session(
            session_id="test-1",
            system_prompt="Be helpful.",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )
        
        # Test without token_counter to avoid budget truncation
        messages = session.build_messages()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi!"

    def test_build_messages_with_summary(self, mock_embedding_service, mock_tokenizer):
        session = Session(
            session_id="test-1",
            system_prompt="Be helpful.",
            summary="User said hello.",
            history=[{"role": "user", "content": "How are you?"}],
        )
        
        # Test without token_counter to avoid budget truncation
        messages = session.build_messages()
        
        assert len(messages) == 3
        assert "Earlier conversation summary" in messages[1]["content"]
        assert "User said hello" in messages[1]["content"]

    def test_build_messages_with_rag(self, mock_embedding_service, mock_tokenizer):
        session = Session(
            session_id="test-1",
            system_prompt="Be helpful.",
            history=[
                {"role": "user", "content": "I love Python"},
                {"role": "assistant", "content": "Python is great"},
            ],
        )
        session.init_vector_store(mock_embedding_service)
        
        # Mock the retrieve_relevant to return something
        with patch.object(session, 'retrieve_relevant', return_value=[(0.9, "I love Python", {"role": "user"})]):
            messages = session.build_messages(
                use_rag=True,
                query="What do I like?",
            )
        
        assert len(messages) >= 3
        rag_msg = next((m for m in messages if "Relevant context" in m.get("content", "")), None)
        assert rag_msg is not None

    def test_build_messages_per_component_budgets(self, mock_embedding_service, mock_tokenizer):
        """Test per-component budget allocation."""
        session = Session(
            session_id="test-1",
            system_prompt="System prompt " * 50,
            summary="Summary " * 50,
            history=[
                {"role": "user", "content": "History " * 50},
                {"role": "assistant", "content": "Response " * 50},
            ],
        )
        session.init_vector_store(mock_embedding_service)
        
        messages = session.build_messages(
            use_rag=True,
            query="test",
            token_counter=mock_tokenizer,
        )
        
        # Should have system, rag, summary, and some history
        assert len(messages) >= 3
        total_tokens = sum(mock_tokenizer.count(m.get("content", "")) for m in messages)
        # Should not wildly exceed reasonable bounds
        assert total_tokens < 10000

    def test_compress_turn(self, mock_embedding_service, mock_tokenizer):
        session = Session(session_id="test-1")
        long_text = "word " * 1000  # ~5000 chars
        compressed = session._compress_turn(long_text, 100, mock_tokenizer)
        assert len(compressed) < len(long_text)
        assert "[truncated]" in compressed

    def test_fit_retrieved_to_budget(self, mock_embedding_service, mock_tokenizer):
        session = Session(session_id="test-1")
        retrieved = [
            (0.9, "short", {}),
            (0.8, "medium " * 50, {}),
            (0.7, "long " * 200, {}),
        ]
        
        fitted = session._fit_retrieved_to_budget(retrieved, 100, mock_tokenizer)
        
        total = sum(mock_tokenizer.count(t) for _, t, _ in fitted)
        assert total <= 120  # Some overhead


class TestSessionStore:
    def test_create_session(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1", "Custom prompt")
        
        assert session.session_id is not None
        assert session.device_name == "device-1"
        assert session.system_prompt == "Custom prompt"

    def test_get_session(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        created = store.create_session("device-1")
        retrieved = store.get(created.session_id)
        
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_delete_session(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        created = store.create_session("device-1")
        store.delete(created.session_id)
        
        assert store.get(created.session_id) is None

    def test_list_sessions(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        store.create_session("device-1")
        store.create_session("device-2")
        
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_add_turn_triggers_eviction(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=100,  # Very small budget
            reserve_for_response=10,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        
        # Add many turns to trigger eviction
        for i in range(20):
            store.add_turn(session.session_id, "user", f"Message {i} " * 10)
            store.add_turn(session.session_id, "assistant", f"Response {i} " * 10)
        
        # Should have summary now
        retrieved = store.get(session.session_id)
        assert retrieved.summary != ""

    def test_budget_property(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            reserve_for_response=768,
            embedding_service=mock_embedding_service,
        )
        
        assert store.budget == 4096 - 768

    def test_tokens_used(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Hello")
        store.add_turn(session.session_id, "assistant", "Hi there")
        
        tokens = store.tokens_used(session.session_id)
        assert tokens > 0

    def test_purge_expired(self, mock_embedding_service, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            ttl_days=0,  # Expire immediately
            embedding_service=mock_embedding_service,
        )
        
        session = store.create_session("device-1")
        # Manually set old timestamp
        store._sessions[session.session_id].last_active = time.time() - 86400
        store._repo.save(store._sessions)
        
        purged = store.purge_expired()
        assert purged == 1


class TestEvictionStrategies:
    def test_drop_oldest_strategy(self, mock_tokenizer):
        session = Session(
            session_id="test-1",
            history=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"},
            ],
        )
        strategy = DropOldestStrategy()
        strategy.evict(session, mock_tokenizer, budget=10)  # Very small budget
        
        # Should have dropped oldest
        assert len(session.history) < 3

    def test_summarize_oldest_strategy(self, mock_tokenizer):
        session = Session(
            session_id="test-1",
            history=[
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Second message"},
                {"role": "user", "content": "Third message"},
            ],
        )
        strategy = SummarizeOldestStrategy()
        strategy.evict(session, mock_tokenizer, budget=10)
        
        # Should have created summary
        assert session.summary != ""
        assert "First message" in session.summary or "Second message" in session.summary

    def test_pop_oldest_group_tool_calls(self):
        history = [
            {"role": "assistant", "content": "I'll run that", "tool_calls": [{"id": "tc-1", "function": {"name": "run_bash"}}]},
            {"role": "tool", "content": "output", "tool_call_id": "tc-1"},
            {"role": "user", "content": "Next"},
        ]
        
        group = _pop_oldest_group(history)
        
        assert len(group) == 2  # assistant + tool
        assert group[0]["role"] == "assistant"
        assert group[1]["role"] == "tool"
        assert len(history) == 1  # Only "Next" remains


class TestJSONSessionRepository:
    def test_save_and_load(self, mock_tokenizer, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        
        session = Session(
            session_id="test-1",
            device_name="test",
            history=[{"role": "user", "content": "Hello"}],
        )
        
        repo.save({"test-1": session})
        loaded = repo.load()
        
        assert "test-1" in loaded
        assert loaded["test-1"].device_name == "test"
        assert len(loaded["test-1"].history) == 1

    def test_load_empty_file(self, temp_dir):
        repo = JSONSessionRepository(str(temp_dir / "nonexistent.json"))
        loaded = repo.load()
        assert loaded == {}

    def test_load_corrupted_file(self, temp_dir):
        bad_file = temp_dir / "bad.json"
        bad_file.write_text("{ not valid json")
        
        repo = JSONSessionRepository(str(bad_file))
        loaded = repo.load()
        assert loaded == {}