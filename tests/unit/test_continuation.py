"""
Unit tests for the truncation-continuation behavior:
when the model hits its per-call token cap (finish_reason == "length"),
the app continues generating until it stops naturally (Meta-AI-style
queued messages), and only the completed reply is persisted.
"""

import pytest

from app.llm.continuation import (
    continue_text,
    stream_text_continuation,
    CUTOFF_MARKER,
    REPEAT_MARKER,
    detect_repetition,
)
from app.llm.agent_loop import run_agent_turn
from app.llm.agent_loop_streaming import run_agent_turn_streaming
from app.llm.agents.base import BaseAgent, AgentContext, AgentResult, AgentType


class SequenceClient:
    """Non-streaming fake: returns the next response dict per call (repeats last)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_with_tools(self, messages, tools, max_tokens, temperature):
        self.calls.append((len(messages), list(tools), max_tokens, temperature))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


class StreamSequenceClient:
    """Streaming fake: yields one event-sequence per call (exhausts at end)."""

    def __init__(self, streams):
        self.streams = list(streams)

    async def complete_with_tools_stream(self, messages, tools, max_tokens, temperature):
        if not self.streams:
            return
        for event in self.streams.pop(0):
            yield event


def _store_factory(mock_tokenizer, mock_embedding_service, temp_dir):
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
    return store


class TestContinuationHelpers:
    @pytest.mark.asyncio
    async def test_continue_text_concatenates_until_stop(self):
        client = SequenceClient([
            {"content": " continued", "tool_calls": None, "finish_reason": "stop"},
        ])
        reply, completed, used = await continue_text(client, [], "Partial answer", 0.7, remaining_budget=4096)
        assert reply == "Partial answer continued"
        assert completed is True
        assert used > 0

    @pytest.mark.asyncio
    async def test_continue_text_continues_after_length(self):
        client = SequenceClient([
            {"content": " part one", "tool_calls": None, "finish_reason": "length"},
            {"content": " part two", "tool_calls": None, "finish_reason": "stop"},
        ])
        reply, completed, used = await continue_text(client, [], "Start.", 0.7, remaining_budget=4096)
        assert reply == "Start. part one part two"
        assert completed is True
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_continue_text_marks_cutoff_when_rounds_exhausted(self):
        client = SequenceClient([
            {"content": "B", "tool_calls": None, "finish_reason": "length"},
        ])
        reply, completed, used = await continue_text(
            client, [], "A", 0.7, remaining_budget=1024, max_rounds=1
        )
        assert reply == "AB" + CUTOFF_MARKER
        assert completed is False

    @pytest.mark.asyncio
    async def test_continue_text_marks_cutoff_when_budget_exhausted(self):
        client = SequenceClient([
            {"content": "B", "tool_calls": None, "finish_reason": "length"},
        ])
        reply, completed, used = await continue_text(client, [], "A", 0.7, remaining_budget=0)
        assert reply == "A" + CUTOFF_MARKER
        assert completed is False
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_stream_continuation_yields_deltas_then_done(self):
        client = StreamSequenceClient([
            [
                {"type": "content", "content": " continued"},
                {"type": "finish", "finish_reason": "stop", "content": " continued", "tool_calls": []},
            ],
        ])
        events = []
        async for event in stream_text_continuation(client, [], "Partial answer", 0.7, remaining_budget=4096):
            events.append(event)
        assert [e["type"] for e in events] == ["content_delta", "done"]
        assert events[0]["content"] == " continued"
        assert events[-1]["reply"] == "Partial answer continued"


PARTIAL = "To convert alternating current (AC) to direct current (DC), you need a rectifier circuit followed by a smoothing capacitor."
RESTART = PARTIAL + " Here is the full detailed walkthrough: first, a diode bridge rectifies the AC signal."


class TestRepetitionGuard:
    @pytest.mark.asyncio
    async def test_detect_repetition_leading_overlap(self):
        assert detect_repetition(RESTART, PARTIAL) is True

    @pytest.mark.asyncio
    async def test_detect_repetition_ngram_overlap(self):
        # Leading 80 chars are fresh, but >=60% of the 40-char grams already
        # exist in the preceding text -> flagged by the n-gram branch.
        block = "a" * 40
        preceding = block * 2
        piece = "b" * 40 + block * 2
        assert detect_repetition(piece, preceding) is True

    @pytest.mark.asyncio
    async def test_detect_repetition_no_false_positive(self):
        novel = "Just continue from exactly where I left off describing the smoothing stage."
        assert detect_repetition(novel, PARTIAL) is False

    @pytest.mark.asyncio
    async def test_detect_repetition_short_signal_ignored(self):
        assert detect_repetition("ok", PARTIAL) is False

    @pytest.mark.asyncio
    async def test_continue_text_stops_on_repetition(self):
        client = SequenceClient([
            {"content": RESTART, "tool_calls": None, "finish_reason": "length"},
        ])
        reply, completed, used = await continue_text(client, [], PARTIAL, 0.7, remaining_budget=4096)
        assert reply == PARTIAL + REPEAT_MARKER
        assert completed is False
        assert used == 0
        assert len(client.calls) == 1
        assert RESTART not in reply

    @pytest.mark.asyncio
    async def test_stream_continuation_stops_on_repetition(self):
        client = StreamSequenceClient([
            [
                {"type": "content", "content": RESTART},
                {"type": "finish", "finish_reason": "length", "content": RESTART, "tool_calls": []},
            ],
        ])
        events = []
        async for event in stream_text_continuation(client, [], PARTIAL, 0.7, remaining_budget=4096):
            events.append(event)

        assert [e["type"] for e in events] == ["content_delta", "done"]
        assert events[0]["content"] == REPEAT_MARKER
        assert events[-1]["reply"] == PARTIAL + REPEAT_MARKER
        assert RESTART not in events[-1]["reply"]

    @pytest.mark.asyncio
    async def test_agent_turn_stops_on_repetition(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = SequenceClient([
            {"content": PARTIAL, "tool_calls": None, "finish_reason": "length"},
            {"content": RESTART, "tool_calls": None, "finish_reason": "length"},
        ])

        reply = await run_agent_turn(store, session.session_id, client, {}, rag_query=None)

        assert reply == PARTIAL + REPEAT_MARKER
        assert len(client.calls) == 2
        assert RESTART not in reply
        # Only the marker-suffixed reply is persisted, never the repeated block.
        turns = store.get(session.session_id).history
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == reply


class _FakeAgent(BaseAgent):
    def __init__(self, agent_type=AgentType.GENERAL):
        super().__init__(
            agent_type=agent_type,
            name="fake-continuation-agent",
            description="test",
            system_prompt="You are a test agent.",
            allowed_tools=[],
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]
        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=8
        )
        return AgentResult(
            reply=reply, agent_type=self.agent_type,
            tool_calls_made=tool_calls_made, rounds_used=rounds_used,
        )

    async def _stream(self, context):
        messages = self._build_messages_with_system(context)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]
        async for event in self._run_tool_loop_stream(context, messages, tool_schemas, max_rounds=8):
            yield event


def _make_context(store, session_id, client, token_budget=4096):
    return AgentContext(
        session_id=session_id,
        query="Tell me everything about AC to DC conversion.",
        store=store,
        completion_client=client,
        tools={},
        max_tokens=512,
        temperature=0.7,
        token_budget=token_budget,
        max_tool_calls=20,
        max_rounds=12,
    )


class TestAgentLoopContinuation:
    @pytest.mark.asyncio
    async def test_run_agent_turn_extends_truncated_reply(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = SequenceClient([
            {"content": "A rectifier converts AC to", "tool_calls": None, "finish_reason": "length"},
            {"content": " pulsating DC, then filtered.", "tool_calls": None, "finish_reason": "stop"},
        ])

        reply = await run_agent_turn(
            store, session.session_id, client, {}, rag_query=None,
        )

        assert reply == "A rectifier converts AC to pulsating DC, then filtered."
        assert len(client.calls) == 2
        # Only ONE complete assistant turn persisted -- never the partial cut-off.
        turns = store.get(session.session_id).history
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == reply
        assert turns[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_agent_execute_extends_truncated_reply(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = SequenceClient([
            {"content": "A rectifier converts AC to", "tool_calls": None, "finish_reason": "length"},
            {"content": " pulsating DC, then filtered.", "tool_calls": None, "finish_reason": "stop"},
        ])

        agent = _FakeAgent()
        result = await agent.execute(_make_context(store, session.session_id, client))
        assert result.reply == "A rectifier converts AC to pulsating DC, then filtered."
        assert len(client.calls) == 2

        turns = store.get(session.session_id).history
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == result.reply


class TestStreamingContinuation:
    @pytest.mark.asyncio
    async def test_run_agent_turn_streaming_extends_truncated_reply(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = StreamSequenceClient([
            [
                {"type": "content", "content": "A rectifier converts "},
                {"type": "content", "content": "AC to"},
                {"type": "finish", "finish_reason": "length", "content": "A rectifier converts AC to", "tool_calls": []},
            ],
            [
                {"type": "content", "content": " pulsating DC."},
                {"type": "finish", "finish_reason": "stop", "content": " pulsating DC.", "tool_calls": []},
            ],
        ])

        events = []
        async for event in run_agent_turn_streaming(
            store, session.session_id, client, {}, rag_query=None,
        ):
            events.append(event)

        content_deltas = [e for e in events if e.type == "content_delta"]
        done = [e for e in events if e.type == "done"][0]

        assert "".join(d.data["content"] for d in content_deltas) == "A rectifier converts AC to pulsating DC."
        assert done.data["reply"] == "A rectifier converts AC to pulsating DC."

        turns = store.get(session.session_id).history
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == done.data["reply"]

    @pytest.mark.asyncio
    async def test_agent_stream_extends_truncated_reply(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = StreamSequenceClient([
            [
                {"type": "content", "content": "A rectifier converts AC to"},
                {"type": "finish", "finish_reason": "length", "content": "A rectifier converts AC to", "tool_calls": []},
            ],
            [
                {"type": "content", "content": " pulsating DC, after filtering."},
                {"type": "finish", "finish_reason": "stop", "content": " pulsating DC, after filtering.", "tool_calls": []},
            ],
        ])

        agent = _FakeAgent()
        events = []
        async for event in agent._stream(_make_context(store, session.session_id, client)):
            events.append(event)

        content_deltas = [e for e in events if e["type"] == "content_delta"]
        done = [e for e in events if e["type"] == "done"][0]

        expected = "A rectifier converts AC to pulsating DC, after filtering."
        assert "".join(e["content"] for e in content_deltas) == expected
        assert done["reply"] == expected

        turns = store.get(session.session_id).history
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1

    @pytest.mark.asyncio
    async def test_streaming_cutoff_marker(self, mock_tokenizer, mock_embedding_service, temp_dir):
        store = _store_factory(mock_tokenizer, mock_embedding_service, temp_dir)
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Explain AC to DC conversion")

        client = StreamSequenceClient([
            [
                {"type": "content", "content": "A rectifier converts AC to"},
                {"type": "finish", "finish_reason": "length", "content": "A rectifier converts AC to", "tool_calls": []},
            ],
            [
                {"type": "content", "content": " more content"},
                {"type": "finish", "finish_reason": "length", "content": " more content", "tool_calls": []},
            ],
        ])

        agent = _FakeAgent()
        context = _make_context(store, session.session_id, client, token_budget=1)

        events = []
        async for event in agent._stream(context):
            events.append(event)

        done = [e for e in events if e["type"] == "done"][0]
        assert CUTOFF_MARKER in done["reply"]