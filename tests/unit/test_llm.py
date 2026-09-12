"""
Unit tests for LLM components (catalog.py, gpu_detect.py, server_launcher.py, tools.py, agent_loop.py)
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.llm.catalog import MODEL_CATALOG, get_existing_models, download_model
from app.llm.gpu_detect import detect_gpu, get_vram_tier
from app.llm.tools import (
    read_file,
    write_file,
    list_dir,
    run_bash,
    _truncate,
    _summarize_tool_output,
    TOOLS,
)
from app.llm.agent_loop import (
    estimate_response_reserve,
    run_agent_turn,
    _execute_tools_parallel,
    MAX_TOOL_ROUNDS,
)
from app.llm.server_launcher import PERFORMANCE_TIERS, get_adaptive_configs
from app.llm.completion_client import (
    LoopbackCompletionClient,
    _normalize_messages,
    _normalize_tool_calls,
)


class TestCatalog:
    def test_catalog_structure(self):
        """All VRAM tiers should have at least one model."""
        for tier, models in MODEL_CATALOG.items():
            assert len(models) > 0, f"Tier {tier} has no models"
            for name, info in models.items():
                assert "repo_id" in info
                assert "filename" in info
                assert "size" in info

    def test_tiers_are_sorted(self):
        """Tiers should be in ascending VRAM order."""
        expected_tiers = [
            "4GB", "6GB", "8GB", "10GB", "12GB",
            "16GB", "20GB", "24GB", "32GB", "40GB", "48GB"
        ]
        actual_tiers = list(MODEL_CATALOG.keys())
        assert actual_tiers == expected_tiers

    def test_get_existing_models(self, temp_dir):
        """Should find .gguf files in models directory."""
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        (models_dir / "model1.gguf").write_text("fake")
        (models_dir / "model2.gguf").write_text("fake")
        (models_dir / "not_a_model.txt").write_text("fake")
        
        models = get_existing_models(str(models_dir))
        
        assert len(models) == 2
        assert all(m.endswith(".gguf") for m in models)

    @patch("app.llm.catalog.hf_hub_download")
    def test_download_model(self, mock_download, temp_dir):
        mock_download.return_value = str(temp_dir / "downloaded.gguf")
        
        path = download_model(str(temp_dir), "8GB", "Qwen 2.5 7B (Q4_K_M - Reliable Tool-Calling & General Chat - ~4.7GB)")
        
        assert path == str(temp_dir / "downloaded.gguf")
        mock_download.assert_called_once()


class TestGPUDetect:
    def test_get_vram_tier_boundaries(self):
        assert get_vram_tier(4) == "4GB"
        assert get_vram_tier(5) == "4GB"
        assert get_vram_tier(6) == "6GB"
        assert get_vram_tier(7) == "6GB"
        assert get_vram_tier(8) == "8GB"
        assert get_vram_tier(9) == "8GB"
        assert get_vram_tier(10) == "10GB"
        assert get_vram_tier(12) == "12GB"
        assert get_vram_tier(16) == "16GB"
        assert get_vram_tier(20) == "20GB"
        assert get_vram_tier(24) == "24GB"
        assert get_vram_tier(32) == "32GB"
        assert get_vram_tier(40) == "40GB"
        assert get_vram_tier(48) == "48GB"
        assert get_vram_tier(64) == "48GB"  # Caps at 48GB

    @patch("app.llm.gpu_detect.shutil.which")
    @patch("app.llm.gpu_detect.subprocess.check_output")
    def test_detect_gpu_nvidia(self, mock_check_output, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_check_output.return_value = "RTX 3080, 10240"
        
        vendor, name, vram = detect_gpu()
        
        assert vendor == "NVIDIA"
        assert name == "RTX 3080"
        assert vram == 10


class TestServerLauncher:
    def test_performance_tiers_exist(self):
        for tier in ["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "20GB", "24GB", "32GB", "40GB", "48GB"]:
            assert tier in PERFORMANCE_TIERS
            assert len(PERFORMANCE_TIERS[tier]) > 0

    def test_get_adaptive_configs_duplicates_for_kv_quant(self):
        configs = get_adaptive_configs("8GB")
        
        # Each base config should appear twice (kv_quant True and False)
        base_count = len(PERFORMANCE_TIERS["8GB"])
        assert len(configs) == base_count * 2
        
        # Check kv_quant values
        kv_true = [c for c in configs if c["kv_quant"] is True]
        kv_false = [c for c in configs if c["kv_quant"] is False]
        assert len(kv_true) == base_count
        assert len(kv_false) == base_count

    def test_unknown_tier_falls_back_to_4gb(self):
        configs = get_adaptive_configs("UNKNOWN_TIER")
        assert len(configs) == len(PERFORMANCE_TIERS["4GB"]) * 2


class TestTools:
    def test_truncate_short(self):
        text = "short"
        assert _truncate(text) == "short"

    def test_truncate_long(self):
        text = "x" * 10000
        result = _truncate(text)
        assert len(result) <= 8000 + 50  # truncation message
        assert "[truncated" in result

    def test_summarize_tool_output_short(self):
        text = "short output"
        result = _summarize_tool_output(text, 512, "read_file")
        assert result == text

    def test_summarize_tool_output_long_preserves_errors(self):
        text = "Processing...\n" * 100 + "ERROR: Something failed\n" + "Done\n" * 10
        result = _summarize_tool_output(text, 100, "run_bash")
        assert "ERROR" in result
        assert "Something failed" in result

    def test_summarize_tool_output_long_preserves_success(self):
        text = "Processing...\n" * 100 + "SUCCESS: All done\n" + "Cleanup\n" * 10
        result = _summarize_tool_output(text, 100, "run_bash")
        assert "SUCCESS" in result or "All done" in result

    def test_tools_registry_has_expected_tools(self):
        assert "read_file" in TOOLS
        assert "write_file" in TOOLS
        assert "list_dir" in TOOLS
        assert "run_bash" in TOOLS
        
        for tool_name, tool in TOOLS.items():
            assert "fn" in tool
            assert "schema" in tool
            assert tool["schema"]["function"]["name"] == tool_name

    def _reload_tools_with_workspace(self, workspace_dir):
        """Reload tools module with new workspace dir."""
        import importlib
        import app.llm.tools
        os.environ["LLM_WORKSPACE_DIR"] = str(workspace_dir)
        importlib.reload(app.llm.tools)
        return app.llm.tools

    @pytest.mark.asyncio
    async def test_read_file_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        test_file = workspace_dir / "test.txt"
        test_file.write_text("Hello, world!")
        
        result = await tools.read_file("test.txt")
        
        assert "Hello, world!" in result

    @pytest.mark.asyncio
    async def test_write_file_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        result = await tools.write_file("new.txt", "New content")
        
        assert "Wrote" in result
        assert (workspace_dir / "new.txt").read_text() == "New content"

    @pytest.mark.asyncio
    async def test_list_dir_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        (workspace_dir / "file1.txt").write_text("1")
        (workspace_dir / "file2.txt").write_text("2")
        (workspace_dir / "subdir").mkdir()
        
        result = await tools.list_dir(".")
        
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    @pytest.mark.asyncio
    async def test_run_bash_disabled_by_default(self):
        tools = self._reload_tools_with_workspace(Path("/tmp/workspace"))
        os.environ.pop("LLM_ALLOW_SHELL", None)
        result = await tools.run_bash("echo hello")
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_run_bash_enabled(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        os.environ["LLM_ALLOW_SHELL"] = "1"
        
        result = await tools.run_bash("echo hello")
        
        assert "hello" in result


class TestAgentLoop:
    def test_estimate_response_reserve_simple(self):
        reserve = estimate_response_reserve("Hi", 256, 2048)
        assert reserve == 256

    def test_estimate_response_reserve_code(self):
        reserve = estimate_response_reserve("Write a Python function", 256, 2048)
        assert reserve > 256

    def test_estimate_response_reserve_complex(self):
        reserve = estimate_response_reserve(
            "Build a complete REST API with authentication, database, and tests. Also add documentation.",
            256, 2048
        )
        assert reserve > 500

    def test_estimate_response_reserve_bounds(self):
        reserve = estimate_response_reserve("x" * 10000, 256, 2048)
        assert reserve <= 2048
        
        reserve = estimate_response_reserve("", 256, 2048)
        assert reserve >= 256

    def test_max_tool_rounds_constant(self):
        assert MAX_TOOL_ROUNDS == 8

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, mock_tokenizer):
        """Test parallel tool execution."""
        from unittest.mock import AsyncMock

        async def tool_a() -> str:
            return "result_a"

        async def tool_b() -> str:
            return "result_b"

        tools = {
            "tool_a": {
                "fn": tool_a,
                "schema": {"type": "function", "function": {"name": "tool_a"}},
            },
            "tool_b": {
                "fn": tool_b,
                "schema": {"type": "function", "function": {"name": "tool_b"}},
            },
        }
        
        tool_calls = [
            {"id": "1", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tool_b", "arguments": "{}"}},
        ]
        
        results = await _execute_tools_parallel(tools, tool_calls, 30.0, 2)
        
        assert len(results) == 2
        assert "result_a" in results
        assert "result_b" in results


class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_run_agent_turn_no_tools(self, mock_tokenizer, mock_embedding_service, temp_dir):
        """Test agent turn without tool calls."""
        from unittest.mock import AsyncMock
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.completion_client import CompletionClient
        
        # Mock completion client (async)
        mock_client = AsyncMock(spec=CompletionClient)
        mock_client.complete_with_tools.side_effect = [
            {"content": "Hello! How can I help?", "tool_calls": None},  # Final response
        ]
        
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
        
        reply = await run_agent_turn(
            store=store,
            session_id=session.session_id,
            completion_client=mock_client,
            tools={},
            rag_query="Hi",
        )
        
        assert reply == "Hello! How can I help?"
        assert mock_client.complete_with_tools.call_count == 1


class TestCompletionClientToolCallNormalization:
    def test_normalize_tool_calls_adds_missing_type(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "x", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        }
        out = _normalize_tool_calls(msg)
        assert out["tool_calls"][0]["type"] == "function"
        assert out["tool_calls"][0]["id"] == "x"
        assert out is not msg

    def test_normalize_tool_calls_preserves_existing_type(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        }
        assert _normalize_tool_calls(msg)["tool_calls"][0]["type"] == "function"

    def test_normalize_tool_calls_ignores_plain_message(self):
        msg = {"role": "user", "content": "hi"}
        assert _normalize_tool_calls(msg) == msg

    def test_normalize_messages_leaves_non_dicts_alone(self):
        assert _normalize_messages([{"role": "user", "content": "hi"}, "raw"]) == [
            {"role": "user", "content": "hi"},
            "raw",
        ]

    @pytest.mark.asyncio
    async def test_complete_with_tools_stream_emits_type_and_normalizes_payload(self):
        client = LoopbackCompletionClient("http://localhost:8081")
        client._client = MagicMock()

        stale_history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "old_1",
                        "function": {"name": "web_search", "arguments": '{"query": "Edo"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old_1", "content": "results"},
        ]

        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": ""}}]}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "Volta"}'}}]}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
            "data: [DONE]",
        ]

        class FakeResp:
            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def aiter_lines(self):
                for line in lines:
                    yield line

        client._client.stream.return_value = FakeResp()

        events = []
        async for event in client.complete_with_tools_stream(
            stale_history,
            [{"type": "function", "function": {"name": "web_search"}}],
            32,
            0.7,
        ):
            events.append(event)

        finish = events[-1]
        assert finish["type"] == "finish"
        tc = finish["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "web_search"
        assert "Volta" in tc["function"]["arguments"]

        _, kwargs = client._client.stream.call_args
        payload = kwargs["json"]
        assert payload["messages"][0]["tool_calls"][0]["type"] == "function"


class TestDeadRoundGuard:
    def test_is_dead_tool_result(self):
        from app.llm.agents.base import _is_dead_tool_result
        assert _is_dead_tool_result(
            "Error: web search is temporarily unavailable (all providers blocked or unreachable)"
        ) is True
        assert _is_dead_tool_result("Search for 'x': no results found.") is True
        assert _is_dead_tool_result("error running command: boom") is True
        assert _is_dead_tool_result(
            "Search results for: cats\n1. Cats Are Great\n   URL: example.com/cats"
        ) is False
        assert _is_dead_tool_result("") is False

    def test_is_dead_round(self):
        from app.llm.agents.base import _is_dead_round
        assert _is_dead_round([]) is False
        assert _is_dead_round(["Error: a", "Error: b"]) is True
        assert _is_dead_round(["real content"]) is False
        assert _is_dead_round(["Error: a", "real content"]) is False


class TestFileToolsGuardrail:
    def test_file_tools_note_empty_when_file_tools_available(self):
        from app.llm.agents.base import _file_tools_note
        assert _file_tools_note({"read_file": {}, "web_search": {}}) == ""
        assert _file_tools_note({"run_bash": {}}) == ""
        assert _file_tools_note({"list_dir": {}}) == ""
        assert _file_tools_note({"write_file": {}}) == ""

    def test_file_tools_note_added_when_only_web_search(self):
        from app.llm.agents.base import _file_tools_note
        note = _file_tools_note({"web_search": {}})
        assert "not available" in note
        assert "read_file" in note
        assert "analyze it directly" in note

    def test_file_tools_note_added_when_no_tools(self):
        from app.llm.agents.base import _file_tools_note
        assert _file_tools_note({}) != ""

    def test_build_messages_appends_guardrail_without_file_tools(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.agents.base import AgentContext
        from app.llm.agents.reviewer import ReviewerAgent

        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "verify this code")

        context = AgentContext(
            session_id=session.session_id,
            query="verify this code",
            store=store,
            completion_client=object(),
            tools={"web_search": {}},
        )
        agent = ReviewerAgent()
        messages = agent._build_messages_with_system(context)
        assert messages[0]["role"] == "system"
        assert "not available in this session" in messages[0]["content"]
        assert "analyze it directly" in messages[0]["content"]

    def test_build_messages_no_guardrail_when_file_tools_available(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.agents.base import AgentContext
        from app.llm.agents.reviewer import ReviewerAgent

        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "verify this code")

        context = AgentContext(
            session_id=session.session_id,
            query="verify this code",
            store=store,
            completion_client=object(),
            tools={"read_file": {}, "list_dir": {}, "web_search": {}},
        )
        agent = ReviewerAgent()
        messages = agent._build_messages_with_system(context)
        assert messages[0]["role"] == "system"
        assert "not available in this session" not in messages[0]["content"]

    def _dead_agent_setup(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.sessions.store import SessionStore
        from app.sessions.repository import JSONSessionRepository
        from app.sessions.eviction import SummarizeOldestStrategy
        from app.llm.agents.base import AgentContext
        from app.llm.agents.researcher import ResearcherAgent

        repo = JSONSessionRepository(str(temp_dir / "sessions.json"))
        store = SessionStore(
            counter=mock_tokenizer,
            repository=repo,
            eviction=SummarizeOldestStrategy(),
            n_ctx=4096,
            embedding_service=mock_embedding_service,
        )
        session = store.create_session("device-1")
        store.add_turn(session.session_id, "user", "Who wrote Solo Leveling?")

        async def dead_search(**kwargs):
            return "Error: web search is temporarily unavailable (all providers blocked or unreachable)"

        tools = {
            "web_search": {
                "fn": dead_search,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {
                            "properties": {
                                "query": {"type": "string"},
                                "num_results": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    },
                },
            }
        }
        context = AgentContext(
            session_id=session.session_id,
            query="Who wrote Solo Leveling?",
            store=store,
            completion_client=object(),
            tools=tools,
            max_tokens=64,
            max_retries=0,
            token_budget=4096,
            max_rounds=8,
            max_tool_calls=20,
        )
        return store, session, context, tools

    @pytest.mark.asyncio
    async def test_non_stream_loop_stops_after_two_dead_rounds(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.llm.agents.base import DEAD_ROUND_REPLY
        from app.llm.agents.researcher import ResearcherAgent

        store, session, context, tools = self._dead_agent_setup(
            mock_tokenizer, mock_embedding_service, temp_dir
        )

        payload = {
            "content": "I'll search.",
            "tool_calls": [
                {"id": "tc-1", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "X"}'}}
            ],
        }

        class ToolLoopClient:
            def __init__(self):
                self.calls = 0

            async def complete_with_tools(self, messages, tool_schemas, max_tokens, temperature=0.7, stop=None):
                self.calls += 1
                return payload

        client = ToolLoopClient()
        context.completion_client = client
        schema = tools["web_search"]["schema"]

        agent = ResearcherAgent()
        reply, rounds_used, tool_calls_made = await agent._run_tool_loop(
            context, store.build_messages(session.session_id), [schema], max_rounds=8
        )

        assert reply == DEAD_ROUND_REPLY
        assert rounds_used == 2
        assert tool_calls_made == 2
        assert client.calls == 2
        assert store.get(session.session_id).history[-1]["role"] == "assistant"
        assert store.get(session.session_id).history[-1]["content"] == DEAD_ROUND_REPLY

    @pytest.mark.asyncio
    async def test_stream_loop_stops_after_two_dead_rounds(self, mock_tokenizer, mock_embedding_service, temp_dir):
        from app.llm.agents.base import DEAD_ROUND_REPLY
        from app.llm.agents.researcher import ResearcherAgent

        store, session, context, tools = self._dead_agent_setup(
            mock_tokenizer, mock_embedding_service, temp_dir
        )

        class StreamToolLoopClient:
            def __init__(self):
                self.calls = 0

            async def complete_with_tools_stream(self, messages, tool_schemas, max_tokens, temperature=0.7, stop=None):
                self.calls += 1
                tc = {"id": "tc-1", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "X"}'}}
                yield {"type": "tool_calls", "tool_calls": [dict(tc)]}
                yield {"type": "finish", "finish_reason": "tool_calls", "content": "", "tool_calls": [dict(tc)]}

        client = StreamToolLoopClient()
        context.completion_client = client
        schema = tools["web_search"]["schema"]

        agent = ResearcherAgent()
        events = []
        async for event in agent._run_tool_loop_stream(
            context, store.build_messages(session.session_id), [schema], max_rounds=8
        ):
            events.append(event)

        assert client.calls == 2
        assert events[-1]["type"] == "done"
        assert events[-1]["reply"] == DEAD_ROUND_REPLY
        assert events[-2]["type"] == "content_delta"
        assert events[-2]["content"] == DEAD_ROUND_REPLY
        assert store.get(session.session_id).history[-1]["role"] == "assistant"