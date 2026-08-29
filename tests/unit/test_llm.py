"""
Unit tests for LLM components (catalog.py, gpu_detect.py, server_launcher.py, tools.py, agent_loop.py)
"""

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

    def test_read_file_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        test_file = workspace_dir / "test.txt"
        test_file.write_text("Hello, world!")
        
        result = tools.read_file("test.txt")
        
        assert "Hello, world!" in result

    def test_write_file_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        result = tools.write_file("new.txt", "New content")
        
        assert "Wrote" in result
        assert (workspace_dir / "new.txt").read_text() == "New content"

    def test_list_dir_tool(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        (workspace_dir / "file1.txt").write_text("1")
        (workspace_dir / "file2.txt").write_text("2")
        (workspace_dir / "subdir").mkdir()
        
        result = tools.list_dir(".")
        
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    def test_run_bash_disabled_by_default(self):
        tools = self._reload_tools_with_workspace(Path("/tmp/workspace"))
        os.environ.pop("LLM_ALLOW_SHELL", None)
        result = tools.run_bash("echo hello")
        assert "disabled" in result.lower()

    def test_run_bash_enabled(self, workspace_dir):
        tools = self._reload_tools_with_workspace(workspace_dir)
        os.environ["LLM_ALLOW_SHELL"] = "1"
        
        result = tools.run_bash("echo hello")
        
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
        tools = {
            "tool_a": {
                "fn": lambda: "result_a",
                "schema": {"type": "function", "function": {"name": "tool_a"}},
            },
            "tool_b": {
                "fn": lambda: "result_b",
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
            {"content": "Hello! How can I help?", "tool_calls": None},  # First call: tool check
            {"content": "Hello! How can I help?", "tool_calls": None},  # Second call: final response
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
        assert mock_client.complete_with_tools.call_count == 2