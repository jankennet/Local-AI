"""
Shared test configuration and fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LLM_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("LLM_EMBEDDING_MODEL_CODE", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("LLM_VECTOR_BACKEND", "simple")
os.environ.setdefault("LLM_RERANKER_ENABLED", "false")
os.environ.setdefault("LLM_ORCHESTRATOR_ENABLED", "false")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable token counts."""
    from app.tokenizer import TokenCounter

    class MockTokenCounter(TokenCounter):
        def count(self, text: str) -> int:
            if not text:
                return 0
            return max(1, len(text) // 4)

    return MockTokenCounter()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for fast tests."""
    from app.embeddings import EmbeddingService
    import numpy as np
    
    class MockEmbeddingService(EmbeddingService):
        def __init__(self):
            self._model_name = "mock"
            self._dimension = 384
            # Deterministic embeddings based on text hash
            self._cache = {}
        
        def _get_vec(self, text: str):
            if text not in self._cache:
                # Create deterministic vector from text hash
                import hashlib
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                rng = np.random.default_rng(hash_val)
                vec = rng.random(self._dimension).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                self._cache[text] = vec
            return self._cache[text]

        def embed(self, texts):
            return np.stack([self._get_vec(t) for t in texts])

        def embed_single(self, text):
            return self._get_vec(text)

        @property
        def dimension(self):
            return self._dimension
        
        @property
        def model_name(self):
            return self._model_name

    return MockEmbeddingService()


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": "test-session-123",
        "device_name": "test-device",
        "system_prompt": "You are a helpful assistant.",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ],
        "summary": "User greeted, assistant responded politely.",
        "created_at": 1234567890.0,
        "last_active": 1234567890.0,
    }


@pytest.fixture
def workspace_dir(temp_dir):
    """Create a workspace directory for file tool tests."""
    workspace = temp_dir / "workspace"
    workspace.mkdir()
    old_workspace = os.environ.get("LLM_WORKSPACE_DIR")
    os.environ["LLM_WORKSPACE_DIR"] = str(workspace)
    yield workspace
    if old_workspace:
        os.environ["LLM_WORKSPACE_DIR"] = old_workspace
    else:
        os.environ.pop("LLM_WORKSPACE_DIR", None)