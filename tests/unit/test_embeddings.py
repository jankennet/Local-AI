"""
Unit tests for embeddings.py
"""

import numpy as np
import pytest

from app.embeddings import (
    EmbeddingService,
    SimpleVectorStore,
    VectorStore,
    create_vector_store,
    rerank,
)


class TestEmbeddingService:
    def test_embed_single(self, mock_embedding_service):
        """Test single text embedding."""
        vec = mock_embedding_service.embed_single("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_batch(self, mock_embedding_service):
        """Test batch embedding."""
        vecs = mock_embedding_service.embed(["hello", "world", "test"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (3, 384)

    def test_normalized_embeddings(self, mock_embedding_service):
        """Test embeddings are normalized (unit vectors)."""
        vec = mock_embedding_service.embed_single("test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_dimension_property(self, mock_embedding_service):
        assert mock_embedding_service.dimension == 384

    def test_model_name_property(self, mock_embedding_service):
        assert mock_embedding_service.model_name == "mock"


class TestSimpleVectorStore:
    def test_add_and_search(self, mock_embedding_service):
        """Test adding and searching vectors."""
        store = SimpleVectorStore(mock_embedding_service)
        
        store.add("hello world", {"source": "test"})
        store.add("goodbye world", {"source": "test"})
        
        results = store.search("hello", top_k=1)
        assert len(results) == 1
        score, text, meta = results[0]
        # The mock uses deterministic embeddings, so the search should return the closest match
        assert "hello" in text.lower() or "world" in text.lower()
        assert meta["source"] == "test"

    def test_search_empty(self, mock_embedding_service):
        """Test search on empty store."""
        store = SimpleVectorStore(mock_embedding_service)
        results = store.search("anything", top_k=5)
        assert results == []

    def test_clear(self, mock_embedding_service):
        """Test clearing the store."""
        store = SimpleVectorStore(mock_embedding_service)
        store.add("test", {})
        store.clear()
        assert len(store) == 0

    def test_len(self, mock_embedding_service):
        store = SimpleVectorStore(mock_embedding_service)
        assert len(store) == 0
        store.add("one", {})
        assert len(store) == 1
        store.add("two", {})
        assert len(store) == 2


class TestRerank:
    def test_rerank_empty(self):
        """Rerank empty list returns empty."""
        results = rerank("query", [], 5)
        assert results == []

    def test_rerank_returns_top_k(self, mock_embedding_service):
        """Rerank returns at most top_k results."""
        candidates = [
            (0.5, "first", {}),
            (0.3, "second", {}),
            (0.1, "third", {}),
        ]
        results = rerank("query", candidates, top_k=2)
        assert len(results) == 2

    def test_rerank_disabled_when_few_candidates(self):
        """When candidates <= top_k, no reranking happens."""
        candidates = [(0.5, "first", {})]
        results = rerank("query", candidates, top_k=5)
        # When there are fewer candidates than top_k, it returns the candidates as-is
        # The rerank function still gets called but returns the same order
        assert len(results) == 1
        assert results[0][1] == "first"


class TestCreateVectorStore:
    def test_create_simple(self, mock_embedding_service):
        store = create_vector_store(mock_embedding_service, backend="simple")
        assert isinstance(store, SimpleVectorStore)

    def test_unknown_backend_raises(self, mock_embedding_service):
        with pytest.raises(ValueError, match="Unknown vector backend"):
            create_vector_store(mock_embedding_service, backend="unknown")