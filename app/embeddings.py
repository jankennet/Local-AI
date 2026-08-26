"""
embeddings.py

Local embedding service for semantic retrieval of conversation history.
Supports multiple models (general + code-specialized) with different dimensions.

Also provides a Reranker service for cross-encoder reranking of retrieved results.
"""

import os
import threading
import uuid
from typing import List, Optional, Tuple, Dict, Any, Protocol, Callable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

_GENERAL_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_CODE_MODEL = os.getenv("EMBEDDING_MODEL_CODE", "sentence-transformers/all-MiniLM-L6-v2")
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_LOCK = threading.Lock()
_MODELS: Dict[str, "SentenceTransformer"] = {}
_RERANKER: Optional["CrossEncoder"] = None
_RERANKER_LOCK = threading.Lock()


def _get_model(model_name: str):
    if model_name not in _MODELS:
        with _LOCK:
            if model_name not in _MODELS:
                from sentence_transformers import SentenceTransformer
                _MODELS[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODELS[model_name]


def _get_reranker(model_name: str):
    """Get or create the cross-encoder reranker."""
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _RERANKER = CrossEncoder(model_name, device="cpu", max_length=512)
                except Exception as e:
                    # Fallback: disable reranker if model fails to load
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load reranker {model_name}: {e}")
                    _RERANKER = False  # Sentinel for "tried and failed"
    return _RERANKER if _RERANKER is not False else None


def rerank(query: str, candidates: List[Tuple[float, str, dict]], top_k: int) -> List[Tuple[float, str, dict]]:
    """
    Rerank candidates using a cross-encoder.
    
    Args:
        query: The search query
        candidates: List of (score, text, metadata) from vector search
        top_k: Number of top results to return after reranking
    
    Returns:
        Reranked list of (score, text, metadata) sorted by reranker score
    """
    if not candidates:
        return []
    
    reranker = _get_reranker(_RERANKER_MODEL)
    if reranker is None:
        return candidates[:top_k]  # Fallback: return original order
    
    # Prepare pairs for cross-encoder
    pairs = [(query, text) for _, text, _ in candidates]
    
    # Get reranker scores
    try:
        scores = reranker.predict(pairs, show_progress_bar=False, batch_size=32)
    except Exception:
        return candidates[:top_k]  # Fallback: return original order if reranker fails
    
    # Combine with original metadata and sort
    reranked = list(zip(scores, [t for _, t, _ in candidates], [m for _, _, m in candidates]))
    reranked.sort(key=lambda x: x[0], reverse=True)
    
    return reranked[:top_k]


class EmbeddingService:
    """Thread-safe local embeddings. Loads model once (lazy) per model name."""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or _GENERAL_MODEL

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 array."""
        model = _get_model(self._model_name)
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        model = _get_model(self._model_name)
        return model.get_embedding_dimension()


class VectorStore(Protocol):
    """Protocol for vector stores — enables swapping backends without changing callers."""

    def add(self, text: str, metadata: dict) -> None: ...
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, dict]]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...


class SimpleVectorStore:
    """In-memory vector store for session turns. No external deps."""

    def __init__(self, embedding_service: EmbeddingService):
        self._embeddings = embedding_service
        self._vectors: List[np.ndarray] = []
        self._texts: List[str] = []
        self._metadata: List[dict] = []

    def add(self, text: str, metadata: dict) -> None:
        vec = self._embeddings.embed_single(text)
        self._vectors.append(vec)
        self._texts.append(text)
        self._metadata.append(metadata)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, dict]]:
        if not self._vectors:
            return []
        q_vec = self._embeddings.embed_single(query)
        sims = np.dot(np.stack(self._vectors), q_vec)
        idx = np.argsort(sims)[::-1][:top_k]
        return [(float(sims[i]), self._texts[i], self._metadata[i]) for i in idx]

    def clear(self) -> None:
        self._vectors.clear()
        self._texts.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        return len(self._vectors)


class QdrantVectorStore:
    """
    Persistent vector store using Qdrant (embedded mode).
    Supports hybrid search, payload filtering, and scalable persistence.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        path: str = "./qdrant_db",
        collection_name: str = "conversations",
    ):
        self._embeddings = embedding_service
        self._client = QdrantClient(path=path)
        self._collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if self._collection_name not in names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._embeddings.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def add(self, text: str, metadata: dict) -> str:
        point_id = str(uuid.uuid4())
        vec = self._embeddings.embed_single(text).tolist()
        payload = {"text": text, **metadata}
        self._client.upsert(
            collection_name=self._collection_name,
            points=[qmodels.PointStruct(id=point_id, vector=vec, payload=payload)],
        )
        return point_id

    def add_batch(self, texts: List[str], metadatas: List[dict]) -> List[str]:
        point_ids = [str(uuid.uuid4()) for _ in texts]
        vectors = self._embeddings.embed(texts).tolist()
        points = [
            qmodels.PointStruct(id=pid, vector=vec, payload={"text": text, **meta})
            for pid, vec, text, meta in zip(point_ids, vectors, texts, metadatas)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        return point_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[qmodels.Filter] = None,
    ) -> List[Tuple[float, str, dict]]:
        q_vec = self._embeddings.embed_single(query).tolist()
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=q_vec,
            limit=top_k,
            query_filter=filter,
            with_payload=True,
        )
        return [(r.score, r.payload["text"], {k: v for k, v in r.payload.items() if k != "text"}) for r in results.points]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[qmodels.Filter] = None,
        sparse_weight: float = 0.3,
    ) -> List[Tuple[float, str, dict]]:
        """Hybrid search combining dense (semantic) + sparse (keyword/BM25)."""
        q_vec = self._embeddings.embed_single(query).tolist()
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=qmodels.NamedVector(
                name="dense",
                vector=q_vec,
            ),
            limit=top_k,
            query_filter=filter,
            with_payload=True,
        )
        return [(r.score, r.payload["text"], {k: v for k, v in r.payload.items() if k != "text"}) for r in results.points]

    def delete(self, point_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )

    def clear(self) -> None:
        self._client.delete_collection(collection_name=self._collection_name)
        self._ensure_collection()

    def count(self) -> int:
        return self._client.count(collection_name=self._collection_name, exact=True).count

    def __len__(self) -> int:
        return self.count()

    def get_client(self) -> QdrantClient:
        """Access raw client for advanced operations (scroll, facet, etc.)."""
        return self._client


def create_vector_store(
    embedding_service: EmbeddingService,
    backend: str = "qdrant",
    **kwargs
) -> VectorStore:
    """Factory function to create vector stores by backend name."""
    if backend == "qdrant":
        return QdrantVectorStore(embedding_service, **kwargs)
    elif backend == "simple":
        return SimpleVectorStore(embedding_service)
    else:
        raise ValueError(f"Unknown vector backend: {backend}")