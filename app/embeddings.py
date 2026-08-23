"""
embeddings.py

Local embedding service for semantic retrieval of conversation history.
Supports multiple models (general + code-specialized) with different dimensions.
"""

import os
import threading
from typing import List, Optional, Tuple, Dict

import numpy as np

_GENERAL_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_CODE_MODEL = os.getenv("EMBEDDING_MODEL_CODE", "microsoft/codebert-base")
_LOCK = threading.Lock()
_MODELS: Dict[str, "SentenceTransformer"] = {}


def _get_model(model_name: str):
    if model_name not in _MODELS:
        with _LOCK:
            if model_name not in _MODELS:
                from sentence_transformers import SentenceTransformer
                _MODELS[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODELS[model_name]


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
        return model.get_sentence_embedding_dimension()


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