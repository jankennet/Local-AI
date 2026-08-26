"""
session.py

The Session entity: one conversation belonging to one device. No
behavior lives here beyond simple data — logic (eviction, persistence,
budget) is deliberately kept in separate collaborators (SRP).
"""

from dataclasses import dataclass, field, asdict, fields
import time
from typing import List, Optional, Tuple, Callable

from ..embeddings import EmbeddingService, VectorStore, rerank
from ..config import settings


@dataclass
class Session:
    session_id: str
    device_name: str = "unknown device"
    system_prompt: str = "You are a helpful assistant."
    history: list = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Non-serialized runtime fields
    _vector_store: Optional[VectorStore] = field(default=None, init=False, repr=False, compare=False)
    _embedding_service: Optional[EmbeddingService] = field(default=None, init=False, repr=False, compare=False)

    def to_dict(self) -> dict:
        # Only serialize actual dataclass fields (excludes init=False fields)
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def init_vector_store(
        self,
        embedding_service: EmbeddingService,
        store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None,
    ) -> None:
        self._embedding_service = embedding_service
        if store_factory:
            self._vector_store = store_factory(embedding_service)
        else:
            from ..embeddings import SimpleVectorStore
            self._vector_store = SimpleVectorStore(embedding_service)
        for i, msg in enumerate(self.history):
            content = msg.get("content") or ""
            if content.strip():
                self._vector_store.add(content, {"turn_index": i, "role": msg.get("role")})

    def add_to_vector_store(self, role: str, content: str, turn_index: int) -> None:
        if self._vector_store and content.strip():
            self._vector_store.add(content, {"turn_index": turn_index, "role": role})

    def retrieve_relevant(self, query: str, top_k: int = None, initial_k: int = None, use_reranker: bool = None) -> List[Tuple[float, str, dict]]:
        if not self._vector_store:
            return []
        
        # Use settings defaults if not provided
        if top_k is None:
            top_k = settings.rag_top_k
        if initial_k is None:
            initial_k = settings.rag_initial_k
        if use_reranker is None:
            use_reranker = settings.reranker_enabled
        
        # Stage 1: Broad vector search
        candidates = self._vector_store.search(query, initial_k)
        
        # Stage 2: Rerank with cross-encoder
        if use_reranker and len(candidates) > top_k:
            return rerank(query, candidates, top_k)
        
        return candidates[:top_k]

    def build_messages(self, use_rag: bool = False, query: Optional[str] = None, rag_top_k: int = None, rag_initial_k: int = None, use_reranker: bool = None) -> list:
        msgs = [{"role": "system", "content": self.system_prompt}]

        if use_rag and query and self._vector_store:
            # Build a better retrieval query: combine user message with recent context + summary
            recent_context = " ".join(
                (m.get("content") or "") for m in self.history[-3:] if m.get("content")
            )
            retrieval_query = f"{query} {self.summary} {recent_context}".strip()
            retrieved = self.retrieve_relevant(retrieval_query, rag_top_k, rag_initial_k, use_reranker)
            if retrieved:
                context_lines = [f"[Relevant context]: {text}" for _, text, _ in retrieved]
                msgs.append({"role": "system", "content": "\n".join(context_lines)})

        if self.summary:
            msgs.append({"role": "system",
                         "content": f"[Earlier conversation summary]: {self.summary}"})
        msgs.extend(self.history)
        return msgs
