"""
store.py

SessionStore: the one place that coordinates sessions. It depends on
three injected collaborators (DIP) instead of owning their logic:
  - TokenCounter      (how to count tokens)
  - SessionRepository (how to persist)
  - EvictionStrategy  (how to shrink an over-budget session)

This means store.py never changes when you swap tokenizer, storage
backend, or eviction policy — only the composition root (main.py) does.
"""

import time
import uuid
from typing import Callable, Dict, List, Optional

from .session import Session
from .repository import SessionRepository
from .eviction import EvictionStrategy, _session_tokens
from ..tokenizer import TokenCounter
from ..embeddings import EmbeddingService, VectorStore
from ..metrics import set_session_tokens


class SessionStore:
    def __init__(
        self,
        counter: TokenCounter,
        repository: SessionRepository,
        eviction: EvictionStrategy,
        n_ctx: int,
        reserve_for_response: int = 768,
        ttl_days: int = 30,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None,
    ):
        self._counter = counter
        self._repo = repository
        self._eviction = eviction
        self._n_ctx = n_ctx
        self._reserve = reserve_for_response
        self._ttl_seconds = ttl_days * 86400
        self._embedding_service = embedding_service
        self._vector_store_factory = vector_store_factory
        self._sessions: Dict[str, Session] = self._repo.load()
        self._init_vector_stores()

    def _init_vector_stores(self) -> None:
        if self._embedding_service:
            for s in self._sessions.values():
                s.init_vector_store(self._embedding_service, self._vector_store_factory)

    @property
    def budget(self) -> int:
        return self._n_ctx - self._reserve

    # ---- lifecycle -----------------------------------------------------
    def create_session(self, device_name: str = "unknown device",
                        system_prompt: Optional[str] = None) -> Session:
        sid = str(uuid.uuid4())
        s = Session(
            session_id=sid,
            device_name=device_name,
            system_prompt=system_prompt or "You are a helpful assistant.",
        )
        if self._embedding_service:
            s.init_vector_store(self._embedding_service, self._vector_store_factory)
        self._sessions[sid] = s
        self._repo.save(self._sessions)
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._repo.save(self._sessions)

    def list_sessions(self) -> List[Session]:
        return list(self._sessions.values())

    # ---- conversation ----------------------------------------------------
    def add_turn(self, session_id: str, role: str, content: str, **extra) -> None:
        s = self._sessions[session_id]
        turn_index = len(s.history)
        s.history.append({"role": role, "content": content, **extra})
        s.last_active = time.time()
        if self._embedding_service:
            s.add_to_vector_store(role, content, turn_index)
        self._eviction.evict(s, self._counter, self.budget)
        if self._embedding_service:
            s.init_vector_store(self._embedding_service, self._vector_store_factory)
        self._repo.save(self._sessions)
        
        # Record token usage metrics
        try:
            tokens_used = _session_tokens(s, self._counter)
            set_session_tokens(session_id, tokens_used, self.budget)
        except Exception:
            pass  # Metrics are best-effort

    def build_messages(self, session_id: str, use_rag: bool = False, query: Optional[str] = None, rag_top_k: int = None, rag_initial_k: int = None, use_reranker: bool = None) -> list:
        return self._sessions[session_id].build_messages(
            use_rag=use_rag,
            query=query,
            rag_top_k=rag_top_k,
            rag_initial_k=rag_initial_k,
            use_reranker=use_reranker,
            token_counter=self._counter,
        )

    def tokens_used(self, session_id: str) -> int:
        """Current token count for a session, using the same counting
        logic eviction uses to decide when to shrink it."""
        return _session_tokens(self._sessions[session_id], self._counter)

    # ---- age-out ---------------------------------------------------------
    def purge_expired(self) -> int:
        """Removes sessions untouched for longer than the configured TTL.
        Returns the number of sessions removed."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            self._repo.save(self._sessions)
        return len(expired)