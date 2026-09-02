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

import logging
import time
import uuid
from typing import Callable, Dict, List, Optional

from .session import Session
from .repository import SessionRepository
from .eviction import EvictionStrategy, _session_tokens
from ..tokenizer import TokenCounter
from ..embeddings import EmbeddingService, VectorStore, deduplicate_tool_history
from ..metrics import set_session_tokens

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(
        self,
        counter: TokenCounter,
        repository: SessionRepository,
        eviction: EvictionStrategy,
        n_ctx: int,
        reserve_for_response: int = 768,
        ttl_days: int = 30,
        max_sessions_per_user: int = 50,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None,
    ):
        self._counter = counter
        self._repo = repository
        self._eviction = eviction
        self._n_ctx = n_ctx
        self._reserve = reserve_for_response
        self._ttl_seconds = ttl_days * 86400
        self._max_sessions_per_user = max_sessions_per_user
        self._embedding_service = embedding_service
        self._vector_store_factory = vector_store_factory
        self._sessions: Dict[str, Session] = self._repo.load()
        # Vector stores are lazily initialized on first use (RAG or add_to_vector_store)

    @property
    def budget(self) -> int:
        return self._n_ctx - self._reserve

    # ---- lifecycle -----------------------------------------------------
    def create_session(
        self,
        device_name: str = "unknown device",
        system_prompt: Optional[str] = None,
        external_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Session:
        # Enforce max sessions per user (based on external_id prefix)
        if external_id and self._max_sessions_per_user > 0:
            source_user = self._parse_external_id(external_id)
            if source_user:
                user_sessions = self._count_user_sessions(source_user)
                if user_sessions >= self._max_sessions_per_user:
                    self._evict_oldest_user_session(source_user)

        sid = str(uuid.uuid4())
        s = Session(
            session_id=sid,
            device_name=device_name,
            system_prompt=system_prompt or "You are a helpful assistant.",
            external_id=external_id,
            metadata=metadata or {},
        )
        # Vector store is lazily initialized when first needed (RAG or add_to_vector_store)
        self._sessions[sid] = s
        self._repo.save(self._sessions)
        return s

    def _parse_external_id(self, external_id: str) -> Optional[str]:
        """Parse external_id to extract source:user_id. Returns 'source:user_id' or None."""
        parts = external_id.split(":", 2)
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}"
        return None

    def _count_user_sessions(self, source_user: str) -> int:
        count = 0
        for s in self._sessions.values():
            if s.external_id.startswith(source_user + ":"):
                count += 1
        return count

    def _evict_oldest_user_session(self, source_user: str) -> None:
        oldest = None
        oldest_time = float("inf")
        for sid, s in self._sessions.items():
            if s.external_id.startswith(source_user + ":"):
                if s.last_active < oldest_time:
                    oldest_time = s.last_active
                    oldest = sid
        if oldest:
            del self._sessions[oldest]
            self._repo.save(self._sessions)

    def get_by_external_id(self, external_id: str) -> Optional[Session]:
        for s in self._sessions.values():
            if s.external_id == external_id:
                return s
        return None

    def list_by_user(self, user_id: str) -> List[Session]:
        results = []
        for s in self._sessions.values():
            if s.external_id.startswith(f"discord:{user_id}:") or s.external_id.startswith(f"vscode:{user_id}:"):
                results.append(s)
        return results

    def list_by_source(self, source: str) -> List[Session]:
        prefix = f"{source}:"
        return [s for s in self._sessions.values() if s.external_id.startswith(prefix)]

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
            s.add_to_vector_store(role, content, turn_index, self._embedding_service, self._vector_store_factory)
        
        # Deduplicate tool history if this is a tool message
        if role == "tool" and self._embedding_service:
            s.history = deduplicate_tool_history(
                s.history,
                embedding_service=self._embedding_service,
            )
        
        self._eviction.evict(s, self._counter, self.budget)
        self._repo.save(self._sessions)
        
        # Record token usage metrics
        try:
            tokens_used = _session_tokens(s, self._counter)
            set_session_tokens(session_id, tokens_used, self.budget)
            logger.info(f"Session {session_id[:8]}: {tokens_used}/{self.budget} tokens ({tokens_used/self.budget*100:.1f}%) role={role}")
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
            embedding_service=self._embedding_service if use_rag else None,
            vector_store_factory=self._vector_store_factory if use_rag else None,
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