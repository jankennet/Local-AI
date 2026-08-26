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
from ..tokenizer import TokenCounter


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

    def _compress_turn(self, text: str, max_tokens: int, counter: TokenCounter) -> str:
        """Compress a single turn to fit within max_tokens by truncating."""
        tokens = counter.count(text)
        if tokens <= max_tokens:
            return text
        # Truncate to roughly max_tokens (approximate)
        ratio = max_tokens / tokens
        keep_chars = int(len(text) * ratio * 0.9)  # conservative
        return text[:keep_chars] + "… [truncated]"

    def _fit_retrieved_to_budget(
        self,
        retrieved: List[Tuple[float, str, dict]],
        budget: int,
        counter: TokenCounter,
    ) -> List[Tuple[float, str, dict]]:
        """Trim/compress retrieved turns to fit within token budget."""
        if not retrieved:
            return []
        
        # First pass: compress each turn proportionally
        compressed = []
        for score, text, meta in retrieved:
            # Reserve tokens per turn (max per turn = budget / num_turns)
            max_per_turn = max(50, budget // len(retrieved))
            compressed_text = self._compress_turn(text, max_per_turn, counter)
            compressed.append((score, compressed_text, meta))
        
        # Second pass: if still over budget, drop lowest-scoring turns
        total_tokens = sum(counter.count(t) for _, t, _ in compressed) + 4 * len(compressed)
        while total_tokens > budget and compressed:
            # Drop the lowest-scoring turn
            compressed.pop()
            total_tokens = sum(counter.count(t) for _, t, _ in compressed) + 4 * len(compressed)
        
        return compressed

    def _fit_history_to_budget(
        self,
        history: list,
        budget: int,
        counter: TokenCounter,
    ) -> list:
        """Trim history from oldest to fit within token budget."""
        if not history:
            return []
        
        # Count tokens for each message
        msg_tokens = []
        total = 0
        for msg in history:
            content = msg.get("content") or ""
            tokens = counter.count(content) + 4  # +4 for role/overhead
            msg_tokens.append((tokens, msg))
            total += tokens
        
        if total <= budget:
            return history
        
        # Remove oldest messages until we fit
        kept = []
        kept_tokens = 0
        for tokens, msg in msg_tokens:
            if kept_tokens + tokens <= budget:
                kept.append(msg)
                kept_tokens += tokens
            else:
                # Try to compress this message
                content = msg.get("content") or ""
                remaining = budget - kept_tokens
                if remaining > 50:
                    compressed = self._compress_turn(content, remaining - 4, counter)
                    kept.append({**msg, "content": compressed})
                break
        
        return kept

    def build_messages(
        self,
        use_rag: bool = False,
        query: Optional[str] = None,
        rag_top_k: int = None,
        rag_initial_k: int = None,
        use_reranker: bool = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> list:
        if not token_counter:
            # Fallback to simple behavior
            msgs = [{"role": "system", "content": self.system_prompt}]
            if use_rag and query and self._vector_store:
                recent_context = " ".join(
                    (m.get("content") or "") for m in self.history[-3:] if m.get("content")
                )
                retrieval_query = f"{query} {self.summary} {recent_context}".strip()
                retrieved = self.retrieve_relevant(retrieval_query, rag_top_k, rag_initial_k, use_reranker)
                if retrieved:
                    context_lines = [f"[Relevant context]: {text}" for _, text, _ in retrieved]
                    msgs.append({"role": "system", "content": "\n".join(context_lines)})
            if self.summary:
                msgs.append({"role": "system", "content": f"[Earlier conversation summary]: {self.summary}"})
            msgs.extend(self.history)
            return msgs
        
        # Per-component budget allocation
        total_budget = token_counter.count(self.system_prompt) + 4
        if self.summary:
            total_budget += token_counter.count(self.summary) + 4
        for m in self.history:
            total_budget += token_counter.count(m.get("content") or "") + 4
        
        available_budget = max(0, total_budget - token_counter.count(self.system_prompt) - 4)
        
        # Calculate per-component budgets
        system_budget = int(available_budget * settings.budget_system_prompt_pct)
        summary_budget = int(available_budget * settings.budget_summary_pct)
        history_budget = int(available_budget * settings.budget_history_pct)
        rag_budget = min(settings.rag_token_budget, int(available_budget * settings.budget_rag_pct))
        tools_budget = int(available_budget * settings.budget_tools_pct)
        
        msgs = [{"role": "system", "content": self.system_prompt}]
        
        # Add RAG context if enabled
        rag_msgs = []
        if use_rag and query and self._vector_store:
            recent_context = " ".join(
                (m.get("content") or "") for m in self.history[-3:] if m.get("content")
            )
            retrieval_query = f"{query} {self.summary} {recent_context}".strip()
            retrieved = self.retrieve_relevant(retrieval_query, rag_top_k, rag_initial_k, use_reranker)
            if retrieved:
                retrieved = self._fit_retrieved_to_budget(retrieved, rag_budget, token_counter)
                if retrieved:
                    context_lines = [f"[Relevant context]: {text}" for _, text, _ in retrieved]
                    rag_msgs.append({"role": "system", "content": "\n".join(context_lines)})
        
        # Add summary if present
        summary_msgs = []
        if self.summary:
            summary_text = f"[Earlier conversation summary]: {self.summary}"
            if token_counter.count(summary_text) > summary_budget:
                summary_text = self._compress_turn(summary_text, summary_budget, token_counter)
            summary_msgs.append({"role": "system", "content": summary_text})
        
        # Add history (most recent first, respecting budget)
        history_msgs = self._fit_history_to_budget(self.history, history_budget, token_counter)
        
        # Combine: system + RAG + summary + history
        msgs.extend(rag_msgs)
        msgs.extend(summary_msgs)
        msgs.extend(history_msgs)
        
        return msgs
