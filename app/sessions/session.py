"""
session.py

The Session entity: one conversation belonging to one device. No
behavior lives here beyond simple data — logic (eviction, persistence,
budget) is deliberately kept in separate collaborators (SRP).
"""

from dataclasses import dataclass, field, asdict, fields
import json
import time
from typing import List, Optional, Tuple, Callable

from ..embeddings import EmbeddingService, VectorStore, rerank, deduplicate_results
from ..config import settings
from ..tokenizer import TokenCounter


def _message_tokens(msg: dict, counter: TokenCounter) -> int:
    """Token estimate for a single OpenAI-style message.

    count(content)+4 alone is a safe estimate for plain messages, but
    assistant messages that carry a tool_calls JSON blob can be far
    larger — count that payload too, or over-budget sessions sneak past.
    """
    total = counter.count(msg.get("content") or "") + 4
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        total += counter.count(json.dumps(tool_calls, ensure_ascii=False))
    return total


def _truncate_text(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Compress *text* to roughly *max_tokens*, marking the cut."""
    tokens = counter.count(text)
    if max_tokens <= 0 or tokens <= max_tokens:
        return text
    ratio = max_tokens / tokens
    keep_chars = int(len(text) * ratio * 0.9)
    return text[:max(1, keep_chars)] + "… [truncated]"


def _is_rag_block(msg: dict) -> bool:
    return msg.get("role") == "system" and (msg.get("content") or "").startswith("[Relevant context]")


def _is_summary_block(msg: dict) -> bool:
    return msg.get("role") == "system" and (msg.get("content") or "").startswith("[Earlier conversation summary]")


def _pop_group_at(messages: list, index: int) -> list:
    """Remove the message at *index* plus every tool message answering its
    tool_calls (they must travel together or the API rejects the request)."""
    group = [messages.pop(index)]
    if group[0].get("tool_calls"):
        ids = {tc["id"] for tc in group[0]["tool_calls"]}
        while (
            index < len(messages)
            and messages[index].get("role") == "tool"
            and messages[index].get("tool_call_id") in ids
        ):
            group.append(messages.pop(index))
    return group


def _clamp_messages(messages: list, budget: int, counter: TokenCounter) -> list:
    """Hard safety net: shrink a fully-built message list to fit *budget*.

    Drop order (each stage re-measures with accurate tool_calls counting):
      1. RAG context blocks
      2. summary block (compress, then drop if it can't be made small)
      3. oldest history groups, atomically (never orphans a tool response)
      4. compress the newest user turn — it is never dropped
      5. last resort: truncate the leading system prompt

    Returns a NEW list; the input is never mutated.
    """
    if budget is None or budget <= 0:
        return messages

    msgs = [dict(m) for m in messages]

    def total() -> int:
        return sum(_message_tokens(m, counter) for m in msgs)

    if total() <= budget:
        return msgs

    # 1) Drop RAG context blocks.
    msgs = [m for m in msgs if not _is_rag_block(m)]
    if total() <= budget:
        return msgs

    # 2) Compress / drop the summary block(s).
    while total() > budget:
        summary_idx = next((i for i, m in enumerate(msgs) if _is_summary_block(m)), None)
        if summary_idx is None:
            break
        m = msgs[summary_idx]
        room = budget - (total() - _message_tokens(m, counter))
        if room < 24:
            del msgs[summary_idx]
        else:
            new_content = _truncate_text(m.get("content") or "", max(8, room), counter)
            if counter.count(new_content) >= counter.count(m.get("content") or ""):
                del msgs[summary_idx]  # truncation didn't shrink it
            else:
                msgs[summary_idx] = dict(m, content=new_content)

    if total() <= budget:
        return msgs

    # The newest user turn is sacred — the model needs it to respond.
    newest_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            newest_user_idx = i
            break

    # 3) Drop oldest history groups.
    while total() > budget:
        cut = None
        for i, m in enumerate(msgs):
            if i == newest_user_idx:
                continue
            if m.get("role") != "system":
                cut = i
                break
        if cut is None:
            break
        popped = _pop_group_at(msgs, cut)
        if not popped:
            break
        if newest_user_idx is not None and cut < newest_user_idx:
            newest_user_idx -= len(popped)

    if total() <= budget:
        return msgs

    # 4) Compress the newest user turn, never dropping it.
    if newest_user_idx is not None:
        m = msgs[newest_user_idx]
        room = budget - (total() - _message_tokens(m, counter))
        if room > 0:
            msgs[newest_user_idx] = dict(m, content=_truncate_text(m.get("content") or "", room, counter))

    if total() <= budget:
        return msgs

    # 5) Last resort — truncate the leading system prompt.
    sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if sys_idx is not None:
        m = msgs[sys_idx]
        room = budget - (total() - _message_tokens(m, counter))
        if room > 0:
            msgs[sys_idx] = dict(m, content=_truncate_text(m.get("content") or "", max(1, room), counter))

    return msgs


@dataclass
class Session:
    session_id: str
    device_name: str = "unknown device"
    system_prompt: str = "You are a helpful assistant."
    history: list = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    external_id: str = ""
    metadata: dict = field(default_factory=dict)

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
        """Initialize vector store lazily - only if not already initialized."""
        if self._vector_store is not None:
            return  # Already initialized
        self._embedding_service = embedding_service
        if store_factory:
            self._vector_store = store_factory(embedding_service)
        else:
            from ..embeddings import SimpleVectorStore
            self._vector_store = SimpleVectorStore(embedding_service)
        # Re-index existing history
        for i, msg in enumerate(self.history):
            content = msg.get("content") or ""
            if content.strip():
                self._vector_store.add(content, {"turn_index": i, "role": msg.get("role"), "session_id": self.session_id})

    def _ensure_vector_store(self, embedding_service: EmbeddingService, store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None) -> None:
        """Ensure vector store is initialized (lazy initialization)."""
        if self._vector_store is None:
            self.init_vector_store(embedding_service, store_factory)

    def add_to_vector_store(self, role: str, content: str, turn_index: int, embedding_service: EmbeddingService = None, store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None) -> None:
        """Add to vector store only if already initialized (lazy init happens in build_messages/retrieve_relevant)."""
        if content.strip() and self._vector_store:
            self._vector_store.add(content, {"turn_index": turn_index, "role": role, "session_id": self.session_id})

    def retrieve_relevant(self, query: str, top_k: int = None, initial_k: int = None, use_reranker: bool = None, use_dedup: bool = None, embedding_service: EmbeddingService = None, store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None) -> List[Tuple[float, str, dict]]:
        if self._vector_store is None and embedding_service:
            self._ensure_vector_store(embedding_service, store_factory)
        if not self._vector_store:
            return []
        
        # Use settings defaults if not provided
        if top_k is None:
            top_k = settings.rag_top_k
        if initial_k is None:
            initial_k = settings.rag_initial_k
        if use_reranker is None:
            use_reranker = settings.reranker_enabled
        if use_dedup is None:
            use_dedup = settings.rag_dedup_enabled
        
        # Stage 1: Broad vector search (scoped to this session only)
        filter_ = {"session_id": self.session_id}
        candidates = self._vector_store.search(query, initial_k, filter=filter_)
        
        # Stage 2: Rerank with cross-encoder
        if use_reranker and len(candidates) > top_k:
            candidates = rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]
        
        # Stage 3: Semantic deduplication
        if use_dedup and len(candidates) > 1 and self._embedding_service:
            candidates = deduplicate_results(
                candidates,
                threshold=settings.rag_dedup_threshold,
                embedding_service=self._embedding_service,
            )
        
        return candidates

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
        """Trim history to fit within token budget, keeping the MOST RECENT
        messages and dropping the oldest. Chat semantics require that the
        current user turn (newest) is always present, so if nothing fits we
        still keep the newest message (compressing it if possible).

        History is trimmed by GROUP, not by single message: an assistant
        turn with tool_calls and the tool messages answering it travel
        together, so the next /v1/chat/completions call is never handed an
        orphaned tool_call_id or tool response."""
        if not history:
            return []

        # Split into logical groups so tool responses never get orphaned.
        groups = []
        i = 0
        while i < len(history):
            msg = history[i]
            group = [msg]
            i += 1
            if msg.get("tool_calls"):
                ids = {tc["id"] for tc in msg["tool_calls"]}
                while (
                    i < len(history)
                    and history[i].get("role") == "tool"
                    and history[i].get("tool_call_id") in ids
                ):
                    group.append(history[i])
                    i += 1
            groups.append(group)

        group_tokens = []
        total = 0
        for g in groups:
            t = sum(_message_tokens(m, counter) for m in g)
            group_tokens.append(t)
            total += t

        if total <= budget:
            return list(history)

        # Keep newest groups first (drop oldest instead).
        kept_reversed = []
        kept_tokens = 0
        for t, g in reversed(list(zip(group_tokens, groups))):
            if kept_tokens + t <= budget:
                kept_reversed.append(g)
                kept_tokens += t
                continue
            # This group doesn't fit. If we haven't kept anything yet, we
            # must keep the newest group anyway — the model needs the
            # current user turn to respond.
            if not kept_reversed:
                if len(g) == 1 and g[0].get("role") == "user":
                    room = budget - kept_tokens
                    if room > 50:
                        g = [{**g[0], "content": self._compress_turn(g[0].get("content") or "", room - 4, counter)}]
                kept_reversed.append(g)
                kept_tokens += t
            break

        # Restore chronological order for the prompt.
        kept = list(reversed(kept_reversed))
        return [m for g in kept for m in g]

    def build_messages(
        self,
        use_rag: bool = False,
        query: Optional[str] = None,
        rag_top_k: int = None,
        rag_initial_k: int = None,
        use_reranker: bool = None,
        token_counter: Optional[TokenCounter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_factory: Optional[Callable[[EmbeddingService], VectorStore]] = None,
        budget: Optional[int] = None,
    ) -> list:
        """Build the message list for this session.

        *budget* — when provided, per-component budgets are derived from the
        REAL context-window budget (n_ctx − reserve) instead of the current
        content, and the final list is hard-clamped to fit. When None, the
        legacy content-derived allocation is used (no clamp).
        """
        if not token_counter:
            # Fallback to simple behavior
            msgs = [{"role": "system", "content": self.system_prompt}]
            # Initialize vector store lazily on first RAG query
            if use_rag and self._vector_store is None and embedding_service:
                self._ensure_vector_store(embedding_service, vector_store_factory)
            if use_rag and query and self._vector_store:
                recent_context = " ".join(
                    (m.get("content") or "") for m in self.history[-3:] if m.get("content")
                )
                retrieval_query = f"{query} {self.summary} {recent_context}".strip()
                retrieved = self.retrieve_relevant(retrieval_query, rag_top_k, rag_initial_k, use_reranker, True, embedding_service, vector_store_factory)
                if retrieved:
                    context_lines = [f"[Relevant context]: {text}" for _, text, _ in retrieved]
                    msgs.append({"role": "system", "content": "\n".join(context_lines)})
            if self.summary:
                msgs.append({"role": "system", "content": f"[Earlier conversation summary]: {self.summary}"})
            msgs.extend(self.history)
            return msgs
        
        # Per-component budget allocation. With a real budget we slice from
        # the context window; without one we fall back to the legacy
        # content-derived allocation (budget is circular with content, but
        # preserved for backward compatibility).
        if budget is not None:
            available_budget = max(0, budget)
        else:
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
        
        # Initialize vector store lazily on first RAG query
        if use_rag and self._vector_store is None and embedding_service:
            self._ensure_vector_store(embedding_service, vector_store_factory)

        # Add RAG context if enabled
        rag_msgs = []
        if use_rag and query and self._vector_store:
            recent_context = " ".join(
                (m.get("content") or "") for m in self.history[-3:] if m.get("content")
            )
            retrieval_query = f"{query} {self.summary} {recent_context}".strip()
            retrieved = self.retrieve_relevant(retrieval_query, rag_top_k, rag_initial_k, use_reranker, True, embedding_service, vector_store_factory)
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

        # Hard safety net: with a real budget, guarantee the final list fits.
        if budget is not None:
            msgs = _clamp_messages(msgs, budget, token_counter)

        return msgs

    def get_token_breakdown(self, counter: TokenCounter) -> dict:
        """Get detailed token usage breakdown per component.
        
        Returns a dict with:
        - total: Total tokens in session
        - system_prompt: Tokens for system prompt
        - summary: Tokens for conversation summary
        - history: Tokens for conversation history
        - history_count: Number of history messages
        - rag_estimate: Estimated RAG tokens (if RAG were used)
        - breakdown_by_role: Token counts by role
        """
        system_tokens = counter.count(self.system_prompt) + 4
        summary_tokens = counter.count(self.summary) + 4 if self.summary else 0
        
        history_tokens = 0
        breakdown_by_role = {"user": 0, "assistant": 0, "tool": 0, "system": system_tokens}
        
        for msg in self.history:
            content = msg.get("content") or ""
            role = msg.get("role", "unknown")
            tokens = counter.count(content) + 4
            history_tokens += tokens
            breakdown_by_role[role] = breakdown_by_role.get(role, 0) + tokens
        
        total = system_tokens + summary_tokens + history_tokens
        
        return {
            "total": total,
            "system_prompt": system_tokens,
            "summary": summary_tokens,
            "history": history_tokens,
            "history_count": len(self.history),
            "breakdown_by_role": breakdown_by_role,
        }
