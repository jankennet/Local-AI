"""
eviction.py

Eviction strategies (Strategy pattern — OCP: add a new strategy by
adding a new class, never by editing SessionStore or existing strategies).

Every strategy has the same job: given a session that's over budget,
shrink it until it fits, using the injected TokenCounter to measure.

Evicts by GROUP, not by single message: an assistant message with
tool_calls and the tool-result message(s) answering it must be dropped
together, or the next request to /v1/chat/completions gets rejected
(a tool_call_id with no matching tool response, or vice versa, is an
invalid request under the OpenAI-compatible API).
"""

from abc import ABC, abstractmethod

from .session import Session
from ..tokenizer import TokenCounter


def _session_tokens(session: Session, counter: TokenCounter) -> int:
    total = counter.count(session.system_prompt) + 4
    if session.summary:
        total += counter.count(session.summary) + 4
    for m in session.history:
        total += counter.count(m.get("content") or "") + 4
    return total


def _pop_oldest_group(history: list) -> list:
    """Pop the oldest logical unit from the front of history: either one
    plain turn, or an assistant/tool_calls turn plus every tool message
    that answers it. Returns the popped messages as a list."""
    first = history.pop(0)
    group = [first]
    if first.get("tool_calls"):
        ids = {tc["id"] for tc in first["tool_calls"]}
        while history and history[0].get("role") == "tool" and history[0].get("tool_call_id") in ids:
            group.append(history.pop(0))
    return group


class EvictionStrategy(ABC):
    @abstractmethod
    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        """Mutate `session` in place until _session_tokens(session) <= budget."""
        ...


class DropOldestStrategy(EvictionStrategy):
    """Discards the oldest turns outright. Cheapest option, loses context."""

    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        while _session_tokens(session, counter) > budget and session.history:
            _pop_oldest_group(session.history)


class SummarizeOldestStrategy(EvictionStrategy):
    """Folds the oldest turns into a short rolling summary instead of
    discarding them outright, so the model keeps a thread of earlier
    context. Local/cheap fold — no extra model call."""

    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        while _session_tokens(session, counter) > budget and session.history:
            group = _pop_oldest_group(session.history)
            for msg in group:
                content = (msg.get("content") or "").strip().replace("\n", " ")[:200]
                if not content and msg.get("tool_calls"):
                    names = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                    content = f"[called tool(s): {names}]"
                line = f"{msg['role']}: {content}"
                session.summary = f"{session.summary} | {line}" if session.summary else line
            if len(session.summary) > 1500:
                session.summary = session.summary[-1500:]

        # Pathological fallback: even system + summary alone are too big.
        if _session_tokens(session, counter) > budget and session.summary:
            session.summary = session.summary[-max(0, len(session.summary) - 500):]