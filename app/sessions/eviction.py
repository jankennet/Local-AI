"""
eviction.py

Eviction strategies (Strategy pattern — OCP: add a new strategy by
adding a new class, never by editing SessionStore or existing strategies).

Every strategy has the same job: given a session that's over budget,
shrink it until it fits, using the injected TokenCounter to measure.
"""

from abc import ABC, abstractmethod

from .session import Session
from ..tokenizer import TokenCounter


def _session_tokens(session: Session, counter: TokenCounter) -> int:
    total = counter.count(session.system_prompt) + 4
    if session.summary:
        total += counter.count(session.summary) + 4
    for m in session.history:
        total += counter.count(m["content"]) + 4
    return total


class EvictionStrategy(ABC):
    @abstractmethod
    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        """Mutate `session` in place until _session_tokens(session) <= budget."""
        ...


class DropOldestStrategy(EvictionStrategy):
    """Discards the oldest turns outright. Cheapest option, loses context."""

    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        while _session_tokens(session, counter) > budget and session.history:
            session.history.pop(0)


class SummarizeOldestStrategy(EvictionStrategy):
    """Folds the oldest turns into a short rolling summary instead of
    discarding them outright, so the model keeps a thread of earlier
    context. Local/cheap fold — no extra model call."""

    def evict(self, session: Session, counter: TokenCounter, budget: int) -> None:
        while _session_tokens(session, counter) > budget and session.history:
            oldest = session.history.pop(0)
            snippet = oldest["content"].strip().replace("\n", " ")[:200]
            line = f"{oldest['role']}: {snippet}"
            session.summary = f"{session.summary} | {line}" if session.summary else line
            if len(session.summary) > 1500:
                session.summary = session.summary[-1500:]

        # Pathological fallback: even system + summary alone are too big.
        if _session_tokens(session, counter) > budget and session.summary:
            session.summary = session.summary[-max(0, len(session.summary) - 500):]
