"""
Session management package.

Provides session persistence with atomic writes and configurable backends.
"""

from .session import Session
from .store import SessionStore
from .repository import SessionRepository, JSONSessionRepository, SQLiteSessionRepository
from .eviction import EvictionStrategy, DropOldestStrategy, SummarizeOldestStrategy

__all__ = [
    "Session",
    "SessionStore",
    "SessionRepository",
    "JSONSessionRepository",
    "SQLiteSessionRepository",
    "EvictionStrategy",
    "DropOldestStrategy",
    "SummarizeOldestStrategy",
]