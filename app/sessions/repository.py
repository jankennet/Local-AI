"""
repository.py

Persistence for sessions, behind an interface (Repository pattern).
SessionStore depends on SessionRepository, never on "JSON file" directly
(DIP) — so swapping to SQLite or Redis later is a new class here, not a
rewrite of SessionStore.
"""

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Dict

from .session import Session

_lock = threading.Lock()


class SessionRepository(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Session]: ...

    @abstractmethod
    def save(self, sessions: Dict[str, Session]) -> None: ...


class JSONSessionRepository(SessionRepository):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[str, Session]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            return {sid: Session.from_dict(data) for sid, data in raw.items()}
        except Exception:
            # Corrupt file: start fresh rather than crash the whole server.
            return {}

    def save(self, sessions: Dict[str, Session]) -> None:
        with _lock:
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({sid: s.to_dict() for sid, s in sessions.items()}, f)
            os.replace(tmp_path, self.path)  # atomic-ish write, avoids truncated file on crash
