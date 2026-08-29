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
import atexit
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path

from .session import Session

_lock = threading.Lock()


class SessionRepository(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Session]: ...

    @abstractmethod
    def save(self, sessions: Dict[str, Session]) -> None: ...


class JSONSessionRepository(SessionRepository):
    """
    JSON-based session repository with true atomic persistence.
    
    Features:
    - Atomic writes using temp file + fsync + os.replace
    - Crash recovery: cleans up leftover temp files on startup
    - Durability: fsync ensures data reaches disk before replace
    - Thread-safe: global lock prevents concurrent writes
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self._cleanup_temp_files()
        atexit.register(self._cleanup_temp_files)

    def _cleanup_temp_files(self) -> None:
        """Remove any leftover temp files from crashed writes."""
        try:
            if self._temp_path.exists():
                self._temp_path.unlink()
        except Exception:
            pass  # Best effort cleanup

    def load(self) -> Dict[str, Session]:
        if not self.path.exists():
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
            # Write to temp file
            with open(self._temp_path, "w") as f:
                json.dump({sid: s.to_dict() for sid, s in sessions.items()}, f)
                f.flush()
                os.fsync(f.fileno())  # Ensure data reaches disk
            
            # Atomic replace (POSIX guarantees atomicity)
            os.replace(self._temp_path, self.path)
            
            # fsync the directory to ensure the rename is durable
            dir_fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


class SQLiteSessionRepository(SessionRepository):
    """
    SQLite-based session repository with full ACID guarantees.
    
    Features:
    - True atomic transactions (BEGIN/COMMIT/ROLLBACK)
    - WAL mode for concurrent reads during writes
    - Crash recovery via SQLite's built-in recovery
    - Better performance for large session counts
    - Supports concurrent access from multiple processes
    """
    
    def __init__(self, path: str, wal_mode: bool = True):
        import sqlite3
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL" if wal_mode else "PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_active REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_last_active ON sessions(last_active)")
        self._conn.commit()
    
    def load(self) -> Dict[str, Session]:
        cursor = self._conn.execute("SELECT session_id, data FROM sessions")
        result = {}
        for session_id, data in cursor:
            try:
                session_dict = json.loads(data)
                result[session_id] = Session.from_dict(session_dict)
            except Exception:
                continue  # Skip corrupted sessions
        return result
    
    def save(self, sessions: Dict[str, Session]) -> None:
        # Use a transaction for atomicity
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            # Get existing session IDs
            existing = {row[0] for row in cursor.execute("SELECT session_id FROM sessions")}
            current = set(sessions.keys())
            
            # Delete removed sessions
            for sid in existing - current:
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            
            # Upsert current sessions
            for sid, session in sessions.items():
                data = json.dumps(session.to_dict())
                cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, data, created_at, last_active) VALUES (?, ?, ?, ?)",
                    (sid, data, session.created_at, session.last_active)
                )
            
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
