"""
cleanup.py

Background age-out sweep. One function, one job: periodically ask the
store to purge sessions past their TTL. Wired up as an asyncio task in
main.py's lifespan — nothing else needs to know this exists.
"""

import asyncio

from .sessions.store import SessionStore
from .metrics import record_session_expired


async def periodic_cleanup(store: SessionStore, interval_seconds: int):
    while True:
        removed = store.purge_expired()
        if removed:
            print(f"[cleanup] purged {removed} expired session(s)")
            for _ in range(removed):
                record_session_expired()
        await asyncio.sleep(interval_seconds)
