"""
completion_client.py

Talks to the model. Behind an interface so the session route doesn't
care HOW a completion is produced (DIP) — today it's a loopback HTTP
call to this same process's own /v1/chat/completions; later you could
swap in a direct in-process call without touching routes/sessions_router.py.
"""

from typing import Protocol
import requests


class CompletionClient(Protocol):
    def complete(self, messages: list, max_tokens: int, temperature: float) -> str: ...


class LoopbackCompletionClient:
    def __init__(self, port: int):
        self._url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        resp = requests.post(
            self._url,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
