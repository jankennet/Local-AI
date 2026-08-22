"""
completion_client.py

Talks to the model. Behind an interface so callers don't care HOW a
completion is produced (DIP) — today it's a loopback HTTP call to this
same process's own /v1/chat/completions; later you could swap in a
direct in-process call without touching callers.

Two methods, not one, because they return different shapes for
different callers:
  - complete()            -> plain string, for the existing /sessions
                              chat path that never sees tool calls.
  - complete_with_tools()  -> the raw message dict (role, content,
                              tool_calls), for the agent loop, which
                              needs to know whether the model wants to
                              call a tool before it can decide what to
                              do next.
"""

from typing import Optional, Protocol
import requests


class CompletionClient(Protocol):
    def complete(self, messages: list, max_tokens: int, temperature: float) -> str: ...

    def complete_with_tools(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> dict: ...


class LoopbackCompletionClient:
    def __init__(self, base_url: str):
        self._url = f"{base_url}/v1/chat/completions"

    def complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        message = self._post(messages, None, max_tokens, temperature)
        return message.get("content") or ""

    def complete_with_tools(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> dict:
        return self._post(messages, tools, max_tokens, temperature)

    def _post(
        self, messages: list, tools: Optional[list], max_tokens: int, temperature: float
    ) -> dict:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        resp = requests.post(self._url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]