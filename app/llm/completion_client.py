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
  - complete_stream()      -> async generator for streaming responses
  - complete_with_tools_stream() -> async generator for streaming tool calls
"""

import json
import time
from typing import AsyncGenerator, Optional, Protocol
import httpx

from ..metrics import record_completion


class CompletionClient(Protocol):
    def complete(self, messages: list, max_tokens: int, temperature: float) -> str: ...

    def complete_with_tools(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> dict: ...

    async def complete_stream(
        self, messages: list, max_tokens: int, temperature: float
    ) -> AsyncGenerator[str, None]: ...

    async def complete_with_tools_stream(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> AsyncGenerator[dict, None]: ...


class LoopbackCompletionClient:
    def __init__(self, base_url: str, model_name: str = "local"):
        self._url = f"{base_url}/v1/chat/completions"
        self._model_name = model_name
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    def complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        message = self._post_sync(messages, None, max_tokens, temperature)
        return message.get("content") or ""

    def complete_with_tools(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> dict:
        return self._post_sync(messages, tools, max_tokens, temperature)

    def _post_sync(
        self, messages: list, tools: Optional[list], max_tokens: int, temperature: float
    ) -> dict:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        start = time.time()
        try:
            resp = httpx.post(self._url, json=payload, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            record_completion(self._model_name, time.time() - start, prompt_tokens, completion_tokens)
            return message
        except Exception as e:
            record_completion(self._model_name, time.time() - start, 0, 0, error=type(e).__name__)
            raise

    async def complete_stream(
        self, messages: list, max_tokens: int, temperature: float
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens as they arrive."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        start = time.time()
        total_tokens = 0
        try:
            async with self._client.stream("POST", self._url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                total_tokens += 1
                                yield content
                        except json.JSONDecodeError:
                            continue
            record_completion(self._model_name, time.time() - start, 0, total_tokens)
        except Exception as e:
            record_completion(self._model_name, time.time() - start, 0, 0, error=type(e).__name__)
            raise

    async def complete_with_tools_stream(
        self, messages: list, tools: list, max_tokens: int, temperature: float
    ) -> AsyncGenerator[dict, None]:
        """Stream completion with tool calls as they arrive."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "stream": True,
        }
        start = time.time()
        tool_calls_buffer = []
        content_buffer = ""
        try:
            async with self._client.stream("POST", self._url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            # Handle content streaming
                            content = delta.get("content")
                            if content:
                                content_buffer += content
                                yield {"type": "content", "content": content}
                            
                            # Handle tool calls streaming
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    index = tc.get("index", 0)
                                    while len(tool_calls_buffer) <= index:
                                        tool_calls_buffer.append({"id": "", "function": {"name": "", "arguments": ""}})
                                    if tc.get("id"):
                                        tool_calls_buffer[index]["id"] = tc["id"]
                                    if tc.get("function", {}).get("name"):
                                        tool_calls_buffer[index]["function"]["name"] = tc["function"]["name"]
                                    if tc.get("function", {}).get("arguments"):
                                        tool_calls_buffer[index]["function"]["arguments"] += tc["function"]["arguments"]
                                yield {"type": "tool_calls", "tool_calls": tool_calls_buffer.copy()}
                            
                            # Check for finish reason
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                yield {"type": "finish", "finish_reason": finish_reason, "content": content_buffer, "tool_calls": tool_calls_buffer}
                                break
                        except json.JSONDecodeError:
                            continue
            record_completion(self._model_name, time.time() - start, 0, len(content_buffer) // 4)
        except Exception as e:
            record_completion(self._model_name, time.time() - start, 0, 0, error=type(e).__name__)
            raise

    async def aclose(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()