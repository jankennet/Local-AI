"""
agent_loop.py

The tool-call loop: model -> tool call -> tool result -> model -> ...
-> final text. One function. Reuses CompletionClient and the tool
registry as-is — no new abstractions, no agent class hierarchy.

Assumes the caller already added the user's message to the session
(store.add_turn(session_id, "user", ...)) before calling this.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from ..sessions.store import SessionStore
from .completion_client import CompletionClient
from .tools import TOOLS
from ..metrics import (
    record_agent_round, record_agent_turn, record_tool_call,
)
from ..config import settings

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
DEFAULT_TOOL_TIMEOUT = 30.0
MAX_TOOL_RETRIES = 2


async def run_agent_turn(
    store: SessionStore,
    session_id: str,
    completion_client: CompletionClient,
    tools: dict,
    max_tokens: int = 512,
    temperature: float = 0.7,
    rag_query: str = None,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    max_retries: int = MAX_TOOL_RETRIES,
    rag_top_k: int = None,
    rag_initial_k: int = None,
    use_reranker: bool = None,
) -> str:
    """Runs the tool-call loop for the pending user turn on `session_id`.
    Returns the final assistant reply text. Every step (assistant
    tool_calls, tool results, final reply) is persisted via store.add_turn
    as it happens, so eviction/budgeting sees them like any other turn.

    Tool calls are executed in parallel for better performance.
    Failed tools are retried up to max_retries times.
    """
    tool_schemas = [t["schema"] for t in tools.values()]

    for round_num in range(MAX_TOOL_ROUNDS):
        record_agent_round(session_id)
        messages = store.build_messages(
            session_id,
            use_rag=rag_query is not None,
            query=rag_query,
            rag_top_k=rag_top_k,
            rag_initial_k=rag_initial_k,
            use_reranker=use_reranker,
        )

        message = completion_client.complete_with_tools(
            messages, tool_schemas, max_tokens, temperature
        )

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            reply = message.get("content") or ""
            store.add_turn(session_id, "assistant", reply)
            record_agent_turn(session_id, "assistant")
            return reply

        store.add_turn(session_id, "assistant", message.get("content") or "", tool_calls=tool_calls)
        record_agent_turn(session_id, "assistant")

        results = await _execute_tools_parallel(
            tools, tool_calls, tool_timeout, max_retries
        )

        for call, result in zip(tool_calls, results):
            store.add_turn(session_id, "tool", result, tool_call_id=call["id"])
            record_agent_turn(session_id, "tool")

    return "Stopped after too many tool calls in a row — try breaking the task into smaller steps."


async def _execute_tools_parallel(
    tools: dict,
    tool_calls: list,
    timeout: float,
    max_retries: int,
) -> list[str]:
    """Execute multiple tool calls in parallel with retry logic."""
    async def execute_with_retry(call: dict) -> str:
        name = call["function"]["name"]
        start = time.time()
        retries = 0
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_execute_tool, tools, call),
                    timeout=timeout,
                )
                record_tool_call(name, time.time() - start, True, retries)
                return result
            except asyncio.TimeoutError:
                retries += 1
                logger.warning(f"Tool {name} timed out (attempt {attempt + 1}/{max_retries + 1})")
                if attempt == max_retries:
                    record_tool_call(name, time.time() - start, False, retries)
                    return f"Error: tool '{name}' timed out after {timeout}s"
            except Exception as e:
                retries += 1
                logger.warning(f"Tool {name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    record_tool_call(name, time.time() - start, False, retries)
                    return f"Error running {name}: {type(e).__name__}: {e}"
        record_tool_call(name, time.time() - start, False, retries)
        return f"Error: tool '{name}' failed after {max_retries + 1} attempts"

    tasks = [execute_with_retry(call) for call in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=False)


def _execute_tool(tools: dict, call: dict) -> str:
    name = call["function"]["name"]
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return "Error: malformed tool arguments"

    _validate_args(tool.get("schema"), args, name)

    try:
        return tool["fn"](**args)
    except Exception as e:
        return f"Error running {name}: {type(e).__name__}: {e}"


def _validate_args(schema: Optional[dict], args: dict, tool_name: str) -> None:
    """Validate tool arguments against schema. Raises ValueError on invalid args."""
    if not schema:
        return

    params = schema.get("function", {}).get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})

    for req in required:
        if req not in args:
            raise ValueError(f"Missing required argument '{req}' for tool '{tool_name}'")

    for key, value in args.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type and not _type_matches(value, expected_type):
                raise ValueError(f"Argument '{key}' for tool '{tool_name}' expected type {expected_type}, got {type(value).__name__}")


def _type_matches(value: any, expected_type: str) -> bool:
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True
    return isinstance(value, expected)