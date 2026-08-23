"""
agent_loop.py

The tool-call loop: model -> tool call -> tool result -> model -> ...
-> final text. One function. Reuses CompletionClient and the tool
registry as-is — no new abstractions, no agent class hierarchy.

Assumes the caller already added the user's message to the session
(store.add_turn(session_id, "user", ...)) before calling this.
"""

import json

from ..sessions.store import SessionStore
from .completion_client import CompletionClient

MAX_TOOL_ROUNDS = 8


def run_agent_turn(
    store: SessionStore,
    session_id: str,
    completion_client: CompletionClient,
    tools: dict,
    max_tokens: int = 512,
    temperature: float = 0.7,
    rag_query: str = None,
) -> str:
    """Runs the tool-call loop for the pending user turn on `session_id`.
    Returns the final assistant reply text. Every step (assistant
    tool_calls, tool results, final reply) is persisted via store.add_turn
    as it happens, so eviction/budgeting sees them like any other turn."""
    tool_schemas = [t["schema"] for t in tools.values()]

    for _ in range(MAX_TOOL_ROUNDS):
        messages = store.build_messages(session_id, use_rag=rag_query is not None, query=rag_query)

        message = completion_client.complete_with_tools(
            messages, tool_schemas, max_tokens, temperature
        )

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            reply = message.get("content") or ""
            store.add_turn(session_id, "assistant", reply)
            return reply

        store.add_turn(session_id, "assistant", message.get("content") or "", tool_calls=tool_calls)

        for call in tool_calls:
            result = _execute_tool(tools, call)
            store.add_turn(session_id, "tool", result, tool_call_id=call["id"])

    return "Stopped after too many tool calls in a row — try breaking the task into smaller steps."


def _execute_tool(tools: dict, call: dict) -> str:
    name = call["function"]["name"]
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return "Error: malformed tool arguments"

    try:
        return tool["fn"](**args)
    except Exception as e:
        return f"Error running {name}: {type(e).__name__}: {e}"