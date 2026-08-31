"""
agent_loop_streaming.py

Streaming version of the agent loop that yields events in real-time.
Events include: tool_calls, tool_results, content_delta, round_start, round_end, error, done.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

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


@dataclass
class StreamEvent:
    """Event emitted during streaming agent execution."""
    type: str  # "round_start", "content_delta", "tool_call", "tool_result", "round_end", "error", "done"
    data: dict
    round_num: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "round": self.round_num,
        })


def estimate_response_reserve(query: str, min_reserve: int, max_reserve: int) -> int:
    """Estimate response token reserve based on query complexity."""
    q_lower = query.lower().strip()
    length = len(q_lower)
    words = len(q_lower.split())
    
    if length < 50 and words < 10:
        score = 0.0
    elif length < 200 and words < 40:
        score = 0.3
    else:
        score = 0.6
    
    code_indicators = ['code', 'function', 'class', 'debug', 'error', 'implement',
                       'write', 'create', 'build', 'refactor', 'optimize', 'fix',
                       'how to', 'example', 'script', 'api', 'database', 'sql',
                       'algorithm', 'data structure', 'async', 'thread']
    for ind in code_indicators:
        if ind in q_lower:
            score += 0.1
            break
    
    if '?' in q_lower and q_lower.count('?') > 1:
        score += 0.15
    if any(w in q_lower for w in ['and', 'also', 'then', 'next', 'after']):
        score += 0.1
    
    score = min(1.0, score)
    return int(min_reserve + (max_reserve - min_reserve) * score)


async def run_agent_turn_streaming(
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
) -> AsyncGenerator[StreamEvent, None]:
    """
    Runs the tool-call loop with streaming events.
    Yields StreamEvent for each step of the process.
    """
    tool_schemas = [t["schema"] for t in tools.values()]

    dynamic_max_tokens = max_tokens
    if rag_query:
        dynamic_max_tokens = estimate_response_reserve(
            rag_query,
            settings.reserve_for_response_min,
            settings.reserve_for_response_max,
        )

    for round_num in range(MAX_TOOL_ROUNDS):
        record_agent_round(session_id)
        
        yield StreamEvent(
            type="round_start",
            data={"round": round_num + 1, "max_rounds": MAX_TOOL_ROUNDS},
            round_num=round_num
        )

        messages = store.build_messages(
            session_id,
            use_rag=rag_query is not None,
            query=rag_query,
            rag_top_k=rag_top_k,
            rag_initial_k=rag_initial_k,
            use_reranker=use_reranker,
        )

        current_temperature = settings.tool_call_temperature

        # Stream the completion with tools
        tool_calls_buffer = []
        content_buffer = ""
        tool_calls_complete = False
        
        async for chunk in completion_client.complete_with_tools_stream(
            messages, tool_schemas, dynamic_max_tokens, current_temperature
        ):
            if chunk["type"] == "content":
                content_buffer += chunk["content"]
                yield StreamEvent(
                    type="content_delta",
                    data={"content": chunk["content"]},
                    round_num=round_num
                )
            elif chunk["type"] == "tool_calls":
                tool_calls_buffer = chunk["tool_calls"]
                yield StreamEvent(
                    type="tool_calls_delta",
                    data={"tool_calls": tool_calls_buffer},
                    round_num=round_num
                )
            elif chunk["type"] == "finish":
                finish_reason = chunk["finish_reason"]
                content_buffer = chunk.get("content", content_buffer)
                tool_calls_buffer = chunk.get("tool_calls", tool_calls_buffer)
                
                if finish_reason == "tool_calls" and tool_calls_buffer:
                    tool_calls_complete = True
                    break
                elif finish_reason == "stop":
                    # Final response without tool calls
                    if content_buffer:
                        store.add_turn(session_id, "assistant", content_buffer)
                        record_agent_turn(session_id, "assistant")
                        yield StreamEvent(
                            type="done",
                            data={"reply": content_buffer},
                            round_num=round_num
                        )
                        return
        
        if not tool_calls_complete or not tool_calls_buffer:
            # No tool calls, we're done
            if content_buffer:
                store.add_turn(session_id, "assistant", content_buffer)
                record_agent_turn(session_id, "assistant")
                yield StreamEvent(
                    type="done",
                    data={"reply": content_buffer},
                    round_num=round_num
                )
                return
            continue

        # Filter out incomplete tool calls
        tool_calls = [tc for tc in tool_calls_buffer if tc.get("id") and tc.get("function", {}).get("name")]
        
        if not tool_calls:
            continue

        # Store assistant message with tool calls
        store.add_turn(session_id, "assistant", content_buffer or "", tool_calls=tool_calls)
        record_agent_turn(session_id, "assistant")

        yield StreamEvent(
            type="tool_calls",
            data={"tool_calls": tool_calls},
            round_num=round_num
        )

        # Execute tools in parallel with progress events
        results = []
        for call in tool_calls:
            yield StreamEvent(
                type="tool_start",
                data={"tool_call_id": call["id"], "tool_name": call["function"]["name"], "arguments": call["function"]["arguments"]},
                round_num=round_num
            )
        
        results = await _execute_tools_parallel_streaming(
            tools, tool_calls, tool_timeout, max_retries
        )

        # Store tool results and yield events
        for call, result in zip(tool_calls, results):
            store.add_turn(session_id, "tool", result, tool_call_id=call["id"])
            record_agent_turn(session_id, "tool")
            
            yield StreamEvent(
                type="tool_result",
                data={"tool_call_id": call["id"], "tool_name": call["function"]["name"], "result": result},
                round_num=round_num
            )

        yield StreamEvent(
            type="round_end",
            data={"round": round_num + 1, "tool_calls_made": len(tool_calls)},
            round_num=round_num
        )

    yield StreamEvent(
        type="done",
        data={"reply": "Stopped after too many tool calls in a row — try breaking the task into smaller steps."},
        round_num=MAX_TOOL_ROUNDS - 1
    )


async def _execute_tools_parallel_streaming(
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
                    _execute_tool_async(tools, call),
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


async def _execute_tool_async(tools: dict, call: dict) -> str:
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
        return await tool["fn"](**args)
    except Exception as e:
        return f"Error running {name}: {type(e).__name__}: {e}"


def _validate_args(schema: Optional[dict], args: dict, tool_name: str) -> None:
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