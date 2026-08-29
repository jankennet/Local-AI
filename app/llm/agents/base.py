"""
Base agent interface and shared types for multi-agent orchestration.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..completion_client import CompletionClient
from ..tools import TOOLS
from ...metrics import record_agent_turn, record_tool_call
from ...sessions.store import SessionStore
from ...config import settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    PLANNER = "planner"
    CODER = "coder"
    CODE_READER = "code_reader"
    CODE_WRITER = "code_writer"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    GENERAL = "general"


@dataclass
class AgentContext:
    session_id: str
    query: str
    store: SessionStore
    completion_client: CompletionClient
    tools: dict
    max_tokens: int = 512
    temperature: float = 0.7
    rag_top_k: int = 5
    rag_initial_k: int = 20
    use_reranker: bool = True
    tool_timeout: float = 30.0
    max_retries: int = 2
    metadata: dict = field(default_factory=dict)
    # Budget-aware fields
    token_budget: int = 4096          # Total token budget for this agent execution
    tokens_used: int = 0              # Tokens consumed so far
    max_tool_calls: int = 20          # Hard limit on tool calls
    max_rounds: int = 12              # Hard limit on rounds


@dataclass
class AgentResult:
    reply: str
    agent_type: AgentType
    tool_calls_made: int = 0
    rounds_used: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""

    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: str,
        allowed_tools: Optional[list[str]] = None,
    ):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or list(TOOLS.keys())

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task. Must be implemented by subclasses."""
        pass

    def _get_filtered_tools(self, context: AgentContext) -> dict:
        """Return only the tools this agent is allowed to use."""
        return {k: v for k, v in context.tools.items() if k in self.allowed_tools}

    def _build_messages_with_system(
        self, context: AgentContext, use_rag: bool = True
    ) -> list:
        """Build messages including this agent's system prompt."""
        messages = context.store.build_messages(
            context.session_id,
            use_rag=use_rag,
            query=context.query,
            rag_top_k=context.rag_top_k,
            rag_initial_k=context.rag_initial_k,
            use_reranker=context.use_reranker,
        )

        # Prepend agent-specific system prompt
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{self.system_prompt}\n\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    async def _run_tool_loop(
        self,
        context: AgentContext,
        messages: list,
        tool_schemas: list,
        max_rounds: int = 8,
    ) -> tuple[str, int, int]:
        """Run the tool-calling loop with budget awareness. Returns (final_reply, rounds_used, tool_calls_made)."""
        rounds_used = 0
        tool_calls_made = 0

        # Use context's budget limits if set, otherwise fall back to parameter
        effective_max_rounds = context.max_rounds if context.max_rounds > 0 else max_rounds
        effective_max_tool_calls = context.max_tool_calls if context.max_tool_calls > 0 else 20

        for round_num in range(effective_max_rounds):
            rounds_used += 1
            record_agent_turn(context.session_id, f"{self.agent_type.value}_round")

            # Check budget before making another round
            if context.tokens_used >= context.token_budget:
                logger.warning(f"Agent {self.agent_type.value}: token budget exceeded ({context.tokens_used}/{context.token_budget})")
                reply = "Token budget exhausted. Stopping early."
                context.store.add_turn(context.session_id, "assistant", reply)
                record_agent_turn(context.session_id, "assistant")
                return reply, rounds_used, tool_calls_made

            if tool_calls_made >= effective_max_tool_calls:
                logger.warning(f"Agent {self.agent_type.value}: tool call limit reached ({tool_calls_made}/{effective_max_tool_calls})")
                reply = "Tool call limit reached. Stopping early."
                context.store.add_turn(context.session_id, "assistant", reply)
                record_agent_turn(context.session_id, "assistant")
                return reply, rounds_used, tool_calls_made

            current_temperature = settings.tool_call_temperature

            message = context.completion_client.complete_with_tools(
                messages, tool_schemas, context.max_tokens, current_temperature
            )

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                reply = message.get("content") or ""
                context.store.add_turn(context.session_id, "assistant", reply)
                record_agent_turn(context.session_id, "assistant")
                # Estimate tokens for the reply
                context.tokens_used += context.store._counter.count(reply) if hasattr(context.store, '_counter') else len(reply) // 4
                return reply, rounds_used, tool_calls_made

            context.store.add_turn(
                context.session_id, "assistant", message.get("content") or "", tool_calls=tool_calls
            )
            record_agent_turn(context.session_id, "assistant")

            results = await self._execute_tools_parallel(
                context, tool_calls
            )
            tool_calls_made += len(tool_calls)

            for call, result in zip(tool_calls, results):
                context.store.add_turn(
                    context.session_id, "tool", result, tool_call_id=call["id"]
                )
                record_agent_turn(context.session_id, "tool")
                # Track token usage from tool results
                if hasattr(context.store, '_counter'):
                    context.tokens_used += context.store._counter.count(result)
                else:
                    context.tokens_used += len(result) // 4

            messages = context.store.build_messages(
                context.session_id,
                use_rag=False,
            )

            # Track token usage for the message history
            if hasattr(context.store, '_counter'):
                for msg in messages:
                    if msg.get("content"):
                        context.tokens_used += context.store._counter.count(msg["content"])

        return (
            "Stopped after too many tool calls — try breaking the task into smaller steps.",
            rounds_used,
            tool_calls_made,
        )

    async def _execute_tools_parallel(
        self, context: AgentContext, tool_calls: list
    ) -> list[str]:
        """Execute multiple tool calls in parallel with retry logic."""
        import time

        async def execute_with_retry(call: dict) -> str:
            name = call["function"]["name"]
            start = time.time()
            retries = 0
            for attempt in range(context.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self._execute_tool, context.tools, call),
                        timeout=context.tool_timeout,
                    )
                    record_tool_call(name, time.time() - start, True, retries)
                    return result
                except asyncio.TimeoutError:
                    retries += 1
                    logger.warning(
                        f"Tool {name} timed out (attempt {attempt + 1}/{context.max_retries + 1})"
                    )
                    if attempt == context.max_retries:
                        record_tool_call(name, time.time() - start, False, retries)
                        return f"Error: tool '{name}' timed out after {context.tool_timeout}s"
                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Tool {name} failed (attempt {attempt + 1}/{context.max_retries + 1}): {e}"
                    )
                    if attempt == context.max_retries:
                        record_tool_call(name, time.time() - start, False, retries)
                        return f"Error running {name}: {type(e).__name__}: {e}"
            record_tool_call(name, time.time() - start, False, retries)
            return f"Error: tool '{name}' failed after {context.max_retries + 1} attempts"

        tasks = [execute_with_retry(call) for call in tool_calls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _execute_tool(self, tools: dict, call: dict) -> str:
        import json

        name = call["function"]["name"]
        tool = tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"

        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return "Error: malformed tool arguments"

        self._validate_args(tool.get("schema"), args, name)

        try:
            return tool["fn"](**args)
        except Exception as e:
            return f"Error running {name}: {type(e).__name__}: {e}"

    def _validate_args(self, schema: Optional[dict], args: dict, tool_name: str) -> None:
        if not schema:
            return

        params = schema.get("function", {}).get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        for req in required:
            if req not in args:
                raise ValueError(
                    f"Missing required argument '{req}' for tool '{tool_name}'"
                )

        for key, value in args.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._type_matches(value, expected_type):
                    raise ValueError(
                        f"Argument '{key}' for tool '{tool_name}' expected type {expected_type}, got {type(value).__name__}"
                    )

    def _type_matches(self, value: Any, expected_type: str) -> bool:
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