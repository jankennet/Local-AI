"""
Coder Agent: Specialized in code tasks — reading, writing, refactoring, debugging.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


CODER_SYSTEM_PROMPT = """You are a Coder Agent. You specialize in software development tasks.

Capabilities:
- Read and analyze code files
- Write new code (functions, classes, modules, tests)
- Refactor and optimize existing code
- Debug errors and fix bugs
- Run commands to test/build (if shell enabled)
- Explain code behavior

Guidelines:
- Always read relevant files first before making changes
- Write clean, idiomatic code following project conventions
- Use existing patterns and libraries in the codebase
- Add tests when creating new functionality
- Prefer small, focused changes over large rewrites
- Run tests/linters after changes when possible

When given a coding task:
1. Explore the codebase to understand context
2. Create a clear plan if the task is complex
3. Implement the solution incrementally
4. Verify your changes work correctly"""


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CODER,
            name="Coder",
            description="Handles code tasks: reading, writing, refactoring, debugging",
            system_prompt=CODER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "write_file", "list_dir", "run_bash"],
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=True)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=12
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"coding_task": True},
        )