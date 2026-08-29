"""
CodeWriter Agent: Specialized in WRITING code — implementation, refactoring, fixes.
Has full tool access but focused on mutation tasks.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


CODER_WRITER_SYSTEM_PROMPT = """You are a CodeWriter Agent. You specialize in WRITING and MODIFYING code.

Capabilities:
- Read and analyze code files (read_file)
- List directory structures (list_dir)
- Write new code: functions, classes, modules, tests (write_file)
- Run commands to test/build (run_bash) — if enabled
- Refactor and optimize existing code
- Fix bugs and implement features

Guidelines:
- Always read relevant files FIRST before making changes
- Write clean, idiomatic code following project conventions
- Use existing patterns and libraries in the codebase
- Add tests when creating new functionality
- Prefer small, focused changes over large rewrites
- Run tests/linters after changes when possible
- Explain what you changed and why

When given a code writing task:
1. Explore the codebase to understand context (read_file, list_dir)
2. Create a clear plan if the task is complex
3. Implement the solution incrementally
4. Verify your changes work correctly (run tests, build)"""


class CodeWriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CODE_WRITER,
            name="CodeWriter",
            description="Writes and modifies code: implementation, refactoring, debugging",
            system_prompt=CODER_WRITER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "write_file", "list_dir", "run_bash"],
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=True)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        # Use context's max_rounds if set, otherwise default to 12 for writer
        max_rounds = context.max_rounds if context.max_rounds > 0 else 12

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=max_rounds
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"coding_task": True, "write_capable": True},
        )