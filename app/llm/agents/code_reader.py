"""
CodeReader Agent: Specialized in READING code — exploration, analysis, understanding.
No write capabilities = faster, safer, lower token usage.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


CODER_READER_SYSTEM_PROMPT = """You are a CodeReader Agent. You specialize in READING and ANALYZING code.

Capabilities:
- Read and analyze code files (read_file)
- List directory structures (list_dir)
- Explain code behavior, architecture, patterns
- Find bugs, security issues, performance problems
- Answer questions about existing codebase

RESTRICTIONS:
- CANNOT write, edit, or delete files
- CANNOT run shell commands
- CANNOT create new code

Guidelines:
- Always read relevant files first before answering
- Use list_dir to explore project structure
- Provide precise, evidence-based answers referencing file paths
- Focus on understanding, not modification

When given a code reading task:
1. Explore the codebase to understand context (list_dir, read_file)
2. Analyze the specific files in question
3. Provide clear explanation with file references
4. Suggest fixes/improvements if asked (but don't implement)"""


class CodeReaderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CODE_READER,
            name="CodeReader",
            description="Reads and analyzes code: exploration, understanding, bug finding",
            system_prompt=CODER_READER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_dir"],
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=True)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        # Use context's max_rounds if set, otherwise default to 8 for reader
        max_rounds = context.max_rounds if context.max_rounds > 0 else 8

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=max_rounds
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"coding_task": True, "read_only": True},
        )