"""
General Agent: Fallback for tasks that don't fit specialized categories.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


GENERAL_SYSTEM_PROMPT = """You are a General Purpose Agent. You handle tasks that don't fit into specialized categories.

Capabilities:
- Answer questions and explain concepts
- Help with planning and decision-making
- Creative writing and content generation
- General problem-solving
- Light file operations (read/list)
- Coordinate with other agents when needed

Guidelines:
- Be helpful, accurate, and concise
- Use tools when they add value (reading files, listing directories)
- Don't over-engineer simple requests
- If a task clearly belongs to a specialist (coding, research, planning), note that in your response
- Ask clarifying questions when the request is ambiguous

When given a task:
1. Understand what's being asked
2. Use available tools to gather needed information
3. Provide a clear, direct response
4. Suggest specialist agents if the task would benefit from them"""


class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.GENERAL,
            name="General",
            description="Handles general-purpose tasks and questions",
            system_prompt=GENERAL_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_dir", "write_file"],
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=True)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=8
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"general_task": True},
        )