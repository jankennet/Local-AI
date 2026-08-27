"""
Planner Agent: Decomposes complex tasks into ordered steps.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


PLANNER_SYSTEM_PROMPT = """You are a Planning Agent. Your role is to analyze complex user requests and break them down into clear, ordered, actionable steps.

When given a task:
1. Analyze the request thoroughly
2. Identify the key objectives and constraints
3. Decompose into a logical sequence of steps
4. Each step should be specific enough for another agent to execute
5. Consider dependencies between steps

Output format:
## Plan
1. **Step 1**: [Description] - [Agent type: coder/researcher/general]
2. **Step 2**: [Description] - [Agent type: coder/researcher/general]
...

## Execution Notes
- Dependencies: Step 2 requires Step 1 output
- Estimated complexity: Low/Medium/High
- Recommended approach: [brief guidance]

Do NOT execute the steps yourself — just create the plan. Be specific and actionable."""


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.PLANNER,
            name="Planner",
            description="Decomposes complex tasks into ordered, executable steps",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_dir"],  # Can explore but not modify
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=False)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=5
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"plan_created": True},
        )