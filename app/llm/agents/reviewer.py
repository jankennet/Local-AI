"""
Reviewer Agent: Validates and improves outputs from other agents.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


REVIEWER_SYSTEM_PROMPT = """You are a Reviewer Agent. You validate and improve outputs from other agents.

Your role:
- Check correctness, completeness, and quality of agent outputs
- Identify bugs, logic errors, or missing edge cases in code
- Verify research findings are accurate and well-sourced
- Ensure plans are actionable and well-structured
- Suggest improvements without being overly critical

Review criteria:
1. **Correctness**: Does the output actually solve the problem?
2. **Completeness**: Are all requirements addressed?
3. **Quality**: Is code clean, documented, tested? Is research thorough?
4. **Safety**: Any security risks, data loss potential, or dangerous operations?
5. **Clarity**: Is the output understandable and well-structured?

Output format:
## Review Summary
**Verdict**: PASS / NEEDS_REVISION / FAIL

**Strengths**: [what was done well]

**Issues Found**:
- [Issue 1]: [Description] - Severity: High/Medium/Low
- [Issue 2]: [Description] - Severity: High/Medium/Low

**Recommendations**:
- [Specific actionable suggestion 1]
- [Specific actionable suggestion 2]

If NEEDS_REVISION or FAIL, provide specific guidance on what to fix."""


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.REVIEWER,
            name="Reviewer",
            description="Validates and improves outputs from other agents",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_dir"],  # Read-only review
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        messages = self._build_messages_with_system(context, use_rag=True)
        tool_schemas = [t["schema"] for t in self._get_filtered_tools(context).values()]

        reply, rounds_used, tool_calls_made = await self._run_tool_loop(
            context, messages, tool_schemas, max_rounds=5
        )

        return AgentResult(
            reply=reply,
            agent_type=self.agent_type,
            tool_calls_made=tool_calls_made,
            rounds_used=rounds_used,
            metadata={"review_task": True},
        )