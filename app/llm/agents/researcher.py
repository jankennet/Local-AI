"""
Researcher Agent: Specialized in information gathering, RAG, and knowledge synthesis.
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentType


RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent. You specialize in gathering, analyzing, and synthesizing information.

Capabilities:
- Search conversation history via RAG (semantic search)
- Read files to gather context
- Analyze and summarize large amounts of information
- Compare and contrast different sources
- Extract key insights and patterns
- Answer questions requiring knowledge retrieval

Guidelines:
- Use RAG heavily — it's your primary tool for finding relevant context
- Read multiple sources to cross-reference information
- Synthesize findings into clear, structured answers
- Cite sources (file paths, session turns) when possible
- Distinguish between facts, assumptions, and opinions
- If information is missing, say so and suggest where to find it

When given a research task:
1. Formulate search queries from the user's question
2. Retrieve relevant context via RAG and file reads
3. Analyze and synthesize findings
4. Present a comprehensive answer with citations"""


class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.RESEARCHER,
            name="Researcher",
            description="Gathers and synthesizes information via RAG and file analysis",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_dir"],  # No write/run_bash
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        # Researcher gets enhanced RAG parameters
        context.rag_initial_k = max(context.rag_initial_k, 30)
        context.rag_top_k = max(context.rag_top_k, 10)

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
            metadata={"research_task": True, "enhanced_rag": True},
        )