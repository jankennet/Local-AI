"""
Multi-Agent Orchestrator: Routes tasks to specialized agents and coordinates execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .base import AgentContext, AgentResult, AgentType
from .classifier import ClassificationResult, get_classifier
from .planner import PlannerAgent
from .code_reader import CodeReaderAgent
from .code_writer import CodeWriterAgent
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent
from .general import GeneralAgent
from ..completion_client import CompletionClient
from ...sessions.store import SessionStore
from ...config import settings

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    final_reply: str
    primary_agent: AgentType
    agents_used: list[AgentType] = field(default_factory=list)
    planner_result: Optional[AgentResult] = None
    execution_results: list[AgentResult] = field(default_factory=list)
    review_result: Optional[AgentResult] = None
    total_tool_calls: int = 0
    total_rounds: int = 0
    metadata: dict = field(default_factory=dict)


class AgentOrchestrator:
    """Orchestrates multi-agent workflows for complex tasks."""

    def __init__(self):
        self._agents = {
            AgentType.PLANNER: PlannerAgent(),
            AgentType.CODE_READER: CodeReaderAgent(),
            AgentType.CODE_WRITER: CodeWriterAgent(),
            AgentType.RESEARCHER: ResearcherAgent(),
            AgentType.REVIEWER: ReviewerAgent(),
            AgentType.GENERAL: GeneralAgent(),
        }
        self._classifier = get_classifier()
        self._enable_planning = True
        self._enable_review = True

    def configure(self, enable_planning: bool = True, enable_review: bool = True) -> None:
        """Configure orchestrator behavior."""
        self._enable_planning = enable_planning
        self._enable_review = enable_review

    async def execute(
        self,
        context: AgentContext,
        force_agent: Optional[AgentType] = None,
    ) -> OrchestrationResult:
        """Execute a task using the appropriate agent(s)."""
        agents_used = []
        execution_results = []
        planner_result = None
        review_result = None

        # Classify the query
        session_context = self._get_session_context(context.store, context.session_id)
        classification = self._classifier.classify(context.query, session_context)

        primary_agent_type = force_agent or classification.agent_type
        logger.info(
            f"Orchestrator: classified as {primary_agent_type.value} "
            f"(confidence: {classification.confidence:.2f}, "
            f"requires_planning: {classification.requires_planning})"
        )

        # Step 1: Planning (for complex tasks)
        if self._enable_planning and classification.requires_planning and primary_agent_type != AgentType.PLANNER:
            logger.info("Orchestrator: invoking Planner for task decomposition")
            # Allocate 15% of budget to planning
            plan_budget = int(context.token_budget * 0.15) if context.token_budget > 0 else 1024
            planner_context = AgentContext(
                session_id=context.session_id,
                query=f"Create a detailed execution plan for: {context.query}",
                store=context.store,
                completion_client=context.completion_client,
                tools=context.tools,
                max_tokens=context.max_tokens,
                temperature=0.3,
                rag_top_k=context.rag_top_k,
                rag_initial_k=context.rag_initial_k,
                use_reranker=context.use_reranker,
                tool_timeout=context.tool_timeout,
                max_retries=context.max_retries,
                token_budget=plan_budget,
                max_tool_calls=5,
                max_rounds=4,
            )
            planner_result = await self._agents[AgentType.PLANNER].execute(planner_context)
            agents_used.append(AgentType.PLANNER)
            execution_results.append(planner_result)

            # Augment the original query with the plan
            context.query = f"{context.query}\n\n## Execution Plan\n{planner_result.reply}"
            context.metadata["plan"] = planner_result.reply
            # Deduct planning tokens from main budget
            context.token_budget = max(0, context.token_budget - planner_context.tokens_used)

        # Step 2: Execute with primary agent
        logger.info(f"Orchestrator: executing with {primary_agent_type.value} agent")
        primary_agent = self._agents[primary_agent_type]
        execution_result = await primary_agent.execute(context)
        agents_used.append(primary_agent_type)
        execution_results.append(execution_result)

        # Step 3: Review (if enabled and not a simple task)
        if (
            self._enable_review
            and primary_agent_type in (AgentType.CODE_READER, AgentType.CODE_WRITER, AgentType.RESEARCHER, AgentType.PLANNER)
            and not force_agent  # Don't review if user explicitly chose an agent
        ):
            logger.info("Orchestrator: invoking Reviewer for quality check")
            # Allocate 10% of remaining budget to review
            review_budget = int(context.token_budget * 0.10) if context.token_budget > 0 else 512
            review_context = AgentContext(
                session_id=context.session_id,
                query=(
                    f"Review the following output from the {primary_agent_type.value} agent:\n\n"
                    f"Original task: {context.query.split(chr(10))[0] if context.query else 'N/A'}\n\n"
                    f"Agent output:\n{execution_result.reply}"
                ),
                store=context.store,
                completion_client=context.completion_client,
                tools=context.tools,
                max_tokens=context.max_tokens,
                temperature=0.2,
                rag_top_k=context.rag_top_k,
                rag_initial_k=context.rag_initial_k,
                use_reranker=context.use_reranker,
                tool_timeout=context.tool_timeout,
                max_retries=context.max_retries,
                token_budget=review_budget,
                max_tool_calls=3,
                max_rounds=3,
            )
            review_result = await self._agents[AgentType.REVIEWER].execute(review_context)
            agents_used.append(AgentType.REVIEWER)
            execution_results.append(review_result)

            # If review finds issues, optionally re-execute (simplified: just note in metadata)
            if "NEEDS_REVISION" in review_result.reply or "FAIL" in review_result.reply:
                context.metadata["review_issues"] = True
                logger.warning("Orchestrator: Reviewer found issues in output")
            # Deduct review tokens from main budget
            context.token_budget = max(0, context.token_budget - review_context.tokens_used)

        # Compile final result
        final_reply = self._compile_final_reply(
            execution_result, planner_result, review_result, primary_agent_type
        )

        total_tool_calls = sum(r.tool_calls_made for r in execution_results)
        total_rounds = sum(r.rounds_used for r in execution_results)

        return OrchestrationResult(
            final_reply=final_reply,
            primary_agent=primary_agent_type,
            agents_used=agents_used,
            planner_result=planner_result,
            execution_results=execution_results,
            review_result=review_result,
            total_tool_calls=total_tool_calls,
            total_rounds=total_rounds,
            metadata={
                "classification": {
                    "agent_type": classification.agent_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "requires_planning": classification.requires_planning,
                },
                **context.metadata,
            },
        )

    def _get_session_context(self, store: SessionStore, session_id: str, max_turns: int = 5) -> str:
        """Get recent session history for context-aware classification."""
        try:
            session = store.get(session_id)
            if not session:
                return ""
            recent = session.history[-max_turns:]
            return "\n".join(
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')[:200]}"
                for turn in recent
            )
        except Exception:
            return ""

    def _compile_final_reply(
        self,
        execution_result: AgentResult,
        planner_result: Optional[AgentResult],
        review_result: Optional[AgentResult],
        primary_agent: AgentType,
    ) -> str:
        """Compile the final reply to return to the user."""
        parts = []

        if planner_result:
            parts.append(f"## Plan\n{planner_result.reply}")

        parts.append(f"## {primary_agent.value.capitalize()} Agent Output\n{execution_result.reply}")

        if review_result:
            parts.append(f"## Review\n{review_result.reply}")

        return "\n\n".join(parts)

    def get_agent(self, agent_type: AgentType):
        """Get a specific agent instance."""
        return self._agents.get(agent_type)

    def list_agents(self) -> dict:
        """List all available agents with their descriptions."""
        return {
            agent_type.value: {
                "name": agent.name,
                "description": agent.description,
                "allowed_tools": agent.allowed_tools,
            }
            for agent_type, agent in self._agents.items()
        }


# Singleton instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator