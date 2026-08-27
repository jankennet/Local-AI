"""
Query Classifier: Routes incoming queries to the appropriate specialized agent.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .base import AgentType

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    agent_type: AgentType
    confidence: float
    reasoning: str
    requires_planning: bool = False


class QueryClassifier:
    """Classifies queries to determine the best agent for the task."""

    # Patterns that strongly indicate specific agent types
    CODER_PATTERNS = [
        r"\b(write|create|build|implement|add|refactor|fix|debug|optimize)\b.*\b(code|function|class|script|api|endpoint|database|sql)\b",
        r"\b(bug|error|exception|traceback|fail|crash)\b",
        r"\b(test|testing|unit test|pytest)\b",
        r"\b(read|edit|modify|update|change)\b.*\b(file|code|script)\b",
        r"\b(git|commit|push|pull|merge|branch)\b",
        r"\b(docker|kubernetes|k8s|deploy|ci/cd|pipeline)\b",
        r"\b(run|execute|install|package|dependency)\b",
    ]

    RESEARCHER_PATTERNS = [
        r"\b(what|how|why|explain|describe|compare|analyze)\b",
        r"\b(search|find|look up|lookup|retrieve)\b",
        r"\b(history|context|background|summary|overview)\b",
        r"\b(documentation|docs|spec|specification)\b",
        r"\b(best practice|pattern|approach|strategy|recommend)\b",
        r"\b(learn|understand|study|research)\b",
    ]

    PLANNER_PATTERNS = [
        r"\b(plan|planning|roadmap|steps|break down|decompose)\b",
        r"\b(project|feature|system|architecture|design)\b",
        r"\b(multi-step|complex|large|comprehensive)\b",
        r"\b(start|begin|initiate|kick off)\b.*\b(project|feature|task)\b",
    ]

    REVIEWER_PATTERNS = [
        r"\b(review|audit|check|validate|verify|inspect)\b",
        r"\b(quality|correctness|security|performance)\b",
        r"\b(code review|pr review|pull request)\b",
    ]

    def __init__(self):
        self._compiled_patterns = {
            AgentType.CODER: [re.compile(p, re.IGNORECASE) for p in self.CODER_PATTERNS],
            AgentType.RESEARCHER: [re.compile(p, re.IGNORECASE) for p in self.RESEARCHER_PATTERNS],
            AgentType.PLANNER: [re.compile(p, re.IGNORECASE) for p in self.PLANNER_PATTERNS],
            AgentType.REVIEWER: [re.compile(p, re.IGNORECASE) for p in self.REVIEWER_PATTERNS],
        }

    def classify(self, query: str, session_context: Optional[str] = None) -> ClassificationResult:
        """Classify a query and return the best agent type."""
        scores = {agent_type: 0.0 for agent_type in AgentType}
        matches = {agent_type: [] for agent_type in AgentType}

        # Score based on pattern matches
        for agent_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[agent_type] += 1.0
                    matches[agent_type].append(pattern.pattern)

        # Boost for explicit agent mentions
        query_lower = query.lower()
        if any(w in query_lower for w in ["code", "program", "develop", "script", "function", "class", "api"]):
            scores[AgentType.CODER] += 0.5
        if any(w in query_lower for w in ["research", "investigate", "explore", "analyze"]):
            scores[AgentType.RESEARCHER] += 0.5
        if any(w in query_lower for w in ["plan", "design", "architect"]):
            scores[AgentType.PLANNER] += 0.5
        if any(w in query_lower for w in ["review", "audit", "check"]):
            scores[AgentType.REVIEWER] += 0.5

        # Session context boost (if previous turns suggest a domain)
        if session_context:
            context_lower = session_context.lower()
            if any(w in context_lower for w in ["code", "function", "class", "bug", "error", "test"]):
                scores[AgentType.CODER] += 0.3
            if any(w in context_lower for w in ["research", "analyze", "compare", "document"]):
                scores[AgentType.RESEARCHER] += 0.3

        # Find best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]

        # Default to GENERAL if no strong signals
        if best_score < 0.5:
            return ClassificationResult(
                agent_type=AgentType.GENERAL,
                confidence=0.5,
                reasoning="No strong domain signals detected; using general agent",
            )

        # Check if planning is needed (complex task)
        requires_planning = (
            best_agent in (AgentType.CODER, AgentType.RESEARCHER)
            and best_score > 1.5
            and any(len(q.split()) > 15 for q in [query])
        )

        confidence = min(0.9, 0.5 + best_score * 0.15)

        return ClassificationResult(
            agent_type=best_agent,
            confidence=confidence,
            reasoning=f"Matched patterns: {matches[best_agent]}" if matches[best_agent] else "Keyword-based classification",
            requires_planning=requires_planning,
        )


# Singleton instance
_classifier: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier