"""
Multi-Agent Orchestration Package

Exports:
- Base types: AgentType, AgentContext, AgentResult, BaseAgent
- Specialized agents: PlannerAgent, CodeReaderAgent, CodeWriterAgent, ResearcherAgent, ReviewerAgent, GeneralAgent
- Classifier: QueryClassifier, ClassificationResult
- Orchestrator: AgentOrchestrator, OrchestrationResult
- Singleton accessors: get_orchestrator, get_classifier
"""

from .base import (
    AgentType,
    AgentContext,
    AgentResult,
    BaseAgent,
)
from .planner import PlannerAgent
from .code_reader import CodeReaderAgent
from .code_writer import CodeWriterAgent
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent
from .general import GeneralAgent
from .classifier import QueryClassifier, ClassificationResult, get_classifier
from .orchestrator import AgentOrchestrator, OrchestrationResult, get_orchestrator

__all__ = [
    "AgentType",
    "AgentContext",
    "AgentResult",
    "BaseAgent",
    "PlannerAgent",
    "CodeReaderAgent",
    "CodeWriterAgent",
    "ResearcherAgent",
    "ReviewerAgent",
    "GeneralAgent",
    "QueryClassifier",
    "ClassificationResult",
    "get_classifier",
    "AgentOrchestrator",
    "OrchestrationResult",
    "get_orchestrator",
]