"""Routing tests for the query classifier's explain-vs-plan decision."""

from app.llm.agents.classifier import get_classifier
from app.llm.agents.base import AgentType


def _classify(message: str):
    return get_classifier().classify(message)


class TestRouting:
    def test_casual_how_to_goes_to_general(self):
        assert _classify("Help me with converting ac to dc").agent_type == AgentType.GENERAL

    def test_deep_dive_request_stays_general(self):
        result = _classify(
            "Give me a full detailed explanation of how rectification works, "
            "smoothing, voltage regulation, and all the steps with diagrams"
        )
        assert result.agent_type == AgentType.GENERAL

    def test_real_planning_still_goes_to_planner(self):
        result = _classify("Design the architecture for our new project")
        assert result.agent_type == AgentType.PLANNER

    def test_explain_with_project_intent_stays_planner(self):
        result = _classify("Explain how we should build and break down the new feature")
        assert result.agent_type == AgentType.PLANNER

    def test_research_comparison_unchanged(self):
        result = _classify("Compare the differences between SQL and NoSQL databases")
        assert result.agent_type == AgentType.RESEARCHER