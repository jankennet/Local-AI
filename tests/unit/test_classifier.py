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

    def test_fact_question_who_goes_to_researcher(self):
        result = _classify("Could you tell me who's the CEO of Itel?")
        assert result.agent_type == AgentType.RESEARCHER

    def test_role_of_question_goes_to_researcher(self):
        result = _classify("Who is the founder of Tesla?")
        assert result.agent_type == AgentType.RESEARCHER

    def test_current_office_goes_to_researcher(self):
        result = _classify("Who is the current president of France?")
        assert result.agent_type == AgentType.RESEARCHER

    def test_profile_link_request_goes_to_researcher(self):
        result = _classify("Give me a wikipedia or linkedin link for the CEO of Itel")
        assert result.agent_type == AgentType.RESEARCHER

    def test_fact_question_does_not_trigger_planner(self):
        result = _classify("Can you give me a wikipedia or linkedin link for Johnathan Smith, the CEO of ITEL?")
        assert result.agent_type == AgentType.RESEARCHER
        assert result.requires_planning is False

    def test_plain_how_to_question_still_general(self):
        result = _classify("What does a capacitor do in a power supply?")
        assert result.agent_type == AgentType.GENERAL

    def test_inline_code_explain_goes_to_general(self):
        result = _classify(
            'count = 1 while count <= 5: print(f"The current count is: {count}") '
            'count += 1 print("Loop finished!") explain this code please'
        )
        assert result.agent_type == AgentType.GENERAL
        assert result.requires_planning is False

    def test_js_inline_code_explain_goes_to_general(self):
        result = _classify(
            "Explain this code : "
            "const { CurlImpersonate } = require('curl-impersonate'); "
            "const client = new CurlImpersonate(undefined, { method: 'GET', impersonate: 'chrome-116' }); "
            "const response = await client.makeRequest('https://target-website.com'/);"
        )
        assert result.agent_type == AgentType.GENERAL
        assert result.requires_planning is False

    def test_fenced_inline_code_explain_goes_to_general(self):
        result = _classify(
            "```python\nprint('hello')\n```\nexplain what this does"
        )
        assert result.agent_type == AgentType.GENERAL

    def test_explain_reader_query_does_not_plan(self):
        result = _classify("explain how this class works")
        assert result.agent_type == AgentType.CODE_READER
        assert result.requires_planning is False

    def test_prose_mentioning_class_not_inline_code(self):
        result = _classify(
            "explain how the class and inheritance system works in TypeScript"
        )
        assert result.agent_type != AgentType.GENERAL

    def test_complex_reader_task_still_plans(self):
        result = _classify(
            "Read the code in the test files, analyze the authentication class, "
            "and describe the module structure"
        )
        assert result.agent_type == AgentType.CODE_READER
        assert result.requires_planning is True

    def test_complex_writer_task_still_plans(self):
        result = _classify(
            "Write a REST API with user authentication, database models, "
            "and unit tests covering login, registration, and password reset"
        )
        assert result.requires_planning is True