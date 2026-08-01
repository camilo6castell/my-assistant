"""Tests for LLMRole enum (src/nlp/llm/roles.py)."""

from src.nlp.llm.roles import LLMRole


class TestLLMRole:
    def test_values(self) -> None:
        assert LLMRole.GENERATE.value == "generate"
        assert LLMRole.WEB_SUPPLEMENT.value == "web_supplement"

    def test_all_roles_present(self) -> None:
        assert len(LLMRole) == 2

    def test_is_str(self) -> None:
        assert isinstance(LLMRole.GENERATE, str)

    def test_from_value(self) -> None:
        assert LLMRole("generate") is LLMRole.GENERATE
        assert LLMRole("web_supplement") is LLMRole.WEB_SUPPLEMENT
