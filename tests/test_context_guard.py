"""Tests for context guard (src/nlp/llm/context_guard.py)."""

from src.nlp.llm.context_guard import (
    DEFAULT_OUTPUT_RESERVE,
    SAFETY_MARGIN_RATIO,
    ContextLimitExceeded,
)


class TestContextLimitExceeded:
    def test_creation(self) -> None:
        e = ContextLimitExceeded(estimated_tokens=5000, limit=4096, model="test-model")
        assert e.estimated_tokens == 5000
        assert e.limit == 4096
        assert e.model == "test-model"

    def test_message(self) -> None:
        e = ContextLimitExceeded(estimated_tokens=5000, limit=4096, model="m")
        assert "5000" in str(e)
        assert "4096" in str(e)
        assert "m" in str(e)

    def test_as_detail(self) -> None:
        e = ContextLimitExceeded(estimated_tokens=5000, limit=4096, model="m")
        d = e.as_detail()
        assert d["error"] == "context_limit_exceeded"
        assert d["estimated_tokens"] == 5000
        assert d["limit"] == 4096
        assert d["model"] == "m"


class TestConstants:
    def test_safety_margin_ratio_from_settings(self) -> None:
        from src.config.settings import settings

        assert settings.context_guard_safety_margin == SAFETY_MARGIN_RATIO

    def test_output_reserve_from_settings(self) -> None:
        from src.config.settings import settings

        assert settings.context_guard_output_reserve == DEFAULT_OUTPUT_RESERVE
