"""Tests for Settings and config helpers (src/config/settings.py)."""

import pytest

from src.config.settings import Settings, _split_backend_model
from src.domain.models import LLMRole


class TestSplitBackendModel:
    def test_valid_format(self) -> None:
        assert _split_backend_model("ollama,bge-m3", "EMBEDDER") == ("ollama", "bge-m3")

    def test_strips_whitespace(self) -> None:
        assert _split_backend_model("  ollama , bge-m3  ", "EMBEDDER") == ("ollama", "bge-m3")

    def test_model_with_colon(self) -> None:
        assert _split_backend_model("ollama,qwen3.5:4b", "EMBEDDER") == ("ollama", "qwen3.5:4b")

    def test_missing_comma(self) -> None:
        with pytest.raises(ValueError, match="backend,model"):
            _split_backend_model("ollama", "EMBEDDER")

    def test_empty_backend(self) -> None:
        with pytest.raises(ValueError):
            _split_backend_model(",bge-m3", "EMBEDDER")

    def test_empty_model(self) -> None:
        with pytest.raises(ValueError):
            _split_backend_model("ollama,", "EMBEDDER")


class TestSettingsDefaults:
    def test_chunk_size_positive(self) -> None:
        s = Settings()
        assert s.chunk_size > 0

    def test_chunk_overlap_less_than_size(self) -> None:
        s = Settings()
        assert s.chunk_overlap < s.chunk_size

    def test_context_guard_safety_margin_in_range(self) -> None:
        s = Settings()
        assert 0.0 <= s.context_guard_safety_margin <= 0.5

    def test_context_guard_output_reserve_positive(self) -> None:
        s = Settings()
        assert s.context_guard_output_reserve > 0

    def test_embedding_batch_size_positive(self) -> None:
        s = Settings()
        assert s.embedding_batch_size > 0

    def test_web_search_defaults(self) -> None:
        s = Settings()
        assert s.web_search_depth in ("basic", "advanced")
        assert isinstance(s.web_search_include_answer, bool)


class TestRoleFallbackSpec:
    def test_empty_means_no_fallback(self) -> None:
        s = Settings(llm_rol_generate_fallback="", llm_rol_supplement_fallback="")
        assert s.role_fallback_spec(LLMRole.GENERATE) is None
        assert s.role_fallback_spec(LLMRole.WEB_SUPPLEMENT) is None

    def test_parses_generate_fallback(self) -> None:
        s = Settings(
            llm_rol_generate="gemini,gemini-2.5-flash-lite",
            llm_rol_generate_fallback="ollama,qwen3.5:4b",
        )
        assert s.role_fallback_spec(LLMRole.GENERATE) == ("ollama", "qwen3.5:4b")

    def test_parses_supplement_fallback(self) -> None:
        s = Settings(
            llm_rol_supplement="gemini,gemini-2.5-flash-lite",
            llm_rol_supplement_fallback="ollama,qwen3.5:4b",
        )
        assert s.role_fallback_spec(LLMRole.WEB_SUPPLEMENT) == ("ollama", "qwen3.5:4b")

    def test_invalid_fallback_format_raises(self) -> None:
        with pytest.raises(ValueError, match="LLM_ROL_GENERATE_FALLBACK"):
            Settings(llm_rol_generate_fallback="not-a-backend-model")
