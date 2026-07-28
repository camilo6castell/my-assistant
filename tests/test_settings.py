"""Tests for Settings and config helpers (src/config/settings.py)."""

import pytest

from src.config.settings import Settings, _split_backend_model


class TestSplitBackendModel:
    def test_valid_format(self) -> None:
        assert _split_backend_model("ollama,bge-m3", "EMBEDDER") == ("ollama", "bge-m3")

    def test_strips_whitespace(self) -> None:
        assert _split_backend_model("  ollama , bge-m3  ", "EMBEDDER") == ("ollama", "bge-m3")

    def test_model_with_colon(self) -> None:
        assert _split_backend_model("flm,qwen3.5:9b", "EMBEDDER") == ("flm", "qwen3.5:9b")

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

    def test_confidence_limit_in_range(self) -> None:
        s = Settings()
        assert 0.0 <= s.confidence_limit <= 1.0

    def test_max_review_attempts_nonneg(self) -> None:
        s = Settings()
        assert s.max_review_attempts >= 0

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
