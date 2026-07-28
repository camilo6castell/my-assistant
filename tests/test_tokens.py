"""Tests for token estimation (src/utils/tokens.py)."""

from src.utils.tokens import HEURISTIC_CHARS_PER_TOKEN, estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_positive_for_nonempty(self) -> None:
        assert estimate_tokens("hello") > 0

    def test_scales_with_length(self) -> None:
        short = estimate_tokens("hi")
        long = estimate_tokens("a" * 1000)
        assert long > short

    def test_heuristic_fallback_value(self) -> None:
        assert HEURISTIC_CHARS_PER_TOKEN == 3.5

    def test_estimate_reasonable_range(self) -> None:
        # A 100-char string should be roughly 20-50 tokens
        n = estimate_tokens("a" * 100)
        assert 10 < n < 100
