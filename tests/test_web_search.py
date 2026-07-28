"""Tests for web search models (src/retrieval/web_search.py)."""

from src.retrieval.web_search import WebSearchResult, WebSearchStatus


class TestWebSearchStatus:
    def test_values(self) -> None:
        assert WebSearchStatus.OK.value == "ok"
        assert WebSearchStatus.QUOTA_EXCEEDED.value == "quota_exceeded"
        assert WebSearchStatus.ERROR.value == "error"

    def test_is_str(self) -> None:
        assert isinstance(WebSearchStatus.OK, str)


class TestWebSearchResult:
    def test_creation(self) -> None:
        r = WebSearchResult(title="Test", url="https://example.com", content="snippet")
        assert r.title == "Test"
        assert r.url == "https://example.com"
        assert r.content == "snippet"
