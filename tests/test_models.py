"""Tests for domain models (src/context/models.py)."""

from src.context.models import SearchResult


class TestSearchResult:
    def test_creation(self) -> None:
        r = SearchResult(
            score=0.85,
            text="chunk text",
            source="doc.pdf",
            page=1,
            collection="test",
            chunk_index=0,
        )
        assert r.score == 0.85
        assert r.text == "chunk text"
        assert r.source == "doc.pdf"
        assert r.page == 1
        assert r.collection == "test"
        assert r.chunk_index == 0

    def test_frozen(self) -> None:
        r = SearchResult(score=0.9, text="t", source="s", page=1, collection="c", chunk_index=0)
        try:
            r.score = 1.0
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_hashable(self) -> None:
        r1 = SearchResult(score=0.9, text="t", source="s", page=1, collection="c", chunk_index=0)
        r2 = SearchResult(score=0.8, text="t", source="s", page=1, collection="c", chunk_index=0)
        # frozen=True BaseModel is hashable
        s: set[int] = {hash(r1), hash(r2)}
        assert len(s) == 2
