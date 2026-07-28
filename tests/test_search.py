"""Tests for search helpers (src/retrieval/search.py)."""

from src.context.models import SearchResult
from src.retrieval.search import build_queries, format_context_chunks, rerank


class TestBuildQueries:
    def test_hard_mode_returns_single_query(self) -> None:
        result = build_queries("What is RAG?", "HARD")
        assert result == ["What is RAG?"]

    def test_soft_mode_returns_three_queries(self) -> None:
        result = build_queries("What is RAG?", "SOFT")
        assert len(result) == 3
        assert result[0] == "What is RAG?"
        assert "Explain the concept:" in result[1]
        assert "Relate ideas about:" in result[2]

    def test_hard_mode_with_special_chars(self) -> None:
        result = build_queries("¿Qué es?", "HARD")
        assert result == ["¿Qué es?"]


class TestFormatContextChunks:
    def test_formats_correctly(self) -> None:
        results = [
            SearchResult(
                score=0.9,
                text="hello",
                source="doc.pdf",
                page=1,
                collection="test",
                chunk_index=0,
            )
        ]
        chunks = format_context_chunks(results)
        assert len(chunks) == 1
        assert "SOURCE: doc.pdf" in chunks[0]
        assert "COLLECTION: test" in chunks[0]
        assert "PAGE: 1" in chunks[0]
        assert "hello" in chunks[0]

    def test_empty_results(self) -> None:
        assert format_context_chunks([]) == []

    def test_multiple_results(self) -> None:
        results = [
            SearchResult(
                score=0.9,
                text="a",
                source="a.pdf",
                page=1,
                collection="c1",
                chunk_index=0,
            ),
            SearchResult(
                score=0.8,
                text="b",
                source="b.pdf",
                page=2,
                collection="c2",
                chunk_index=1,
            ),
        ]
        chunks = format_context_chunks(results)
        assert len(chunks) == 2


class TestRerank:
    def test_sorts_by_score_descending(self) -> None:
        results = [
            SearchResult(
                score=0.5,
                text="low",
                source="a",
                page=1,
                collection="c",
                chunk_index=0,
            ),
            SearchResult(
                score=0.9,
                text="high",
                source="b",
                page=1,
                collection="c",
                chunk_index=1,
            ),
            SearchResult(
                score=0.7,
                text="mid",
                source="c",
                page=1,
                collection="c",
                chunk_index=2,
            ),
        ]
        ranked = rerank(results)
        assert [r.text for r in ranked] == ["high", "mid", "low"]

    def test_deduplicates_by_source_and_chunk(self) -> None:
        results = [
            SearchResult(
                score=0.9,
                text="dup",
                source="a",
                page=1,
                collection="c",
                chunk_index=0,
            ),
            SearchResult(
                score=0.8,
                text="dup",
                source="a",
                page=1,
                collection="c",
                chunk_index=0,
            ),
        ]
        ranked = rerank(results)
        assert len(ranked) == 1

    def test_preserves_different_chunks_same_source(self) -> None:
        results = [
            SearchResult(
                score=0.9,
                text="a",
                source="a",
                page=1,
                collection="c",
                chunk_index=0,
            ),
            SearchResult(
                score=0.8,
                text="b",
                source="a",
                page=1,
                collection="c",
                chunk_index=1,
            ),
        ]
        ranked = rerank(results)
        assert len(ranked) == 2

    def test_empty_input(self) -> None:
        assert rerank([]) == []
