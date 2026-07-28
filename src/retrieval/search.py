"""
Note on # pyright: ignore[reportCallIssue] in index.search():
  Pylance reads SWIG C++ stubs for faiss; mypy has Python wrapper stubs.
  The pyright: directive is ignored by mypy, without unused-ignore.
"""

from __future__ import annotations

import numpy as np

from src.cli.modes import ChatMode
from src.config.settings import settings
from src.context.manager import LoadedCollection
from src.context.models import SearchResult
from src.nlp.embedders.encoder import get_encoder

# ======================================================
# QUERY BUILDING
# ======================================================


def build_queries(question: str, mode: str) -> list[str]:
    """
    HARD: 1 literal query.
    SOFT: 3 queries -- literal + 2 semantic variants.

    History travels as API messages in generate.py,
    not as an additional query variant.
    """
    if mode == ChatMode.HARD:
        return [question]

    return [
        question,
        f"Explain the concept: {question}",
        f"Relate ideas about: {question}",
    ]


# ======================================================
# ENCODING
# ======================================================


def encode_queries(queries: list[str]) -> np.ndarray:
    return get_encoder().encode(queries)


# ======================================================
# RETRIEVAL
# ======================================================


def retrieve(
    query_embeddings: np.ndarray,
    collections: list[LoadedCollection],
    top_k_initial: int,
) -> list[SearchResult]:

    results: list[SearchResult] = []

    for collection in collections:
        index = collection["index"]

        if index is None:
            continue

        metadata = collection["metadata"]
        collection_name = collection["collection_name"]

        for q_emb in query_embeddings:
            query: np.ndarray = np.ascontiguousarray([q_emb], dtype=np.float32)
            scores, indices = index.search(  # pyright: ignore[reportCallIssue]
                query, top_k_initial
            )

            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx == -1:
                    continue

                item = metadata[idx]

                # Attribute access -- ChunkMetadata is BaseModel
                results.append(
                    SearchResult(
                        score=float(score),
                        text=item.text,
                        source=item.source,
                        page=item.page,
                        collection=collection_name,
                        chunk_index=item.chunk_index,
                    )
                )

    return results


# ======================================================
# RERANK
# ======================================================


def rerank(results: list[SearchResult]) -> list[SearchResult]:
    results.sort(key=lambda x: x.score, reverse=True)

    dedup: list[SearchResult] = []
    seen: set[tuple[str, str, int]] = set()

    for r in results:
        key = (r.collection, r.source, r.chunk_index)

        if key in seen:
            continue

        seen.add(key)
        dedup.append(r)

    return dedup


# ======================================================
# FORMATTING
# ======================================================


def format_context_chunks(results: list[SearchResult]) -> list[str]:
    """Format SearchResult objects into context chunks for prompt building."""
    return [
        f"SOURCE: {r.source}\nCOLLECTION: {r.collection}\nPAGE: {r.page}\n\n{r.text}"
        for r in results
    ]


# ======================================================
# PUBLIC API
# ======================================================


def search(
    question: str,
    mode: str,
    collections: list[LoadedCollection],
    top_k_initial: int | None = None,
    top_k_final: int | None = None,
) -> tuple[list[SearchResult], float]:
    """
    top_k_initial/top_k_final: internal parameters, no longer with any caller
    that overrides them -- see GenerationOptions in src/api/schemas/chat.py,
    which no longer has these fields (retrieval became exclusively server-side
    configuration via .env). None (the only value that currently arrives from
    retrieve_node/chat.py) uses the settings default for the mode (SOFT/HARD).
    They are kept as function parameters because they remain a reasonable
    internal piece (e.g. tests, or a future internal caller needing a specific
    top_k) -- what was removed was the path that exposed them as per-request
    overrides from the API.
    """

    if mode == ChatMode.SOFT:
        default_initial = settings.soft_top_k_initial
        default_final = settings.soft_top_k_final
    else:
        default_initial = settings.hard_top_k_initial
        default_final = settings.hard_top_k_final

    top_k_initial = default_initial if top_k_initial is None else top_k_initial
    top_k_final = default_final if top_k_final is None else top_k_final

    queries = build_queries(question, mode)
    embeddings = encode_queries(queries)
    results = rerank(retrieve(embeddings, collections, top_k_initial))
    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
