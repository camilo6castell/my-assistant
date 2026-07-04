"""
Nota sobre # pyright: ignore[reportCallIssue] en index.search():
  Pylance lee stubs SWIG C++ de faiss; mypy tiene stubs del wrapper Python.
  La directiva pyright: es ignorada por mypy, sin unused-ignore.
"""

from __future__ import annotations

import numpy as np

from src.chat.modes import ChatMode
from src.config.settings import settings
from src.context.manager import LoadedCollection
from src.context.models import SearchResult
from src.embeddings.encoder import get_encoder


# ======================================================
# QUERY BUILDING
# ======================================================


def build_queries(question: str, mode: str) -> list[str]:
    """
    HARD: 1 query literal.
    SOFT: 3 queries — literal + 2 variantes semánticas.

    El historial viaja como mensajes de API en generate.py,
    no como variante de query adicional.
    """
    if mode == ChatMode.HARD:
        return [question]

    return [
        question,
        f"Explica el concepto: {question}",
        f"Relaciona ideas sobre: {question}",
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

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                item = metadata[idx]

                # Acceso por atributo — ChunkMetadata es BaseModel
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
# PUBLIC API
# ======================================================


def search(
    question: str,
    mode: str,
    collections: list[LoadedCollection],
) -> tuple[list[SearchResult], float]:

    if mode == ChatMode.SOFT:
        top_k_initial = settings.soft_top_k_initial
        top_k_final = settings.soft_top_k_final
    else:
        top_k_initial = settings.hard_top_k_initial
        top_k_final = settings.hard_top_k_final

    queries = build_queries(question, mode)
    embeddings = encode_queries(queries)
    results = rerank(retrieve(embeddings, collections, top_k_initial))
    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
