"""
Nota sobre # pyright: ignore[reportCallIssue] en index.search():
  Pylance lee stubs SWIG C++ de faiss; mypy tiene stubs del wrapper Python.
  La directiva pyright: es ignorada por mypy, sin unused-ignore.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chat.modes import ChatMode
from src.chat.types import TurnMemory
from src.config.settings import (
    BASE_TOP_K_FINAL,
    BASE_TOP_K_INITIAL,
    EMBED_MODEL,
    SOFT_TOP_K_FINAL,
    SOFT_TOP_K_INITIAL,
)
from src.context.manager import LoadedCollection
from src.context.models import SearchResult

model = SentenceTransformer(EMBED_MODEL)


# ======================================================
# QUERY BUILDING
# ======================================================


def build_queries(question: str, mode: str) -> list[str]:
    """
    Genera las variantes de búsqueda a partir de la pregunta.

    RIGUROSO:      1 query — la pregunta literal.
    INTERPRETATIVO: 3 queries — la pregunta literal más dos variantes
                   semánticas que amplían el ángulo de búsqueda.

    El historial de conversación ya viaja vía mensajes de la API
    (build_messages en generate.py), por lo que no es necesario
    incorporarlo aquí como variante de query adicional.
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
    embeddings = model.encode(queries, normalize_embeddings=True)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


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
            scores, indices = index.search(
                query, top_k_initial
            )  # pyright: ignore[reportCallIssue]

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                item = metadata[idx]

                results.append(
                    SearchResult(
                        score=float(score),
                        text=item["text"],
                        source=item["source"],
                        page=item["page"],
                        collection=collection_name,
                        chunk_index=item["chunk_index"],
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
    """
    Punto de entrada de la búsqueda semántica.

    chat_memory se mantiene en la firma para que interface.py no necesite
    cambios, pero ya no se usa aquí — viaja directamente al LLM vía API.
    """
    if mode == ChatMode.SOFT:
        top_k_initial = SOFT_TOP_K_INITIAL
        top_k_final = SOFT_TOP_K_FINAL
    else:
        top_k_initial = BASE_TOP_K_INITIAL
        top_k_final = BASE_TOP_K_FINAL

    queries = build_queries(question, mode)
    embeddings = encode_queries(queries)
    results = rerank(retrieve(embeddings, collections, top_k_initial))
    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
