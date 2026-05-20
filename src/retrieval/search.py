from typing import Any

import numpy as np

from sentence_transformers import SentenceTransformer

from src.chat.modes import ChatMode

from src.config.settings import (
    EMBED_MODEL,
    BASE_TOP_K_INITIAL,
    BASE_TOP_K_FINAL,
    INTERPRETATIVE_TOP_K_INITIAL,
    INTERPRETATIVE_TOP_K_FINAL,
    MAX_TURNS,
)

from src.context.models import SearchResult

model: SentenceTransformer = SentenceTransformer(
    EMBED_MODEL,
)


def cosine_similarity(a: Any, b: Any) -> float:

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_queries(
    question: str,
    mode: str,
    memory: list[dict[str, str]],
) -> list[str]:

    if mode == ChatMode.RIGOROUS:
        return [question]

    history: str = ""

    for turn in memory[-MAX_TURNS:]:
        history += f"{turn['user']} {turn['assistant']} "

    return [
        question,
        history + question,
        f"Explica el concepto: {question}",
        f"Relaciona ideas sobre: {question}",
    ]


def encode_queries(queries: list[str]) -> np.ndarray:

    embeddings: Any = model.encode(
        queries,
        normalize_embeddings=True,
    )

    return np.array(
        embeddings,
        dtype="float32",
    )


def retrieve(
    query_embeddings: np.ndarray,
    collections: list[dict[str, Any]],
    top_k_initial: int,
) -> list[SearchResult]:

    results: list[SearchResult] = []

    for collection in collections:

        index: Any = collection["index"]

        if index is None:
            continue

        metadata: list[dict[str, Any]] = collection["metadata"]

        vectors: Any = collection["vectors"]

        collection_name: str = collection["collection_name"]

        for q_emb in query_embeddings:

            q_emb = np.array([q_emb])

            scores, indices = index.search(
                q_emb,
                top_k_initial,
            )

            for score, idx in zip(scores[0], indices[0]):

                if idx == -1:
                    continue

                item: dict[str, Any] = metadata[idx]

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


def rerank(results: list[SearchResult]) -> list[SearchResult]:

    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    dedup: list[SearchResult] = []

    seen: set[tuple[str, str, int]] = set()

    for r in results:

        key: tuple[str, str, int] = (
            r.collection,
            r.source,
            r.chunk_index,
        )

        if key in seen:
            continue

        seen.add(key)

        dedup.append(r)

    return dedup


def search(
    question: str,
    mode: str,
    chat_memory: list[dict[str, str]],
    collections: list[dict[str, Any]],
) -> tuple[list[SearchResult], float]:

    if mode == ChatMode.INTERPRETATIVE:

        top_k_initial: int = INTERPRETATIVE_TOP_K_INITIAL
        top_k_final: int = INTERPRETATIVE_TOP_K_FINAL

    else:

        top_k_initial = BASE_TOP_K_INITIAL
        top_k_final = BASE_TOP_K_FINAL

    queries: list[str] = build_queries(
        question,
        mode,
        chat_memory,
    )

    embeddings: np.ndarray = encode_queries(queries)

    results: list[SearchResult] = retrieve(
        embeddings,
        collections,
        top_k_initial,
    )

    results = rerank(results)

    final_results: list[SearchResult] = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence: float = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
