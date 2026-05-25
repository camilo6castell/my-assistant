from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chat.modes import ChatMode
from src.chat.types import TurnMemory
from src.config.settings import (
    BASE_TOP_K_FINAL,
    BASE_TOP_K_INITIAL,
    EMBED_MODEL,
    INTERPRETATIVE_TOP_K_FINAL,
    INTERPRETATIVE_TOP_K_INITIAL,
    MAX_TURNS,
)
from src.context.manager import LoadedCollection
from src.context.models import SearchResult

model = SentenceTransformer(EMBED_MODEL)


# ======================================================
# QUERY BUILDING
# ======================================================


def build_queries(
    question: str,
    mode: str,
    memory: list[TurnMemory],
) -> list[str]:

    if mode == ChatMode.RIGOROUS:
        return [question]

    history = " ".join(
        f"{turn['user']} {turn['assistant']}" for turn in memory[-MAX_TURNS:]
    )

    return [
        question,
        f"{history} {question}".strip(),
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
            # Different vector DBs expose different signatures for `search`.
            # Try common variants and fall back to an output-buffer style call.
            q_arr = np.ascontiguousarray([q_emb], dtype=np.float32)
            try:
                scores, indices = index.search(q_arr, top_k_initial)
            except TypeError:
                try:
                    scores, indices = index.search(q_arr, k=top_k_initial)
                except TypeError:
                    # Some indexes expect preallocated output buffers: search(x, k, distances, labels)
                    out_dist = np.empty((1, top_k_initial), dtype=np.float32)
                    out_idx = np.empty((1, top_k_initial), dtype=np.int64)
                    # call will fill out_dist and out_idx in-place
                    index.search(q_arr, top_k_initial, out_dist, out_idx)
                    scores, indices = out_dist, out_idx

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
    chat_memory: list[TurnMemory],
    collections: list[LoadedCollection],
) -> tuple[list[SearchResult], float]:

    if mode == ChatMode.INTERPRETATIVE:
        top_k_initial = INTERPRETATIVE_TOP_K_INITIAL
        top_k_final = INTERPRETATIVE_TOP_K_FINAL
    else:
        top_k_initial = BASE_TOP_K_INITIAL
        top_k_final = BASE_TOP_K_FINAL

    queries = build_queries(question, mode, chat_memory)
    embeddings = encode_queries(queries)
    results = rerank(retrieve(embeddings, collections, top_k_initial))
    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
