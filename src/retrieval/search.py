from typing import List

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

model = SentenceTransformer(
    EMBED_MODEL,
)


def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_queries(
    question: str,
    mode: str,
    memory: list[dict],
):

    if mode == ChatMode.RIGOROUS:
        return [question]

    history = ""

    for turn in memory[-MAX_TURNS:]:
        history += f"{turn['user']} {turn['assistant']} "

    return [
        question,
        history + question,
        f"Explica el concepto: {question}",
        f"Relaciona ideas sobre: {question}",
    ]


def encode_queries(queries):

    embeddings = model.encode(
        queries,
        normalize_embeddings=True,
    )

    return np.array(
        embeddings,
        dtype="float32",
    )


def retrieve(
    query_embeddings,
    collections,
    top_k_initial,
):

    results = []

    for collection in collections:

        index = collection["index"]

        if index is None:
            continue

        metadata = collection["metadata"]

        vectors = collection["vectors"]

        collection_name = collection["collection_name"]

        for q_emb in query_embeddings:

            q_emb = np.array([q_emb])

            scores, indices = index.search(
                q_emb,
                top_k_initial,
            )

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


def rerank(results):

    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    dedup = []

    seen = set()

    for r in results:

        key = (
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
    question,
    mode,
    chat_memory,
    collections,
):

    if mode == ChatMode.INTERPRETATIVE:

        top_k_initial = INTERPRETATIVE_TOP_K_INITIAL
        top_k_final = INTERPRETATIVE_TOP_K_FINAL

    else:

        top_k_initial = BASE_TOP_K_INITIAL
        top_k_final = BASE_TOP_K_FINAL

    queries = build_queries(
        question,
        mode,
        chat_memory,
    )

    embeddings = encode_queries(queries)

    results = retrieve(
        embeddings,
        collections,
        top_k_initial,
    )

    results = rerank(results)

    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
