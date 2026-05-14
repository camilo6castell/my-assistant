from typing import List, Tuple, Optional

import faiss
import numpy as np

from FlagEmbedding import FlagModel

from src.utils.logger import logger

from src.utils.env import (
    EMBED_MODEL,
    BASE_TOP_K_INITIAL,
    BASE_TOP_K_FINAL,
    MAX_TURNS,
)

model = FlagModel(
    EMBED_MODEL,
    use_fp16=False,
    trust_remote_code=False,
)


def build_search_query(
    question: str,
    interpretative_mode: bool,
    chat_memory: List[dict],
) -> List[str]:

    if not interpretative_mode:
        return [question]

    history_text = ""

    for turn in chat_memory[-MAX_TURNS:]:

        history_text += f"{turn['user']} " f"{turn['assistant']} "

    base_query = history_text + question

    return [
        base_query,
        f"Explica conceptualmente: {question}",
        f"Principio general relacionado con: {question}",
    ]


def search(
    question: str,
    interpretative_mode: bool,
    chat_memory: List[dict],
    index: faiss.Index,
    metadata: List[dict],
    chunk_vectors: Optional[np.ndarray],
) -> Tuple[List[int], float]:

    queries = build_search_query(
        question=question,
        interpretative_mode=interpretative_mode,
        chat_memory=chat_memory,
    )

    if interpretative_mode:
        top_k_initial = 25
        top_k_final = 7
    else:
        top_k_initial = BASE_TOP_K_INITIAL
        top_k_final = BASE_TOP_K_FINAL

    logger.info(
        f"Ejecutando búsqueda | "
        f"queries={len(queries)} | "
        f"top_k_initial={top_k_initial}"
    )

    all_candidate_indices = set()

    query_embeddings = []

    for q in queries:

        emb = model.encode([q])

        emb = np.array(emb).astype("float32")

        faiss.normalize_L2(emb)

        query_embeddings.append(emb[0])

        _, indices = index.search(
            emb,
            top_k_initial,
        )

        for idx in indices[0]:

            if idx != -1:
                all_candidate_indices.add(int(idx))

    candidate_indices = list(all_candidate_indices)

    logger.info(f"Candidates recuperados: " f"{len(candidate_indices)}")

    if not candidate_indices:
        return [], 0.0

    scores = []

    for idx in candidate_indices:

        if chunk_vectors is not None:

            chunk_vector = chunk_vectors[idx]

        else:

            chunk_text = metadata[idx]["text"]

            chunk_embedding = model.encode([chunk_text])

            chunk_embedding = np.array(chunk_embedding).astype("float32")

            faiss.normalize_L2(chunk_embedding)

            chunk_vector = chunk_embedding[0]

        score_sum = 0.0

        for q_emb in query_embeddings:
            score_sum += float(np.dot(q_emb, chunk_vector))

        avg_score = score_sum / len(query_embeddings)

        scores.append(avg_score)

    sorted_pairs = sorted(
        zip(candidate_indices, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_pairs = sorted_pairs[:top_k_final]

    top_indices = [idx for idx, _ in top_pairs]

    top_scores = [score for _, score in top_pairs]

    avg_confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0

    logger.info(f"Top chunks seleccionados: " f"{len(top_indices)}")

    return top_indices, avg_confidence
