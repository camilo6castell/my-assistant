from typing import Any

import numpy as np


def cosine_similarity(a: Any, b: Any) -> float:
    return float(np.dot(a, b))


def normalize_embedding(embedding: Any) -> np.ndarray:
    embedding = np.array(embedding).astype("float32")

    norm: float = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm
