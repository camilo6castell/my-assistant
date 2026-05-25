import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def normalize_embedding(embedding: np.ndarray | list[float]) -> np.ndarray:
    arr = np.array(embedding, dtype=np.float32)
    norm: float = float(np.linalg.norm(arr))

    if norm == 0.0:
        return arr

    # FIX #7: cast explícito — arr / norm produce Any en los stubs de numpy
    return np.asarray(arr / norm, dtype=np.float32)
