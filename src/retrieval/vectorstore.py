import numpy as np


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def normalize_embedding(embedding):
    embedding = np.array(embedding).astype("float32")

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm
