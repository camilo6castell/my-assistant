import pickle

import faiss
import numpy as np

from src.utils.logger import logger
from src.config.settings import BASE_VECTOR_PATH


def list_contexts() -> list[str]:

    if not BASE_VECTOR_PATH.exists():

        logger.warning(f"Vectorstore path no existe: " f"{BASE_VECTOR_PATH}")

        return []

    return [d.name for d in BASE_VECTOR_PATH.iterdir() if d.is_dir()]


def load_context(context_name: str):

    path = BASE_VECTOR_PATH / context_name

    if not path.exists():

        logger.error(f"Contexto no encontrado: " f"{context_name}")

        return None

    logger.info(f"Cargando contexto: " f"{context_name}")

    index = faiss.read_index(str(path / "index.faiss"))

    with open(path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    vectors_path = path / "vectors.npy"

    if vectors_path.exists():

        chunk_vectors = np.load(vectors_path)

        logger.info("Embeddings precomputados cargados.")

    else:

        logger.warning("vectors.npy no encontrado.")

        chunk_vectors = None

    logger.info(f"Contexto cargado correctamente: " f"{context_name}")

    return {
        "name": context_name,
        "index": index,
        "metadata": metadata,
        "chunk_vectors": chunk_vectors,
    }
