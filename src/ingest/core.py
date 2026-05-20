from pathlib import Path
from typing import Any
import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from src.utils.logger import logger

from src.config.settings import (
    BASE_VECTOR_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

model: SentenceTransformer = SentenceTransformer(
    EMBED_MODEL,
)


def chunk_text(text: str) -> list[str]:

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []

    start: int = 0

    while start < len(text):

        end: int = start + CHUNK_SIZE

        chunk: str = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def get_collection_paths(
    collection: str,
) -> dict[str, Path]:

    vector_path: Path = BASE_VECTOR_PATH / collection

    vector_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "vector_path": vector_path,
        "index_file": vector_path / "index.faiss",
        "metadata_file": vector_path / "metadata.pkl",
        "vectors_file": vector_path / "vectors.npy",
    }


def load_collection(collection: str) -> dict[str, Any]:

    paths: dict[str, Path] = get_collection_paths(collection)

    index_file: Path = paths["index_file"]
    metadata_file: Path = paths["metadata_file"]
    vectors_file: Path = paths["vectors_file"]

    if index_file.exists():

        logger.info(f"Cargando colección: {collection}")

        index: faiss.Index = faiss.read_index(str(index_file))

        with open(
            metadata_file,
            "rb",
        ) as f:
            metadata: list[dict[str, Any]] = pickle.load(f)

        vectors: np.ndarray | None = None

        if vectors_file.exists():

            vectors = np.load(vectors_file)

            logger.info("Embeddings cargados.")

        else:

            logger.warning("vectors.npy no encontrado.")

    else:

        logger.info(f"Creando nueva colección: {collection}")

        index = None
        metadata = []
        vectors = None

    return {
        "index": index,
        "metadata": metadata,
        "vectors": vectors,
        "paths": paths,
    }


def encode_chunks(
    chunks: list[str],
) -> np.ndarray:

    logger.info(f"Generando embeddings para {len(chunks)} chunks")

    embeddings: Any = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.array(
        embeddings,
        dtype="float32",
    )


def create_faiss_index(
    dimension: int,
) -> faiss.Index:

    logger.info(f"Creando índice FAISS (dim={dimension})")

    return faiss.IndexFlatIP(dimension)


def save_collection(
    collection_data: dict[str, Any],
    new_embeddings: np.ndarray,
    new_metadata: list[dict[str, Any]],
) -> None:

    logger.info("Guardando colección...")

    index: faiss.Index | None = collection_data["index"]
    metadata: list[dict[str, Any]] = collection_data["metadata"]
    existing_vectors: np.ndarray | None = collection_data["vectors"]
    paths: dict[str, Path] = collection_data["paths"]

    if existing_vectors is not None:

        all_vectors: np.ndarray = np.vstack(
            [
                existing_vectors,
                new_embeddings,
            ]
        )

    else:

        all_vectors = new_embeddings

    if index is None:

        dimension: int = new_embeddings.shape[1]

        index = create_faiss_index(dimension)

    index.add(new_embeddings)

    metadata.extend(new_metadata)

    faiss.write_index(
        index,
        str(paths["index_file"]),
    )

    np.save(
        paths["vectors_file"],
        all_vectors,
    )

    with open(
        paths["metadata_file"],
        "wb",
    ) as f:

        pickle.dump(
            metadata,
            f,
        )

    logger.info("Colección guardada correctamente.")


def build_metadata(
    source: str,
    source_type: str,
    page: int,
    chunk: str,
    chunk_index: int,
    collection: str,
) -> dict[str, Any]:

    return {
        "source": source,
        "source_type": source_type,
        "page": page,
        "text": chunk,
        "chunk_index": chunk_index,
        "collection": collection,
    }
