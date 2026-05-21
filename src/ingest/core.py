# src/ingest/core.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Any, Callable, cast

import faiss  # type: ignore[import-not-found]
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import (
    BASE_VECTOR_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
)
from src.utils.logger import logger

# faiss solo se importa para anotaciones en tiempo de análisis estático.
# En runtime faiss ya está importado arriba para su uso real;
# TYPE_CHECKING aquí documenta que usamos sus tipos en las firmas.
if TYPE_CHECKING:
    # faiss no proporciona stubs para el chequeo de tipos, así que
    # usamos Any como reemplazo para evitar errores de mypy/pyright.
    from typing import Any as FaissIndex
else:
    # En tiempo de ejecución, proporcionamos un alias compatible
    # para que las anotaciones que usan FaissIndex no fallen.
    FaissIndex = Any


model = SentenceTransformer(EMBED_MODEL)


# ======================================================
# TIPOS
# ======================================================


class CollectionPaths(TypedDict):
    """Rutas en disco de los artefactos de una colección."""

    vector_path: Path
    index_file: Path
    metadata_file: Path
    vectors_file: Path


class ChunkMetadata(TypedDict):
    """Metadata de un chunk tal como la persiste build_metadata."""

    source: str
    source_type: str
    page: int
    text: str
    chunk_index: int
    collection: str


class RawCollection(TypedDict):
    """
    Estructura que devuelve load_collection antes de que
    ContextManager agregue collection_name.
    """

    index: FaissIndex | None
    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None
    paths: CollectionPaths


# ======================================================
# CHUNKING
# ======================================================


def chunk_text(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ======================================================
# PATHS
# ======================================================


def get_collection_paths(collection: str) -> CollectionPaths:
    vector_path = BASE_VECTOR_PATH / collection
    vector_path.mkdir(parents=True, exist_ok=True)

    return CollectionPaths(
        vector_path=vector_path,
        index_file=vector_path / "index.faiss",
        metadata_file=vector_path / "metadata.pkl",
        vectors_file=vector_path / "vectors.npy",
    )


# ======================================================
# LOAD / SAVE
# ======================================================


def load_collection(collection: str) -> RawCollection:
    paths = get_collection_paths(collection)

    if paths["index_file"].exists():
        logger.info(f"Cargando colección: {collection}")

        read_index = cast(Callable[[str], Any], getattr(faiss, "read_index"))
        index = read_index(str(paths["index_file"]))

        with open(paths["metadata_file"], "rb") as f:
            metadata: list[ChunkMetadata] = pickle.load(f)

        vectors: np.ndarray | None = None

        if paths["vectors_file"].exists():
            vectors = np.load(paths["vectors_file"])
            logger.info("Embeddings cargados.")
        else:
            logger.warning("vectors.npy no encontrado.")

    else:
        logger.info(f"Creando nueva colección: {collection}")
        index = None
        metadata = []
        vectors = None

    return RawCollection(
        index=index,
        metadata=metadata,
        vectors=vectors,
        paths=paths,
    )


def save_collection(
    collection_data: RawCollection,
    new_embeddings: np.ndarray,
    new_metadata: list[ChunkMetadata],
) -> None:
    logger.info("Guardando colección...")

    index = collection_data["index"]
    metadata = collection_data["metadata"]
    existing_vectors = collection_data["vectors"]
    paths = collection_data["paths"]

    if existing_vectors is not None:
        all_vectors = np.vstack([existing_vectors, new_embeddings])
    else:
        all_vectors = new_embeddings

    if index is None:
        dimension = new_embeddings.shape[1]
        index = create_faiss_index(dimension)

    index.add(new_embeddings)
    metadata.extend(new_metadata)


# ======================================================
# EMBEDDINGS / FAISS
# ======================================================


def encode_chunks(chunks: list[str]) -> np.ndarray:
    logger.info(f"Generando embeddings para {len(chunks)} chunks")
    # Request numpy output explicitly so the return type is ndarray
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # The type stubs for SentenceTransformer.encode are imprecise; cast to
    # ndarray to satisfy static checkers, then ensure float32 for FAISS.
    embeddings_array = cast(np.ndarray, embeddings)
    return np.array(embeddings_array, dtype="float32")


def create_faiss_index(dimension: int) -> FaissIndex:
    logger.info(f"Creando índice FAISS (dim={dimension})")
    return faiss.IndexFlatIP(dimension)


# ======================================================
# METADATA BUILDER
# ======================================================


def build_metadata(
    source: str,
    source_type: str,
    page: int,
    chunk: str,
    chunk_index: int,
    collection: str,
) -> ChunkMetadata:
    return ChunkMetadata(
        source=source,
        source_type=source_type,
        page=page,
        text=chunk,
        chunk_index=chunk_index,
        collection=collection,
    )
