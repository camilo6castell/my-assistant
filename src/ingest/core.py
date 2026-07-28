"""
Note on # pyright: ignore[reportCallIssue] in faiss calls:
  Pylance reads SWIG C++ stubs for faiss (add(n, x, ...) / search(n, x, k, D, I, ...))
  instead of the Python wrapper (add(x) / search(x, k) -> (D, I)).
  mypy has correct stubs and does not need suppression.
  The pyright: directive is ignored by mypy, avoiding unused-ignore.

Note on ChunkMetadata as BaseModel:
  Being a BaseModel, metadata is validated at creation time (build_metadata).
  Pickle serialization uses model_dump() to save flat dicts,
  and model_validate() when loading to reconstruct the models — this
  ensures backward compatibility with pickles created before this migration.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.config.settings import settings
from src.nlp.embedders.encoder import get_encoder
from src.utils.logger import logger

if TYPE_CHECKING:
    from faiss import Index as FaissIndex


# ======================================================
# TYPES
# ======================================================


class CollectionPaths(TypedDict):
    """Disk paths for collection artifacts."""

    vector_path: Path
    index_file: Path
    metadata_file: Path
    vectors_file: Path


class ChunkMetadata(BaseModel):
    """
    Metadata for an indexed chunk.

    BaseModel instead of TypedDict because:
      - Validates types at construction time (build_metadata).
      - frozen=True ensures chunks are not mutated post-ingest.
      - model_validate / model_dump handle pickle serialization.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    source_type: str
    page: int
    text: str
    chunk_index: int
    collection: str
    file_id: str | None = None
    """
    Source file ID within an ephemeral collection
    (src/context/ephemeral.py). None for normal persisted collections
    -- explicit default so old pickles (without this key)
    still validate with model_validate() without breaking.
    """


class RawCollection(TypedDict):
    """
    Structure returned by load_collection before ContextManager
    adds collection_name.

    TypedDict (not BaseModel) because it contains faiss.Index and np.ndarray,
    which Pydantic cannot validate.
    """

    index: FaissIndex | None
    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None
    paths: CollectionPaths


# ======================================================
# HELPERS
# ======================================================


def _to_f32(arr: np.ndarray) -> np.ndarray:
    """Converts to float32 C-contiguous array required by faiss at runtime."""
    return np.ascontiguousarray(arr, dtype=np.float32)


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
        end = start + settings.chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += settings.chunk_size - settings.chunk_overlap

    return chunks


# ======================================================
# PATHS
# ======================================================


def get_collection_paths(collection: str) -> CollectionPaths:
    vector_path = settings.vector_store_path_for_backend / collection
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
        logger.info(f"Loading collection: {collection}")

        index: FaissIndex | None = faiss.read_index(str(paths["index_file"]))

        with open(paths["metadata_file"], "rb") as f:
            raw: list[object] = pickle.load(f)
            # model_validate handles both dicts (pickles from before this
            # migration) and already-serialized ChunkMetadata instances.
            metadata: list[ChunkMetadata] = [ChunkMetadata.model_validate(m) for m in raw]

        vectors: np.ndarray | None = None

        if paths["vectors_file"].exists():
            vectors = np.load(paths["vectors_file"])
            logger.info("Embeddings loaded.")
        else:
            logger.warning("vectors.npy not found.")

    else:
        logger.info(f"Creating new collection: {collection}")
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
    logger.info("Saving collection...")

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

    index.add(_to_f32(new_embeddings))  # pyright: ignore[reportCallIssue]
    metadata.extend(new_metadata)

    faiss.write_index(index, str(paths["index_file"]))
    np.save(paths["vectors_file"], all_vectors)

    # Serialize as flat dicts for maximum portability and compatibility.
    with open(paths["metadata_file"], "wb") as f:
        pickle.dump([m.model_dump() for m in metadata], f)

    logger.info("Collection saved successfully.")


# ======================================================
# EMBEDDINGS / FAISS
# ======================================================


def encode_chunks(chunks: list[str]) -> np.ndarray:
    logger.info(
        f"Generating embeddings for {len(chunks)} chunks | "
        f"backend={settings.embedding_backend} | model={settings.embedding_model}"
    )
    return get_encoder().encode(chunks)


def create_faiss_index(dimension: int) -> FaissIndex:
    logger.info(f"Creating FAISS index (dim={dimension})")
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
    file_id: str | None = None,
) -> ChunkMetadata:
    return ChunkMetadata(
        source=source,
        source_type=source_type,
        page=page,
        text=chunk,
        chunk_index=chunk_index,
        collection=collection,
        file_id=file_id,
    )
