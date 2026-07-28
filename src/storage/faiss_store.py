"""
Unified FAISS + pickle + numpy persistence layer.

Single source of truth for:
  - Loading collections from disk (index.faiss + metadata.pkl + vectors.npy)
  - Saving collections to disk
  - Rebuilding FAISS indexes from vectors
  - Vacuuming/compacting collections after deletions
  - Path resolution for collection artifacts

All other modules (ingest/core.py, context/delete.py, context/ephemeral.py)
delegate here instead of reimplementing the same serialization logic.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.config.settings import settings
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
# NUMPY HELPERS
# ======================================================


def to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 C-contiguous array required by FAISS at runtime."""
    return np.ascontiguousarray(arr, dtype=np.float32)


# ======================================================
# PATHS
# ======================================================


def get_collection_paths(collection: str) -> CollectionPaths:
    """Resolve disk paths for a collection's artifacts."""
    vector_path = settings.vector_store_path_for_backend / collection
    vector_path.mkdir(parents=True, exist_ok=True)

    return CollectionPaths(
        vector_path=vector_path,
        index_file=vector_path / "index.faiss",
        metadata_file=vector_path / "metadata.pkl",
        vectors_file=vector_path / "vectors.npy",
    )


# ======================================================
# INDEX
# ======================================================


def create_faiss_index(dimension: int) -> FaissIndex:
    """Create a new FAISS inner-product index."""
    logger.info(f"Creating FAISS index (dim={dimension})")
    return faiss.IndexFlatIP(dimension)


def rebuild_index_from_vectors(vectors: np.ndarray) -> FaissIndex:
    """Rebuild a FAISS index from an existing vectors array."""
    dimension = vectors.shape[1]
    index = create_faiss_index(dimension)
    index.add(to_f32(vectors))  # pyright: ignore[reportCallIssue]
    return index


# ======================================================
# LOAD / SAVE
# ======================================================


def load_collection(collection: str) -> RawCollection:
    """Load a collection (index + metadata + vectors) from disk."""
    paths = get_collection_paths(collection)

    if paths["index_file"].exists():
        logger.info(f"Loading collection: {collection}")

        index: FaissIndex | None = faiss.read_index(str(paths["index_file"]))

        with open(paths["metadata_file"], "rb") as f:
            raw: list[object] = pickle.load(f)
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
    """Append new embeddings/metadata to a collection and persist to disk."""
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

    index.add(to_f32(new_embeddings))  # pyright: ignore[reportCallIssue]
    metadata.extend(new_metadata)

    faiss.write_index(index, str(paths["index_file"]))
    np.save(paths["vectors_file"], all_vectors)

    with open(paths["metadata_file"], "wb") as f:
        pickle.dump([m.model_dump() for m in metadata], f)

    logger.info("Collection saved successfully.")


# ======================================================
# DELETION HELPERS
# ======================================================


def clear_collection_files(collection: str) -> None:
    """Delete all artifact files for a collection."""
    paths = get_collection_paths(collection)
    for key in ("index_file", "metadata_file", "vectors_file"):
        p: Path = paths[key]
        if p.exists():
            p.unlink()
            logger.info(f"File deleted: {p}")


def load_raw(
    collection: str,
) -> tuple[list[ChunkMetadata], np.ndarray | None]:
    """Load metadata and vectors without the FAISS index."""
    paths = get_collection_paths(collection)

    if not paths["metadata_file"].exists():
        logger.warning(f"metadata.pkl not found | collection={collection}")
        return [], None

    with open(paths["metadata_file"], "rb") as f:
        raw: list[object] = pickle.load(f)
        metadata: list[ChunkMetadata] = [ChunkMetadata.model_validate(m) for m in raw]

    vectors: np.ndarray | None = None
    if paths["vectors_file"].exists():
        vectors = np.load(paths["vectors_file"])
    else:
        logger.warning(f"vectors.npy not found | collection={collection}")

    return metadata, vectors


def save_raw(
    collection: str,
    metadata: list[ChunkMetadata],
    vectors: np.ndarray,
    index: FaissIndex,
) -> None:
    """Save metadata, vectors, and index to disk."""
    paths = get_collection_paths(collection)
    paths["vector_path"].mkdir(parents=True, exist_ok=True)

    with open(paths["metadata_file"], "wb") as f:
        pickle.dump([m.model_dump() for m in metadata], f)

    np.save(paths["vectors_file"], vectors)
    faiss.write_index(index, str(paths["index_file"]))

    logger.info(f"Collection saved | collection={collection} | chunks={len(metadata)}")


def rebuild_index(collection: str) -> FaissIndex | None:
    """Rebuild the FAISS index from vectors.npy for a collection."""
    paths = get_collection_paths(collection)

    if not paths["vectors_file"].exists():
        logger.warning(f"Cannot rebuild: vectors.npy missing | collection={collection}")
        return None

    vectors: np.ndarray = np.load(paths["vectors_file"])

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        logger.warning(
            f"vectors.npy empty or with invalid shape "
            f"| shape={vectors.shape} | collection={collection}"
        )
        return None

    index = rebuild_index_from_vectors(vectors)
    faiss.write_index(index, str(paths["index_file"]))

    logger.info(
        f"Index rebuilt | collection={collection} "
        f"| vectors={vectors.shape[0]} | dim={vectors.shape[1]}"
    )
    return index


def vacuum_collection(collection: str) -> dict[str, int]:
    """Compact metadata/vectors after deletions, rebuilding the index."""
    metadata, vectors = load_raw(collection)
    before = len(metadata)

    if not metadata or vectors is None:
        logger.info(f"Vacuum: nothing to compact | collection={collection}")
        return {"before": before, "after": before, "removed": 0}

    valid_indices = [i for i in range(len(metadata)) if i < len(vectors)]
    clean_metadata = [metadata[i] for i in valid_indices]
    clean_vectors = to_f32(vectors[valid_indices])

    after = len(clean_metadata)
    removed = before - after

    if removed == 0 and np.array_equal(vectors, clean_vectors):
        logger.info(f"Vacuum: collection already consistent | collection={collection}")
        return {"before": before, "after": after, "removed": 0}

    index = rebuild_index_from_vectors(clean_vectors)
    save_raw(collection, clean_metadata, clean_vectors, index)

    logger.info(
        f"Vacuum completed | collection={collection} "
        f"| before={before} | after={after} | removed={removed}"
    )
    return {"before": before, "after": after, "removed": removed}


def delete_by_sources(
    collection: str,
    sources: Sequence[str],
    *,
    rebuild: bool = True,
) -> int:
    """Remove all chunks whose source is in `sources`."""
    source_set = set(sources)

    if not source_set:
        logger.warning("delete_by_sources: empty source list.")
        return 0

    metadata, vectors = load_raw(collection)

    if not metadata:
        logger.warning(
            f"delete_by_sources: collection empty or nonexistent | collection={collection}"
        )
        return 0

    keep_indices = [i for i, m in enumerate(metadata) if m.source not in source_set]
    removed_count = len(metadata) - len(keep_indices)

    if removed_count == 0:
        logger.info(
            f"delete_by_sources: no source found | sources={source_set} | collection={collection}"
        )
        return 0

    logger.info(
        f"Deleting chunks | sources={source_set} | count={removed_count} | collection={collection}"
    )

    clean_metadata = [metadata[i] for i in keep_indices]
    clean_vectors: np.ndarray

    if vectors is not None and len(vectors) > 0:
        valid_keep = [i for i in keep_indices if i < len(vectors)]
        clean_vectors = to_f32(vectors[valid_keep])
    else:
        clean_vectors = np.empty((0,), dtype=np.float32)

    if clean_metadata and clean_vectors.ndim == 2 and clean_vectors.shape[0] > 0:
        index = rebuild_index_from_vectors(clean_vectors)
        save_raw(collection, clean_metadata, clean_vectors, index)
    else:
        clear_collection_files(collection)
        logger.info(f"Collection completely emptied | collection={collection}")

    if rebuild and clean_metadata:
        vacuum_collection(collection)

    return removed_count
