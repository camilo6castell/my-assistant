"""
Gestion de eliminacion granular de documentos, URLs y fuentes del
vectorstore. Provee reconstruccion de indice FAISS y compactacion de
metadata tras las eliminaciones.

Nota sobre # pyright: ignore[reportCallIssue] en llamadas a faiss:
  Pylance lee stubs SWIG C++; mypy tiene stubs correctos para el wrapper
  Python. La directiva pyright: es ignorada por mypy, sin unused-ignore.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.config.settings import settings
from src.ingest.core import ChunkMetadata
from src.utils.logger import logger

# ======================================================
# HELPERS INTERNOS
# ======================================================


def _collection_paths(collection: str) -> dict[str, Path]:
    base: Path = Path(settings.vector_store_path) / collection
    return {
        "base": base,
        "index": base / "index.faiss",
        "metadata": base / "metadata.pkl",
        "vectors": base / "vectors.npy",
    }


def _load_raw(
    collection: str,
) -> tuple[list[ChunkMetadata], np.ndarray | None]:
    paths: dict[str, Path] = _collection_paths(collection)

    if not paths["metadata"].exists():
        logger.warning(f"metadata.pkl no encontrado | collection={collection}")
        return [], None

    with open(paths["metadata"], "rb") as f:
        raw: list[object] = pickle.load(f)
        metadata: list[ChunkMetadata] = [ChunkMetadata.model_validate(m) for m in raw]

    vectors: np.ndarray | None = None

    if paths["vectors"].exists():
        vectors = np.load(paths["vectors"])
    else:
        logger.warning(f"vectors.npy no encontrado | collection={collection}")

    return metadata, vectors


def _save_raw(
    collection: str,
    metadata: list[ChunkMetadata],
    vectors: np.ndarray,
    index: faiss.Index,
) -> None:
    paths: dict[str, Path] = _collection_paths(collection)
    paths["base"].mkdir(parents=True, exist_ok=True)

    with open(paths["metadata"], "wb") as f:
        pickle.dump([m.model_dump() for m in metadata], f)

    np.save(paths["vectors"], vectors)
    faiss.write_index(index, str(paths["index"]))

    logger.info(
        f"Colección guardada | collection={collection} | chunks={len(metadata)}"
    )


def _clear_collection_files(collection: str) -> None:
    paths: dict[str, Path] = _collection_paths(collection)

    for key in ("index", "metadata", "vectors"):
        p: Path = paths[key]
        if p.exists():
            p.unlink()
            logger.info(f"Archivo eliminado: {p}")


def _to_f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


# ======================================================
# RECONSTRUCCION DE INDICE
# ======================================================


def rebuild_index(collection: str) -> faiss.Index | None:
    paths: dict[str, Path] = _collection_paths(collection)

    if not paths["vectors"].exists():
        logger.warning(
            f"No se puede reconstruir: vectors.npy ausente "
            f"| collection={collection}"
        )
        return None

    vectors: np.ndarray = np.load(paths["vectors"])

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        logger.warning(
            f"vectors.npy vacío o con forma inválida "
            f"| shape={vectors.shape} | collection={collection}"
        )
        return None

    dimension: int = vectors.shape[1]
    index: faiss.Index = faiss.IndexFlatIP(dimension)
    index.add(_to_f32(vectors))  # pyright: ignore[reportCallIssue]

    faiss.write_index(index, str(paths["index"]))

    logger.info(
        f"Índice reconstruido | collection={collection} "
        f"| vectors={vectors.shape[0]} | dim={dimension}"
    )

    return index


# ======================================================
# VACUUM / COMPACTACION
# ======================================================


def vacuum_collection(collection: str) -> dict[str, int]:
    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None
    metadata, vectors = _load_raw(collection)

    before: int = len(metadata)

    if not metadata or vectors is None:
        logger.info(f"Vacuum: nada que compactar | collection={collection}")
        return {"before": before, "after": before, "removed": 0}

    valid_indices: list[int] = [i for i in range(len(metadata)) if i < len(vectors)]
    clean_metadata: list[ChunkMetadata] = [metadata[i] for i in valid_indices]
    clean_vectors: np.ndarray = _to_f32(vectors[valid_indices])

    after: int = len(clean_metadata)
    removed: int = before - after

    if removed == 0 and np.array_equal(vectors, clean_vectors):
        logger.info(f"Vacuum: colección ya consistente | collection={collection}")
        return {"before": before, "after": after, "removed": 0}

    dimension: int = clean_vectors.shape[1]
    index: faiss.Index = faiss.IndexFlatIP(dimension)
    index.add(clean_vectors)  # pyright: ignore[reportCallIssue]

    _save_raw(collection, clean_metadata, clean_vectors, index)

    logger.info(
        f"Vacuum completado | collection={collection} "
        f"| before={before} | after={after} | removed={removed}"
    )

    return {"before": before, "after": after, "removed": removed}


# ======================================================
# ELIMINACION POR FUENTE
# ======================================================


def delete_by_source(
    collection: str,
    source: str,
    *,
    rebuild: bool = True,
) -> int:
    return delete_by_sources(collection, [source], rebuild=rebuild)


def delete_by_sources(
    collection: str,
    sources: Sequence[str],
    *,
    rebuild: bool = True,
) -> int:
    source_set: set[str] = set(sources)

    if not source_set:
        logger.warning("delete_by_sources: lista de fuentes vacía.")
        return 0

    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None
    metadata, vectors = _load_raw(collection)

    if not metadata:
        logger.warning(
            f"delete_by_sources: colección vacía o inexistente "
            f"| collection={collection}"
        )
        return 0

    # Acceso por atributo — ChunkMetadata es BaseModel, no TypedDict
    keep_indices: list[int] = [
        i for i, m in enumerate(metadata) if m.source not in source_set
    ]

    removed_count: int = len(metadata) - len(keep_indices)

    if removed_count == 0:
        logger.info(
            f"delete_by_sources: ninguna fuente encontrada "
            f"| sources={source_set} | collection={collection}"
        )
        return 0

    logger.info(
        f"Eliminando chunks | sources={source_set} "
        f"| count={removed_count} | collection={collection}"
    )

    clean_metadata: list[ChunkMetadata] = [metadata[i] for i in keep_indices]
    clean_vectors: np.ndarray

    if vectors is not None and len(vectors) > 0:
        valid_keep: list[int] = [i for i in keep_indices if i < len(vectors)]
        clean_vectors = _to_f32(vectors[valid_keep])
    else:
        clean_vectors = np.empty((0,), dtype=np.float32)

    if clean_metadata and clean_vectors.ndim == 2 and clean_vectors.shape[0] > 0:
        dimension: int = clean_vectors.shape[1]
        index: faiss.Index = faiss.IndexFlatIP(dimension)
        index.add(clean_vectors)  # pyright: ignore[reportCallIssue]
        _save_raw(collection, clean_metadata, clean_vectors, index)
    else:
        _clear_collection_files(collection)
        logger.info(f"Colección vaciada completamente | collection={collection}")

    if rebuild and clean_metadata:
        vacuum_collection(collection)

    return removed_count


# ======================================================
# LIMPIEZA TOTAL
# ======================================================


def clear_collection(collection: str) -> None:
    _clear_collection_files(collection)
    logger.info(f"Colección limpiada | collection={collection}")


# ======================================================
# ALIAS SEMANTICOS PARA URLs
# ======================================================


def delete_url(collection: str, url: str, *, rebuild: bool = True) -> int:
    return delete_by_source(collection, url, rebuild=rebuild)


def delete_urls(collection: str, urls: Sequence[str], *, rebuild: bool = True) -> int:
    return delete_by_sources(collection, urls, rebuild=rebuild)


# ======================================================
# INSPECCION
# ======================================================


class SourceSummary(BaseModel):
    """Resumen de una fuente indexada en una colección."""

    model_config = ConfigDict(frozen=True)

    source: str
    source_type: str
    chunks: int


def list_sources(collection: str) -> list[SourceSummary]:
    """
    Devuelve un resumen de las fuentes indexadas en la colección,
    ordenadas por nombre de fuente.
    """
    metadata: list[ChunkMetadata]
    metadata, _ = _load_raw(collection)

    # Dos dicts bien tipados en lugar de dict[str, object],
    # que causaba errores de tipo al acceder al contador.
    chunk_counts: dict[str, int] = {}
    source_types: dict[str, str] = {}

    for entry in metadata:
        src: str = entry.source
        chunk_counts[src] = chunk_counts.get(src, 0) + 1
        if src not in source_types:
            source_types[src] = entry.source_type

    return sorted(
        [
            SourceSummary(
                source=src,
                source_type=source_types[src],
                chunks=count,
            )
            for src, count in chunk_counts.items()
        ],
        key=lambda x: x.source,
    )
