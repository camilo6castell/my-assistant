"""
src/context/delete.py

Gestión de eliminación granular de documentos, URLs y fuentes del
vectorstore. Provee reconstrucción de índice FAISS y compactación de
metadata tras las eliminaciones.

Operaciones principales:
  - delete_by_source    → elimina todos los chunks de un archivo/URL
  - delete_by_sources   → elimina múltiples fuentes en un solo paso
  - delete_url          → alias semántico para fuentes web
  - delete_urls         → alias para lotes de URLs
  - rebuild_index       → reconstruye el índice FAISS desde vectors.npy
  - vacuum_collection   → compacta vectores y metadata, reescribe disco
  - clear_collection    → elimina todos los artefactos de la colección
  - list_sources        → introspección: fuentes y chunks por fuente
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Any

import faiss
import numpy as np

from src.config.settings import BASE_VECTOR_PATH
from src.utils.logger import logger

# ======================================================
# HELPERS INTERNOS
# ======================================================


def _collection_paths(collection: str) -> dict[str, Path]:
    """Devuelve las rutas canónicas de los artefactos de una colección."""
    base: Path = Path(BASE_VECTOR_PATH) / collection
    return {
        "base": base,
        "index": base / "index.faiss",
        "metadata": base / "metadata.pkl",
        "vectors": base / "vectors.npy",
    }


def _load_raw(collection: str) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """
    Carga metadata y vectores sin pasar por load_collection.
    Retorna (metadata, vectors). vectors puede ser None.
    """
    paths: dict[str, Path] = _collection_paths(collection)

    if not paths["metadata"].exists():
        logger.warning(f"metadata.pkl no encontrado | collection={collection}")
        return [], None

    with open(paths["metadata"], "rb") as f:
        metadata: list[dict[str, Any]] = pickle.load(f)

    vectors: np.ndarray | None = None

    if paths["vectors"].exists():
        vectors = np.load(paths["vectors"])
    else:
        logger.warning(f"vectors.npy no encontrado | collection={collection}")

    return metadata, vectors


def _save_raw(
    collection: str,
    metadata: list[dict[str, Any]],
    vectors: np.ndarray,
    index: faiss.Index,
) -> None:
    """Persiste los tres artefactos de una colección en disco."""
    paths: dict[str, Path] = _collection_paths(collection)
    paths["base"].mkdir(parents=True, exist_ok=True)

    with open(paths["metadata"], "wb") as f:
        pickle.dump(metadata, f)

    np.save(paths["vectors"], vectors)

    faiss.write_index(index, str(paths["index"]))

    logger.info(
        f"Colección guardada | collection={collection} " f"| chunks={len(metadata)}"
    )


def _clear_collection_files(collection: str) -> None:
    """Elimina los artefactos de una colección que quedó vacía."""
    paths: dict[str, Path] = _collection_paths(collection)

    for key in ("index", "metadata", "vectors"):
        p: Path = paths[key]
        if p.exists():
            p.unlink()
            logger.info(f"Archivo eliminado: {p}")


# ======================================================
# RECONSTRUCCIÓN DE ÍNDICE
# ======================================================


def rebuild_index(collection: str) -> faiss.Index | None:
    """
    Reconstruye el índice FAISS desde vectors.npy.

    Útil después de cualquier operación que modifique el conjunto de
    vectores. El índice anterior es sobreescrito en disco.

    Retorna el nuevo faiss.Index, o None si no hay vectores.
    """
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
    index.add(vectors)

    faiss.write_index(index, str(paths["index"]))

    logger.info(
        f"Índice reconstruido | collection={collection} "
        f"| vectors={vectors.shape[0]} | dim={dimension}"
    )

    return index


# ======================================================
# VACUUM / COMPACTACIÓN
# ======================================================


def vacuum_collection(collection: str) -> dict[str, int]:
    """
    Compacta una colección eliminando huecos entre vectores y metadata.

    Re-serializa todo desde cero y reconstruye el índice garantizando
    coherencia total entre los tres artefactos.

    Retorna:
        {
            "before":  int,  # chunks antes del vacuum
            "after":   int,  # chunks después
            "removed": int,  # huecos eliminados
        }
    """
    metadata: list[dict[str, Any]]
    vectors: np.ndarray | None
    metadata, vectors = _load_raw(collection)

    before: int = len(metadata)

    if not metadata or vectors is None:
        logger.info(f"Vacuum: nada que compactar | collection={collection}")
        return {"before": before, "after": before, "removed": 0}

    # Filtra entradas cuyo vector esté fuera de rango
    valid_indices: list[int] = [i for i in range(len(metadata)) if i < len(vectors)]
    clean_metadata: list[dict[str, Any]] = [metadata[i] for i in valid_indices]
    clean_vectors: np.ndarray = vectors[valid_indices].astype("float32")

    after: int = len(clean_metadata)
    removed: int = before - after

    if removed == 0 and np.array_equal(vectors, clean_vectors):
        logger.info(f"Vacuum: colección ya consistente | collection={collection}")
        return {"before": before, "after": after, "removed": 0}

    dimension: int = clean_vectors.shape[1]
    index: faiss.Index = faiss.IndexFlatIP(dimension)
    index.add(clean_vectors)

    _save_raw(collection, clean_metadata, clean_vectors, index)

    logger.info(
        f"Vacuum completado | collection={collection} "
        f"| before={before} | after={after} | removed={removed}"
    )

    return {"before": before, "after": after, "removed": removed}


# ======================================================
# ELIMINACIÓN POR FUENTE
# ======================================================


def delete_by_source(
    collection: str,
    source: str,
    *,
    rebuild: bool = True,
) -> int:
    """
    Elimina todos los chunks asociados a una fuente concreta.

    Args:
        collection: Ruta relativa, p. ej. "sociologia/espectaculo".
        source:     Valor exacto del campo "source" en metadata
                    (nombre de archivo o URL completa).
        rebuild:    Si True, ejecuta vacuum tras eliminar.

    Retorna el número de chunks eliminados.
    """
    return delete_by_sources(collection, [source], rebuild=rebuild)


def delete_by_sources(
    collection: str,
    sources: Sequence[str],
    *,
    rebuild: bool = True,
) -> int:
    """
    Eliminación en lote de múltiples fuentes en una sola escritura atómica.

    Args:
        collection: Ruta relativa de la colección.
        sources:    Lista de valores "source" a eliminar.
        rebuild:    Si True, ejecuta vacuum tras eliminar.

    Retorna el número total de chunks eliminados.
    """
    source_set: set[str] = set(sources)

    if not source_set:
        logger.warning("delete_by_sources: lista de fuentes vacía.")
        return 0

    metadata: list[dict[str, Any]]
    vectors: np.ndarray | None
    metadata, vectors = _load_raw(collection)

    if not metadata:
        logger.warning(
            f"delete_by_sources: colección vacía o inexistente "
            f"| collection={collection}"
        )
        return 0

    keep_indices: list[int] = [
        i for i, m in enumerate(metadata) if m.get("source") not in source_set
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

    clean_metadata: list[dict[str, Any]] = [metadata[i] for i in keep_indices]

    if vectors is not None and len(vectors) > 0:
        valid_keep: list[int] = [i for i in keep_indices if i < len(vectors)]
        clean_vectors: np.ndarray = vectors[valid_keep].astype("float32")
    else:
        clean_vectors = np.empty((0,), dtype="float32")

    if clean_metadata and clean_vectors.ndim == 2 and clean_vectors.shape[0] > 0:
        dimension: int = clean_vectors.shape[1]
        index: faiss.Index = faiss.IndexFlatIP(dimension)
        index.add(clean_vectors)
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
    """
    Elimina todos los artefactos de una colección (índice, metadata,
    vectores). El directorio base se conserva para re-ingestión futura.
    """
    _clear_collection_files(collection)
    logger.info(f"Colección limpiada | collection={collection}")


# ======================================================
# ALIAS SEMÁNTICOS PARA URLs
# ======================================================


def delete_url(
    collection: str,
    url: str,
    *,
    rebuild: bool = True,
) -> int:
    """Alias de delete_by_source orientado a fuentes de tipo URL."""
    return delete_by_source(collection, url, rebuild=rebuild)


def delete_urls(
    collection: str,
    urls: Sequence[str],
    *,
    rebuild: bool = True,
) -> int:
    """Alias de delete_by_sources orientado a lotes de URLs."""
    return delete_by_sources(collection, urls, rebuild=rebuild)


# ======================================================
# INTROSPECCIÓN
# ======================================================


def list_sources(collection: str) -> list[dict[str, Any]]:
    """
    Devuelve un resumen de las fuentes indexadas en la colección.

    Retorna lista de dicts:
        [{"source": str, "source_type": str, "chunks": int}, ...]
    """
    metadata: list[dict[str, Any]]
    metadata, _ = _load_raw(collection)

    seen: dict[str, dict[str, Any]] = {}

    for entry in metadata:
        src: str = entry.get("source", "<desconocido>")
        if src not in seen:
            seen[src] = {
                "source": src,
                "source_type": entry.get("source_type", "file"),
                "chunks": 0,
            }
        seen[src]["chunks"] += 1

    return sorted(seen.values(), key=lambda x: x["source"])
