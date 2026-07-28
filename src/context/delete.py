"""
Granular deletion of documents, URLs, and sources from the vectorstore.
All FAISS persistence is delegated to src/storage/faiss_store.py.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from src.storage.faiss_store import (
    clear_collection_files,
    delete_by_sources,
    load_raw,
)
from src.utils.logger import logger

# ======================================================
# SEMANTIC ALIASES
# ======================================================


def delete_by_source(
    collection: str,
    source: str,
    *,
    rebuild: bool = True,
) -> int:
    return delete_by_sources(collection, [source], rebuild=rebuild)


def delete_urls(collection: str, urls: Sequence[str], *, rebuild: bool = True) -> int:
    return delete_by_sources(collection, urls, rebuild=rebuild)


def delete_url(collection: str, url: str, *, rebuild: bool = True) -> int:
    return delete_by_source(collection, url, rebuild=rebuild)


# ======================================================
# FULL CLEANUP
# ======================================================


def clear_collection(collection: str) -> None:
    clear_collection_files(collection)
    logger.info(f"Collection cleared | collection={collection}")


# ======================================================
# INSPECTION
# ======================================================


class SourceSummary(BaseModel):
    """Summary of a source indexed in a collection."""

    model_config = ConfigDict(frozen=True)

    source: str
    source_type: str
    chunks: int


def list_sources(collection: str) -> list[SourceSummary]:
    """Returns a summary of sources indexed in a collection."""
    metadata, _ = load_raw(collection)

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
