"""
Ingest pipeline core: chunking and metadata building.

Persistence (load/save FAISS indexes, metadata, vectors) is delegated
to src/storage/faiss_store.py. Encoding is delegated to
src/nlp/embedders/encoder.py. This module focuses on:
  - Splitting text into chunks (chunk_text)
  - Building chunk metadata (build_metadata, ChunkMetadata)
  - Encoding chunks into embeddings (encode_chunks — thin wrapper)
"""

from __future__ import annotations

import numpy as np

from src.config.settings import settings
from src.nlp.embedders.encoder import get_encoder
from src.storage.faiss_store import (  # re-export for backward compatibility
    ChunkMetadata,
)
from src.utils.logger import logger

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
# EMBEDDINGS (thin wrapper)
# ======================================================


def encode_chunks(chunks: list[str]) -> np.ndarray:
    logger.info(
        f"Generating embeddings for {len(chunks)} chunks | "
        f"backend={settings.embedding_backend} | model={settings.embedding_model}"
    )
    return get_encoder().encode(chunks)


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
