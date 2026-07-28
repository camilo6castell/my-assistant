"""
Store for ephemeral collections: files uploaded "for this conversation
only" (as opposed to a file attached to a persisted collection via
POST /api/v1/files with attach_to_collection=True).

Why a separate module from ContextManager:
  ContextManager (context/manager.py) loads/lists collections that live
  on disk under settings.vector_store_path_for_backend. An ephemeral
  collection is never written to disk: it lives only in process memory,
  indexed by conversation_id (an opaque ID that the client generates
  once per conversation and resends with each relevant request). It does
  not survive a server restart -- which is exactly what you want for
  single-use context.

Why it cannot be done in the browser:
  Chunking + embeddings require sentence_transformers/FAISS running in
  the backend Python process, not something viable in a browser. The
  client only uploads the file; all computation goes through the same
  pipeline used by ingest/ingest.py for persisted collections
  (chunk_text, encode_chunks, build_metadata) -- the only difference is
  that save_collection() is never called here.

File-level deletion:
  Each chunk stores its file_id (see ChunkMetadata.file_id). Deleting a
  specific file filters metadata/vectors by that file_id and rebuilds
  the FAISS index with what remains. For the typical size of an
  ephemeral collection (a few files per conversation) rebuilding the
  index is trivially cheap -- there is no need for FAISS native delete
  support (IndexIDMap + remove_ids), which also is not supported by all
  index types.

Concurrency:
  Designed for a single uvicorn worker (typical case for this project:
  local, single-user). With --workers > 1 each worker would have its
  own in-memory store and a conversation could "lose" its files if the
  load balancer routes it to another worker on the next request.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from src.config.settings import settings
from src.context.manager import LoadedCollection
from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.storage.faiss_store import (
    ChunkMetadata,
    CollectionPaths,
    rebuild_index_from_vectors,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from faiss import Index as FaissIndex


# ======================================================
# PUBLIC TYPES
# ======================================================


class EphemeralFileInfo(BaseModel):
    """Metadata for a file uploaded to a conversation (for API responses)."""

    file_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime


# ======================================================
# INTERNAL STATE
# ======================================================


@dataclass
class _ConversationStore:
    """In-memory state for ephemeral files in a conversation."""

    metadata: list[ChunkMetadata] = field(default_factory=list)
    vectors: np.ndarray | None = None
    index: FaissIndex | None = None
    files: dict[str, EphemeralFileInfo] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


def _placeholder_paths(conversation_id: str) -> CollectionPaths:
    """
    Fake CollectionPaths to satisfy the LoadedCollection TypedDict.

    retrieve() (src/retrieval/search.py) never reads the 'paths' key --
    only index, metadata, and collection_name -- so this never touches
    disk. It exists only to avoid loosening the LoadedCollection type or
    introducing a parallel TypedDict.
    """
    base = settings.vector_store_path_for_backend / "_ephemeral" / conversation_id
    return CollectionPaths(
        vector_path=base,
        index_file=base / "index.faiss",
        metadata_file=base / "metadata.pkl",
        vectors_file=base / "vectors.npy",
    )


def _rebuild_index(vectors: np.ndarray) -> FaissIndex:
    return rebuild_index_from_vectors(vectors)


# ======================================================
# STORE
# ======================================================


class EphemeralStore:
    """In-memory ephemeral collections, one per conversation_id."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationStore] = {}

    # -------------------------------------------------
    # WRITING
    # -------------------------------------------------

    def add_file(
        self,
        conversation_id: str,
        filename: str,
        source_type: str,
        pages: list[tuple[int, str]],
    ) -> EphemeralFileInfo:
        """
        Chunks, embeds, and indexes a file within the ephemeral
        collection of `conversation_id`. If the conversation already has
        files, this one is added to the existing ones (does not replace).
        """
        file_id = uuid.uuid4().hex[:12]
        new_chunks: list[str] = []
        new_metadata: list[ChunkMetadata] = []

        for page_number, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                new_chunks.append(chunk)
                new_metadata.append(
                    build_metadata(
                        source=filename,
                        source_type=source_type,
                        page=page_number,
                        chunk=chunk,
                        chunk_index=i,
                        collection=f"_ephemeral/{conversation_id}",
                        file_id=file_id,
                    )
                )

        if not new_chunks:
            raise ValueError(f"Could not extract chunkable content from '{filename}'.")

        embeddings = encode_chunks(new_chunks)
        store = self._conversations.setdefault(conversation_id, _ConversationStore())

        store.vectors = (
            np.vstack([store.vectors, embeddings]) if store.vectors is not None else embeddings
        )
        store.metadata.extend(new_metadata)
        store.index = _rebuild_index(store.vectors)

        info = EphemeralFileInfo(
            file_id=file_id,
            filename=filename,
            chunk_count=len(new_chunks),
            uploaded_at=datetime.now(UTC),
        )
        store.files[file_id] = info
        store.last_used = datetime.now(UTC)

        logger.info(
            f"[ephemeral] file added | conversation={conversation_id} "
            f"| file={filename} | file_id={file_id} | chunks={len(new_chunks)}"
        )
        return info

    # -------------------------------------------------
    # DELETION
    # -------------------------------------------------

    def remove_file(self, conversation_id: str, file_id: str) -> bool:
        """
        Deletes a specific file from the ephemeral collection of a
        conversation, rebuilding metadata/vectors/index without its
        chunks. If it was the last file, deletes the entire conversation.

        Returns False if conversation_id or file_id do not exist (idempotent:
        the router translates it to 404, not 500).
        """
        store = self._conversations.get(conversation_id)
        if store is None or file_id not in store.files:
            return False

        keep = [i for i, m in enumerate(store.metadata) if m.file_id != file_id]

        if not keep:
            del self._conversations[conversation_id]
            logger.info(
                f"[ephemeral] last file deleted -> conversation "
                f"cleaned | conversation={conversation_id}"
            )
            return True

        store.metadata = [store.metadata[i] for i in keep]
        assert store.vectors is not None  # invariant: metadata implies vectors
        store.vectors = store.vectors[keep]
        store.index = _rebuild_index(store.vectors)
        del store.files[file_id]
        store.last_used = datetime.now(UTC)

        logger.info(
            f"[ephemeral] file deleted | conversation={conversation_id} "
            f"| file_id={file_id} | remaining_chunks={len(store.metadata)}"
        )
        return True

    def remove_conversation(self, conversation_id: str) -> bool:
        """Deletes the ephemeral collection of a conversation (e.g. closing it in the UI)."""
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        if existed:
            logger.info(f"[ephemeral] conversation deleted | conversation={conversation_id}")
        return existed

    # -------------------------------------------------
    # READING
    # -------------------------------------------------

    def list_files(self, conversation_id: str) -> list[EphemeralFileInfo]:
        store = self._conversations.get(conversation_id)
        return list(store.files.values()) if store is not None else []

    def get_collection(self, conversation_id: str) -> LoadedCollection | None:
        """Collection ready for retrieve(), or None if the conversation has no files."""
        store = self._conversations.get(conversation_id)
        if store is None or store.index is None:
            return None

        store.last_used = datetime.now(UTC)
        return LoadedCollection(
            index=store.index,
            metadata=store.metadata,
            vectors=store.vectors,
            paths=_placeholder_paths(conversation_id),
            collection_name=f"_ephemeral/{conversation_id}",
        )

    # -------------------------------------------------
    # MAINTENANCE
    # -------------------------------------------------

    def sweep_expired(self, ttl: timedelta) -> int:
        """
        Deletes conversations inactive for more than `ttl`. Without this,
        ephemeral collections grow without bound while the process lives
        -- designed to be called periodically (see lifespan in
        src/api/app.py).
        """
        cutoff = datetime.now(UTC) - ttl
        expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

        for cid in expired:
            del self._conversations[cid]

        if expired:
            logger.info(f"[ephemeral] TTL sweep | conversations deleted={len(expired)}")

        return len(expired)
