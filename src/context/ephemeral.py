"""
Almacén de colecciones efímeras: archivos subidos "solo para esta
conversación" (a diferencia de un archivo que se adjunta a una colección
persistida vía POST /api/v1/files con attach_to_collection=True).

Por qué existe un módulo separado de ContextManager:
  ContextManager (context/manager.py) carga/lista colecciones que viven
  en disco bajo settings.vector_store_path_for_backend. Una colección
  efímera nunca se escribe a disco: vive solo en memoria del proceso,
  indexada por conversation_id (un id opaco que el cliente genera una
  vez por conversación y reenvía en cada request relevante). No
  sobrevive a un restart del servidor -- es justo lo que se quiere para
  contexto de un solo uso.

Por qué no se puede hacer en el navegador:
  Chunking + embeddings requieren sentence_transformers/FAISS corriendo
  en el proceso Python del backend, no algo viable en un navegador. El
  cliente solo sube el archivo; todo el cómputo pasa por el mismo
  pipeline que usa ingest/ingest.py para colecciones persistidas
  (chunk_text, encode_chunks, build_metadata) -- la única diferencia es
  que acá nunca se llama a save_collection().

Borrado a nivel de archivo:
  Cada chunk guarda su file_id (ver ChunkMetadata.file_id). Borrar un
  archivo puntual filtra metadata/vectors por ese file_id y reconstruye
  el índice FAISS con lo que queda. Para el tamaño típico de una
  colección efímera (unos pocos archivos por conversación) reconstruir
  el índice es trivialmente barato -- no hace falta el soporte de
  borrado nativo de FAISS (IndexIDMap + remove_ids), que además no
  todos los tipos de índice soportan.

Concurrencia:
  Pensado para un solo worker uvicorn (caso típico de este proyecto:
  local, single-user). Con --workers > 1 cada worker tendría su propio
  store en memoria y una conversación podría "perder" sus archivos si
  el load balancer la manda a otro worker en el siguiente request.
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
    ChunkMetadata,
    CollectionPaths,
    build_metadata,
    chunk_text,
    create_faiss_index,
    encode_chunks,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from faiss import Index as FaissIndex


# ======================================================
# TIPOS PÚBLICOS
# ======================================================


class EphemeralFileInfo(BaseModel):
    """Metadata de un archivo subido a una conversación (para respuestas de API)."""

    file_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime


# ======================================================
# ESTADO INTERNO
# ======================================================


@dataclass
class _ConversationStore:
    """Estado en memoria de los archivos efímeros de una conversación."""

    metadata: list[ChunkMetadata] = field(default_factory=list)
    vectors: np.ndarray | None = None
    index: FaissIndex | None = None
    files: dict[str, EphemeralFileInfo] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


def _placeholder_paths(conversation_id: str) -> CollectionPaths:
    """
    CollectionPaths "de mentira" para satisfacer el TypedDict LoadedCollection.

    retrieve() (src/retrieval/search.py) nunca lee la clave 'paths' --
    solo index, metadata y collection_name -- así que esto nunca toca el
    disco. Se arma solo para no tener que aflojar el tipo de
    LoadedCollection ni introducir un TypedDict paralelo.
    """
    base = settings.vector_store_path_for_backend / "_ephemeral" / conversation_id
    return CollectionPaths(
        vector_path=base,
        index_file=base / "index.faiss",
        metadata_file=base / "metadata.pkl",
        vectors_file=base / "vectors.npy",
    )


def _rebuild_index(vectors: np.ndarray) -> FaissIndex:
    dimension = vectors.shape[1]
    index = create_faiss_index(dimension)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))  # pyright: ignore[reportCallIssue]
    return index


# ======================================================
# STORE
# ======================================================


class EphemeralStore:
    """Colecciones efímeras en memoria, una por conversation_id."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationStore] = {}

    # -------------------------------------------------
    # ESCRITURA
    # -------------------------------------------------

    def add_file(
        self,
        conversation_id: str,
        filename: str,
        source_type: str,
        pages: list[tuple[int, str]],
    ) -> EphemeralFileInfo:
        """
        Chunkea, embebe e indexa un archivo dentro de la colección
        efímera de `conversation_id`. Si la conversación ya tenía
        archivos, este se agrega a los existentes (no los reemplaza).
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
            raise ValueError(f"No se pudo extraer contenido de chunkeable de '{filename}'.")

        embeddings = encode_chunks(new_chunks)
        store = self._conversations.setdefault(conversation_id, _ConversationStore())

        store.vectors = (
            np.vstack([store.vectors, embeddings])
            if store.vectors is not None
            else embeddings
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
            f"[ephemeral] archivo agregado | conversation={conversation_id} "
            f"| file={filename} | file_id={file_id} | chunks={len(new_chunks)}"
        )
        return info

    # -------------------------------------------------
    # BORRADO
    # -------------------------------------------------

    def remove_file(self, conversation_id: str, file_id: str) -> bool:
        """
        Borra un archivo puntual de la colección efímera de una
        conversación, reconstruyendo metadata/vectors/index sin sus
        chunks. Si era el último archivo, borra la conversación entera.

        Devuelve False si conversation_id o file_id no existen (idempotente:
        el router lo traduce a 404, no a un 500).
        """
        store = self._conversations.get(conversation_id)
        if store is None or file_id not in store.files:
            return False

        keep = [i for i, m in enumerate(store.metadata) if m.file_id != file_id]

        if not keep:
            del self._conversations[conversation_id]
            logger.info(
                f"[ephemeral] se borró el último archivo -> conversación "
                f"limpiada | conversation={conversation_id}"
            )
            return True

        store.metadata = [store.metadata[i] for i in keep]
        assert store.vectors is not None  # invariante: si hay metadata, hay vectors
        store.vectors = store.vectors[keep]
        store.index = _rebuild_index(store.vectors)
        del store.files[file_id]
        store.last_used = datetime.now(UTC)

        logger.info(
            f"[ephemeral] archivo borrado | conversation={conversation_id} "
            f"| file_id={file_id} | chunks_restantes={len(store.metadata)}"
        )
        return True

    def remove_conversation(self, conversation_id: str) -> bool:
        """Borra toda la colección efímera de una conversación (ej: al cerrarla en la UI)."""
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        if existed:
            logger.info(f"[ephemeral] conversación eliminada | conversation={conversation_id}")
        return existed

    # -------------------------------------------------
    # LECTURA
    # -------------------------------------------------

    def list_files(self, conversation_id: str) -> list[EphemeralFileInfo]:
        store = self._conversations.get(conversation_id)
        return list(store.files.values()) if store is not None else []

    def get_collection(self, conversation_id: str) -> LoadedCollection | None:
        """Colección lista para retrieve(), o None si la conversación no tiene archivos."""
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
    # MANTENIMIENTO
    # -------------------------------------------------

    def sweep_expired(self, ttl: timedelta) -> int:
        """
        Borra conversaciones sin actividad hace más de `ttl`. Sin esto,
        las colecciones efímeras crecen sin límite mientras viva el
        proceso -- pensado para llamarse periódicamente (ver lifespan
        en src/api/app.py).
        """
        cutoff = datetime.now(UTC) - ttl
        expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

        for cid in expired:
            del self._conversations[cid]

        if expired:
            logger.info(f"[ephemeral] limpieza TTL | conversaciones eliminadas={len(expired)}")

        return len(expired)
