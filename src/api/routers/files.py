"""
Router "files" -- subir archivos para usarlos como contexto RAG.

Dos destinos posibles para un archivo subido (toggle attach_to_collection):
  - True:  se procesa con el mismo pipeline que ingest/ingest.py y se
           persiste a disco en `collection` -- queda disponible para
           cualquier conversación futura, igual que si se hubiera
           corrido el CLI de ingest.
  - False (default): se procesa igual, pero el resultado vive solo en
           memoria, scopeado a `conversation_id` (ver
           src/context/ephemeral.py). Nunca toca disco.

Borrado:
  DELETE /files/{conversation_id}/{file_id} borra un archivo puntual de
  la colección efímera de una conversación (y solo de ahí -- no hay
  forma de "borrar un archivo" de una colección persistida desde acá;
  eso es responsabilidad del CLI/ingest, fuera de alcance de este router).

  DELETE /files/{conversation_id} borra toda la colección efímera de la
  conversación de una vez -- pensado para cuando el usuario cierra o
  elimina la conversación completa en la UI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from src.api.deps import get_ephemeral_store
from src.api.schemas.files import (
    DeleteResponse,
    EphemeralFilesResponse,
    FileUploadResponse,
)
from src.context.ephemeral import EphemeralStore
from src.ingest.core import (
    ChunkMetadata,
    RawCollection,
    build_metadata,
    chunk_text,
    encode_chunks,
    load_collection,
    save_collection,
)
from src.ingest.ingest import read_file
from src.utils.logger import logger

router = APIRouter(prefix="/files", tags=["files"])

_SUPPORTED_SUFFIXES = {".pdf", ".html", ".txt"}


def _extract_pages(filename: str, raw_bytes: bytes) -> list[tuple[int, str]]:
    """
    Reutiliza el parsing de PDF/HTML/TXT de ingest/ingest.py, que espera
    un Path en disco. El archivo temporal se borra apenas termina de leerse,
    nunca queda persistido -- eso lo decide attach_to_collection, no esto.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado: '{suffix}'. Soportados: {sorted(_SUPPORTED_SUFFIXES)}.",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        pages = read_file(Path(tmp.name))

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo extraer contenido de '{filename}'.",
        )

    return pages


def _attach_to_persisted_collection(
    collection: str,
    filename: str,
    pages: list[tuple[int, str]],
    file_id: str,
) -> int:
    """Ingesta un archivo a una colección persistida, igual que ingest/ingest.py."""
    collection_data: RawCollection = load_collection(collection)

    new_chunks: list[str] = []
    new_metadata: list[ChunkMetadata] = []

    for page_number, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            new_chunks.append(chunk)
            new_metadata.append(
                build_metadata(
                    source=filename,
                    source_type="file",
                    page=page_number,
                    chunk=chunk,
                    chunk_index=i,
                    collection=collection,
                    file_id=file_id,
                )
            )

    if not new_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No se generaron chunks para '{filename}'.",
        )

    embeddings = encode_chunks(new_chunks)
    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )
    return len(new_chunks)


@router.post("", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile,
    conversation_id: str = Form(...),
    attach_to_collection: bool = Form(False),
    collection: str | None = Form(default=None),
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> FileUploadResponse:
    """
    Sube un archivo (.pdf, .html, .txt) y lo procesa con el mismo
    pipeline de chunking/embeddings que usa el CLI de ingest.

    attach_to_collection=True requiere `collection` (formato
    "namespace/coleccion", igual que el CLI) y persiste el resultado a
    disco -- disponible para cualquier conversación futura.

    attach_to_collection=False (default) lo guarda solo en memoria,
    scopeado a `conversation_id` -- se pierde si el servidor se reinicia
    o si nadie lo usa por más de EPHEMERAL_TTL (ver src/api/deps.py).
    """
    if attach_to_collection and not collection:
        raise HTTPException(
            status_code=422,
            detail="'collection' es requerido cuando attach_to_collection=True.",
        )

    filename = file.filename or "archivo_sin_nombre"
    raw_bytes = await file.read()
    pages = _extract_pages(filename, raw_bytes)

    if attach_to_collection:
        assert collection is not None  # ya validado arriba
        # file_id acá es solo informativo (no hay borrado por-archivo en
        # colecciones persistidas, eso es responsabilidad del CLI de ingest).
        file_id = f"persisted-{abs(hash((collection, filename))) & 0xFFFFFF:06x}"
        chunk_count = _attach_to_persisted_collection(collection, filename, pages, file_id)
        logger.info(
            f"[api] archivo adjuntado a colección persistida | collection={collection} "
            f"| file={filename} | chunks={chunk_count}"
        )
        return FileUploadResponse(
            conversation_id=conversation_id,
            file_id=file_id,
            filename=filename,
            chunk_count=chunk_count,
            attached_to_collection=collection,
        )

    info = ephemeral_store.add_file(
        conversation_id=conversation_id,
        filename=filename,
        source_type="file",
        pages=pages,
    )
    return FileUploadResponse(
        conversation_id=conversation_id,
        file_id=info.file_id,
        filename=info.filename,
        chunk_count=info.chunk_count,
        attached_to_collection=None,
    )


@router.get("/{conversation_id}", response_model=EphemeralFilesResponse)
async def list_ephemeral_files(
    conversation_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> EphemeralFilesResponse:
    """Lista los archivos efímeros adjuntos a una conversación."""
    return EphemeralFilesResponse(
        conversation_id=conversation_id,
        files=ephemeral_store.list_files(conversation_id),
    )


@router.delete("/{conversation_id}/{file_id}", response_model=DeleteResponse)
async def delete_ephemeral_file(
    conversation_id: str,
    file_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> DeleteResponse:
    """
    Borra un archivo puntual de la colección efímera de una conversación.

    404 si la conversación o el archivo no existen -- ya sea porque
    nunca se subió, porque ya se borró antes, o porque expiró por TTL.
    """
    deleted = ephemeral_store.remove_file(conversation_id, file_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el archivo '{file_id}' en la conversación '{conversation_id}'.",
        )
    return DeleteResponse(deleted=True)


@router.delete("/{conversation_id}", response_model=DeleteResponse)
async def delete_ephemeral_conversation(
    conversation_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> DeleteResponse:
    """Borra toda la colección efímera de una conversación (todos sus archivos a la vez)."""
    deleted = ephemeral_store.remove_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"La conversación '{conversation_id}' no tiene archivos efímeros.",
        )
    return DeleteResponse(deleted=True)
