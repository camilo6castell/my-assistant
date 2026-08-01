"""
Router "files" -- upload files to use as RAG context.

Two possible destinations for an uploaded file (toggle attach_to_collection):
  - True:  processed with the same pipeline as ingest/ingest.py and
           persisted to disk in `collection` -- available for any future
           conversation, just as if the ingest CLI had been run.
  - False (default): processed the same way, but the result lives only
           in memory, scoped to `conversation_id` (see
           src/context/ephemeral.py). Never touches disk.

Deletion:
  DELETE /files/{conversation_id}/{file_id} deletes a single file from
  the ephemeral collection of a conversation (and only from there -- there
  is no way to "delete a file" from a persisted collection from here;
  that is the ingest CLI's responsibility, out of scope for this router).

  DELETE /files/{conversation_id} deletes the entire ephemeral collection
  for the conversation at once -- intended for when the user closes or
  deletes the full conversation in the UI.
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
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.ingest.ingest import read_file
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
from src.utils.logger import logger

router = APIRouter(prefix="/files", tags=["files"])

_SUPPORTED_SUFFIXES = {".pdf", ".html", ".txt"}


def _validate_collection_name(collection: str) -> None:
    """
    Persisted collections live at
    <vector_store_path_for_backend>/<namespace>/<collection>/
    (see get_collection_paths), mirroring the ingest CLI's
    "<category>/<collection>" naming. Enforce exactly two non-empty
    segments: anything else either escapes the vector store root (path
    traversal -- `collection` is used directly as a relative path) or
    persists a layout the UI can never list (ContextManager.list_all
    only traverses two levels deep).
    """
    if "\\" in collection:
        raise HTTPException(
            status_code=422,
            detail="Collection must use forward slashes: 'namespace/collection'.",
        )

    parts = collection.split("/")
    if len(parts) != 2 or not all(parts):
        raise HTTPException(
            status_code=422,
            detail="Collection must be in 'namespace/collection' format "
            "(e.g. 'books/novels').",
        )

    if any(segment in {".", ".."} for segment in parts):
        raise HTTPException(
            status_code=422,
            detail="Invalid collection name: '.' and '..' are not allowed.",
        )


def _extract_pages(filename: str, raw_bytes: bytes) -> list[tuple[int, str]]:
    """
    Reuse the PDF/HTML/TXT parsing from ingest/ingest.py, which expects
    a Path on disk. The temporary file is deleted as soon as reading
    finishes, never persisted -- that is decided by attach_to_collection, not here.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: '{suffix}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}.",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        pages = read_file(Path(tmp.name))

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract content from '{filename}'.",
        )

    return pages


def _attach_to_persisted_collection(
    collection: str,
    filename: str,
    pages: list[tuple[int, str]],
    file_id: str,
) -> int:
    """Ingest a file into a persisted collection, same as ingest/ingest.py."""
    collection_data: RawCollection = load_collection(collection)

    # Same dedup as ingest/ingest.py: skip files already indexed in this
    # collection so re-uploads never duplicate embeddings.
    existing_sources: set[str] = {m.source for m in collection_data["metadata"]}
    if filename in existing_sources:
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' is already indexed in collection '{collection}'.",
        )

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
            detail=f"No chunks were generated for '{filename}'.",
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
    Upload a file (.pdf, .html, .txt) and process it with the same
    chunking/embeddings pipeline used by the ingest CLI.

    attach_to_collection=True requires `collection` (format
    "namespace/collection", same as the CLI) and persists the result to
    disk -- available for any future conversation.

    attach_to_collection=False (default) stores it in memory only,
    scoped to `conversation_id` -- lost if the server restarts or if
    nobody uses it for more than EPHEMERAL_TTL (see src/api/deps.py).
    """
    if attach_to_collection and not collection:
        raise HTTPException(
            status_code=422,
            detail="'collection' is required when attach_to_collection=True.",
        )

    if collection:
        _validate_collection_name(collection)

    filename = file.filename or "unnamed_file"
    raw_bytes = await file.read()
    pages = _extract_pages(filename, raw_bytes)

    if attach_to_collection:
        assert collection is not None  # already validated above
        # file_id here is informational only (there is no per-file deletion in
        # persisted collections; that is the responsibility of the ingest CLI).
        file_id = f"persisted-{abs(hash((collection, filename))) & 0xFFFFFF:06x}"
        chunk_count = _attach_to_persisted_collection(collection, filename, pages, file_id)
        logger.info(
            f"[api] file attached to persisted collection | collection={collection} "
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
    """List ephemeral files attached to a conversation."""
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
    Delete a specific file from the ephemeral collection of a conversation.

    Returns 404 if the conversation or file do not exist -- either because
    it was never uploaded, was already deleted, or expired via TTL.
    """
    deleted = ephemeral_store.remove_file(conversation_id, file_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found in conversation '{conversation_id}'.",
        )
    return DeleteResponse(deleted=True)


@router.delete("/{conversation_id}", response_model=DeleteResponse)
async def delete_ephemeral_conversation(
    conversation_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> DeleteResponse:
    """Delete the entire ephemeral collection for a conversation (all files at once)."""
    deleted = ephemeral_store.remove_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' has no ephemeral files.",
        )
    return DeleteResponse(deleted=True)
