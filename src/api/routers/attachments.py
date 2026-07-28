"""
Router "attachments" -- ad-hoc files attached to the NEXT query of a
conversation (see src/context/attachments.py). They are not indexed, nor
do they generate responses on their own: /api/v1/query and /api/v1/query/agent
(src/api/routers/chat.py) read them, inject them into the prompt, and
consume (delete) them after processing the query.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from src.api.deps import get_attachment_store
from src.api.schemas.attachments import AttachmentInfo, AttachmentsResponse
from src.context.attachments import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, AttachmentStore

router = APIRouter(prefix="/attachments", tags=["attachments"])


@router.post("", response_model=AttachmentInfo)
async def upload_attachment(
    file: UploadFile,
    conversation_id: str = Form(...),
    store: AttachmentStore = Depends(get_attachment_store),
) -> AttachmentInfo:
    """
    Upload a plain-text file to be injected raw into the next query of
    this conversation (see inject_attachments() in
    src/prompts/builder.py). `conversation_id` is sent as a
    multipart/form-data field, same as POST /files (see
    src/api/routers/files.py).
    """
    filename = file.filename or "unnamed_file"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes). Limit: {MAX_FILE_BYTES} bytes.",
        )

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="The file is not valid plain-text UTF-8."
        ) from None

    return store.add_file(conversation_id, filename, content)


@router.get("/{conversation_id}", response_model=AttachmentsResponse)
async def list_attachments(
    conversation_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> AttachmentsResponse:
    return AttachmentsResponse(files=store.list_files(conversation_id))


@router.delete("/{conversation_id}/{file_id}")
async def delete_attachment(
    conversation_id: str,
    file_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> dict[str, bool]:
    deleted = store.remove_file(conversation_id, file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found.")
    return {"deleted": True}


@router.delete("/{conversation_id}")
async def delete_all_attachments(
    conversation_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> dict[str, bool]:
    store.remove_conversation(conversation_id)
    return {"deleted": True}
