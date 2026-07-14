"""
Router "attachments" -- archivos ad-hoc adjuntos a la PRÓXIMA query de
una conversación (ver src/context/attachments.py). No indexan nada, no
generan respuestas por sí mismos: /api/v1/query y /api/v1/query/agent
(src/api/routers/chat.py) son los que los leen, los inyectan en el
prompt, y los consumen (los borran) después de procesar la query.
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
    Sube un archivo de texto plano para inyectarlo crudo en la próxima
    query de esta conversación (ver inject_attachments() en
    src/prompts/builder.py). `conversation_id` va como campo de
    multipart/form-data, igual que en POST /files (ver
    src/api/routers/files.py).
    """
    filename = file.filename or "archivo_sin_nombre"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{suffix}'. Soportados: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande ({len(raw)} bytes). Límite: {MAX_FILE_BYTES} bytes.",
        )

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="El archivo no es texto plano UTF-8 válido."
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
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return {"deleted": True}


@router.delete("/{conversation_id}")
async def delete_all_attachments(
    conversation_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> dict[str, bool]:
    store.remove_conversation(conversation_id)
    return {"deleted": True}
