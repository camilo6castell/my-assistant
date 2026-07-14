"""
Schemas de "archivos adjuntos" ad-hoc -- ver src/context/attachments.py
y src/api/routers/attachments.py.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.context.attachments import AttachmentInfo

__all__ = ["AttachmentInfo", "AttachmentsResponse"]


class AttachmentsResponse(BaseModel):
    """Respuesta de GET /api/v1/attachments/{conversation_id}."""

    files: list[AttachmentInfo]
