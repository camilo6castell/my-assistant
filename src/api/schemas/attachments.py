"""
Schemas for ad-hoc "file attachments" -- see src/context/attachments.py
and src/api/routers/attachments.py.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.context.attachments import AttachmentInfo

__all__ = ["AttachmentInfo", "AttachmentsResponse"]


class AttachmentsResponse(BaseModel):
    """Response for GET /api/v1/attachments/{conversation_id}."""

    files: list[AttachmentInfo]
