"""
Schemas for the "files" domain -- uploading files, attaching them to a
persistent collection, or using them only within a conversation.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.context.ephemeral import EphemeralFileInfo

__all__ = ["EphemeralFileInfo", "FileUploadResponse", "EphemeralFilesResponse", "DeleteResponse"]


class FileUploadResponse(BaseModel):
    """
    Response for POST /api/v1/files.

    attached_to_collection: name of the persistent collection if the
    file was saved to disk (attach_to_collection=True in the form).
    None if the file remained only in the ephemeral collection of the
    conversation.
    """

    conversation_id: str
    file_id: str
    filename: str
    chunk_count: int
    attached_to_collection: str | None = None


class EphemeralFilesResponse(BaseModel):
    """Response for GET /api/v1/files/{conversation_id}."""

    conversation_id: str
    files: list[EphemeralFileInfo]


class DeleteResponse(BaseModel):
    """Response for DELETE requests on this router."""

    deleted: bool
