"""
Schemas del dominio "files" -- subir archivos, adjuntarlos a una
colección persistida o usarlos solo dentro de una conversación.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.context.ephemeral import EphemeralFileInfo

__all__ = ["EphemeralFileInfo", "FileUploadResponse", "EphemeralFilesResponse", "DeleteResponse"]


class FileUploadResponse(BaseModel):
    """
    Respuesta de POST /api/v1/files.

    attached_to_collection: nombre de la colección persistida si el
    archivo se guardó en disco (attach_to_collection=True en el form).
    None si el archivo quedó solo en la colección efímera de la
    conversación.
    """

    conversation_id: str
    file_id: str
    filename: str
    chunk_count: int
    attached_to_collection: str | None = None


class EphemeralFilesResponse(BaseModel):
    """Respuesta de GET /api/v1/files/{conversation_id}."""

    conversation_id: str
    files: list[EphemeralFileInfo]


class DeleteResponse(BaseModel):
    """Respuesta de los DELETE de este router."""

    deleted: bool
