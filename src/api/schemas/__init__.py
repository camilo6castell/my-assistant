"""
Schemas Pydantic de la API REST, organizados por dominio -- un archivo
por router (chat.py, files.py, config.py), igual que src/api/routers/.

Este __init__ re-exporta todo para que el resto del código pueda seguir
haciendo `from src.api.schemas import QueryRequest` sin que le importe
en qué submódulo vive cada schema.
"""

from src.api.schemas.chat import (
    CollectionsResponse,
    GenerationOptions,
    QueryRequest,
    QueryResponse,
)
from src.api.schemas.config import ProviderInfo, ProvidersResponse
from src.api.schemas.files import (
    DeleteResponse,
    EphemeralFileInfo,
    EphemeralFilesResponse,
    FileUploadResponse,
)

__all__ = [
    "CollectionsResponse",
    "GenerationOptions",
    "QueryRequest",
    "QueryResponse",
    "ProviderInfo",
    "ProvidersResponse",
    "DeleteResponse",
    "EphemeralFileInfo",
    "EphemeralFilesResponse",
    "FileUploadResponse",
]
