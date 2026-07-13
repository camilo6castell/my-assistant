"""
Schemas del dominio "task" -- el modo de tareas de desarrollo sin RAG
(ver src/api/routers/task.py y src/context/task_files.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.schemas.chat import GenerationOptions
from src.context.task_files import TaskFileInfo

__all__ = ["TaskFileInfo", "TaskFilesResponse", "TaskRequest", "TaskResponse"]


class TaskFilesResponse(BaseModel):
    """Respuesta de GET /api/v1/task/files/{conversation_id}."""

    files: list[TaskFileInfo]


class TaskRequest(BaseModel):
    """
    Payload de POST /api/v1/task/query.

    Sin `collections`, sin `web_search`, sin `mode` SOFT/HARD -- Task no
    tiene retrieval ni distingue modos de respuesta (ver
    build_task_system_prompt() en src/prompts/builder.py). Los archivos
    de contexto no se pasan acá: se leen del TaskFileStore usando
    `conversation_id` (mismo patrón que QueryRequest.conversation_id
    para colecciones efímeras).
    """

    question: str = Field(..., min_length=1)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str | None = None
    generation: GenerationOptions | None = None


class TaskResponse(BaseModel):
    """Respuesta de POST /api/v1/task/query."""

    answer: str
    files_used: list[str] = Field(default_factory=list)
