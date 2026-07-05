"""
Schemas del dominio "chat" -- las funciones regulares de la aplicación:
hacer una pregunta, listar colecciones disponibles.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class GenerationOptions(BaseModel):
    """
    Overrides opcionales de generación para una llamada puntual a
    POST /query o POST /query/agent.

    temperature/max_tokens/think_mode: campos con nombre porque son
    conceptos comunes a la mayoría de providers/modelos razonadores
    (pensados para controles concretos en la UI: slider, input numérico,
    switch). Cada uno en None significa "usar el default de settings/.env
    para el provider activo".

    extra: escape hatch genérico para cualquier parámetro que el backend
    NO modela explícitamente (top_p, presence_penalty, un flag propio de
    un provider nuevo...). Se envía tal cual al provider vía extra_body
    del cliente OpenAI -- sin validar su contenido, porque por definición
    puede ser cualquier cosa que un provider específico entienda. Por eso
    requiere que el provider declare "extra" en su `supports` (ver
    ProviderConfig en src/llm/providers.py): es un opt-in explícito, no
    "cualquier JSON pasa a cualquier servidor".

    Ninguno de estos valores muta Settings ni ProviderConfig -- son
    parámetros por-request, no estado del servidor (ver docstring de
    src/llm/generate.py para el porqué). El servidor valida cada campo
    contra ProviderConfig.supports antes de usarlo y devuelve 400 si el
    provider activo no lo soporta.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    think_mode: bool | None = None
    extra: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    """
    Payload para POST /query y POST /query/agent.

    collections:    tokens con la misma sintaxis que el CLI, ej.
                    ["sociologia", "psicoanalisis/Freud_Suenos"].
                    Puede venir vacío SI conversation_id apunta a una
                    conversación con archivos efímeros subidos (ver
                    POST /api/v1/files) -- en ese caso el retrieval usa
                    solo esos archivos. El router devuelve 422 si no hay
                    ninguna fuente de contexto (ni colecciones ni
                    archivos efímeros).
    mode:           "SOFT" (default) | "HARD"
    chat_history:   turnos previos de conversación. Cada turno es un dict
                    {"user": "...", "assistant": "..."}. El cliente es
                    responsable de mantener y reenviar el historial -- la
                    API es stateless por diseño.
    conversation_id: id opaco que el cliente genera una vez por
                    conversación (ej. crypto.randomUUID() en el
                    frontend). Solo hace falta si esa conversación tiene
                    archivos efímeros adjuntos; si no, se puede omitir.
    generation:     overrides opcionales de temperatura/tokens/think mode.
    """

    question: str = Field(..., min_length=1)
    collections: list[str] = Field(default_factory=list)
    mode: str = Field(default="SOFT", pattern="^(SOFT|HARD)$")
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str | None = None
    generation: GenerationOptions | None = None

    @model_validator(mode="after")
    def _require_some_context_source(self) -> "QueryRequest":
        if not self.collections and not self.conversation_id:
            raise ValueError(
                "Se requiere al menos una colección en 'collections' o un "
                "'conversation_id' con archivos efímeros adjuntos."
            )
        return self


class QueryResponse(BaseModel):
    """
    Respuesta de POST /query y POST /query/agent.

    reformulated: solo relevante en /query/agent. True si el grafo
                  reformuló la query antes de generar.
    """

    answer: str
    confidence: float
    collections_used: list[str]
    reformulated: bool = False


class CollectionsResponse(BaseModel):
    """Respuesta de GET /api/v1/collections."""

    collections: list[str]
