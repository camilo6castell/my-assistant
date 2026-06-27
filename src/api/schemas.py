"""
Schemas Pydantic para la API REST del RAG.

Separados de los modelos de dominio (context/models.py) porque los
contratos de la API son independientes de la representación interna.
Si el dominio cambia internamente, los schemas de la API no cambian
(y viceversa) — esto protege a los clientes externos (n8n, etc.).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ======================================================
# REQUEST
# ======================================================


class QueryRequest(BaseModel):
    """
    Payload para POST /query y POST /query/agent.

    collections: lista de tokens con la misma sintaxis que el CLI
                 ["sociologia", "psicoanalisis/Freud_Suenos"]
    mode:        "SOFT" (default) | "HARD"
    chat_history: turnos previos de conversación. Cada turno es un dict
                  {"user": "...", "assistant": "..."}. El cliente es
                  responsable de mantener y enviar el historial — la API
                  es stateless por diseño.
    """

    question: str = Field(..., min_length=1)
    collections: list[str] = Field(..., min_length=1)
    mode: str = Field(default="SOFT", pattern="^(SOFT|HARD)$")
    chat_history: list[dict[str, str]] = Field(default_factory=list)


# ======================================================
# RESPONSE
# ======================================================


class QueryResponse(BaseModel):
    """
    Respuesta de POST /query y POST /query/agent.

    reformulated: solo presente en /query/agent. True si el grafo
                  reformuló la query antes de generar.
    """

    answer: str
    confidence: float
    collections_used: list[str]
    reformulated: bool = False


class CollectionsResponse(BaseModel):
    """Respuesta de GET /collections."""

    collections: list[str]


class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""

    detail: str
