"""
Schemas del dominio "chat" -- las funciones regulares de la aplicación:
hacer una pregunta, listar colecciones disponibles.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    """
    Overrides opcionales de generación para una llamada puntual a
    POST /query o POST /query/agent.

    La temperatura NO es uno de estos campos -- es responsabilidad
    exclusiva de cada archivo en src/config/models/ (_MODELS[model]),
    igual que top_p, presence_penalty, etc. No es un override
    por-request: es una propiedad del modelo, se edita en el JSON de su
    backend y aplica a todas las llamadas a ese modelo. Si en algún
    momento hace falta volver a exponer un override real de temperatura
    por-request, ver el historial de este archivo antes de reinventar la
    rueda (el diseño anterior con Settings.llm_temperature pisaba
    silenciosamente el valor de _MODELS en cada request; ver
    src/llm/generate.py).

    think_mode: campo con nombre porque es un concepto común a la
    mayoría de providers/modelos razonadores (pensado para el botón
    "Pensar" en la UI). None significa "usar el default que ya está
    escrito en _MODELS[model] para este modelo" -- no "usar 0.2 de un
    .env" como pasaba antes con temperature.

    extra: escape hatch genérico para cualquier parámetro que el backend
    NO modela explícitamente (top_p, presence_penalty, un flag propio de
    un provider nuevo...). Se envía tal cual al provider vía extra_body
    del cliente OpenAI -- sin validar su contenido, porque por definición
    puede ser cualquier cosa que un provider específico entienda. Por eso
    requiere que el provider declare "extra" en su `supports` (ver
    ProviderConfig en src/llm/providers.py): es un opt-in explícito, no
    "cualquier JSON pasa a cualquier servidor".

    max_turns: cuántos turnos de chat_history se incluyen en el prompt
    (ventana deslizante, ver build_messages en src/llm/generate.py).
    None = usar settings.max_turns.

    top_k_initial/top_k_final: overrides del retrieval (ver
    src/retrieval/search.py). None = usar el default de settings para el
    modo (SOFT/HARD) de este request. Si ambos vienen seteados,
    top_k_final no puede ser mayor que top_k_initial -- si solo viene uno
    de los dos, se valida contra el default de settings para ese modo (ver
    _validate_generation_options en src/api/routers/chat.py, que sí
    conoce el modo del request).

    Ninguno de estos valores muta Settings ni ProviderConfig -- son
    parámetros por-request, no estado del servidor (ver docstring de
    src/llm/generate.py para el porqué). El servidor valida cada campo
    contra ProviderConfig.supports antes de usarlo y devuelve 400 si el
    provider activo no lo soporta.
    """

    think_mode: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    extra: dict[str, Any] | None = None
    max_turns: int | None = Field(default=None, gt=0)
    top_k_initial: int | None = Field(default=None, gt=0)
    top_k_final: int | None = Field(default=None, gt=0)


class WebSource(BaseModel):
    """Fuente web citada en una respuesta (ver QueryResponse.web_sources)."""

    title: str
    url: str


class QueryRequest(BaseModel):
    """
    Payload para POST /query y POST /query/agent.

    collections:    tokens con la misma sintaxis que el CLI, ej.
                    ["sociologia", "psicoanalisis/Freud_Suenos"].
    mode:           "SOFT" (default) | "HARD"
    chat_history:   turnos previos de conversación. Cada turno es un dict
                    {"user": "...", "assistant": "..."}. El cliente es
                    responsable de mantener y reenviar el historial -- la
                    API es stateless por diseño.
    conversation_id: id opaco que el cliente genera una vez por
                    conversación (ej. crypto.randomUUID() en el
                    frontend). Hace falta si esa conversación tiene
                    colecciones efímeras (POST /api/v1/files) o archivos
                    adjuntos ad-hoc (POST /api/v1/attachments) -- si no
                    tiene ninguno de los dos, se puede omitir.
    generation:     overrides opcionales de temperatura/tokens/think mode.
    web_search:     si True, complementa (o reemplaza, si no hay
                    colecciones/archivos) el contexto con una búsqueda
                    web vía Tavily -- ver _run_web_search en
                    src/api/routers/chat.py para el detalle de los dos
                    modos. Requiere settings.web_search_enabled=True; si
                    no, el router devuelve 400. Best-effort: si la
                    búsqueda web falla y SÍ hay colecciones/archivos, el
                    request no falla por eso (ver QueryResponse.used_web_search).

    Sin `collections`, sin colección efímera, y con web_search=False: NO
    es un error (a diferencia de una versión anterior de este schema,
    que exigía al menos una fuente de contexto). El router responde
    directo con el LLM, sin ningún system prompt ni retrieval -- ver
    _answer_raw() en src/api/routers/chat.py. Es el modo esperado para
    preguntas sueltas o para adjuntar un archivo puntual
    (POST /api/v1/attachments) sin tener ninguna colección elegida: el
    usuario es responsable de darle rol/reglas/tarea al modelo en su
    propio mensaje.
    """

    question: str = Field(..., min_length=1)
    collections: list[str] = Field(default_factory=list)
    mode: str = Field(default="SOFT", pattern="^(SOFT|HARD)$")
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str | None = None
    generation: GenerationOptions | None = None
    web_search: bool = False


class QueryResponse(BaseModel):
    """
    Respuesta de POST /query y POST /query/agent.

    reformulated:     solo relevante en /query/agent. True si el grafo
                       reformuló la query antes de generar.
    used_web_search:   lo que REALMENTE pasó, no lo que se pidió -- False
                       si se pidió web_search=True pero la búsqueda no
                       devolvió resultados utilizables (ver
                       src/retrieval/web_search.py), aunque el resto de
                       la respuesta se haya generado igual con el
                       contexto local disponible.
    web_sources:       fuentes web efectivamente usadas (título + URL),
                       para que el frontend las muestre como citas. None
                       si used_web_search es False.
    web_search_quota_exceeded: True si Tavily devolvió que se agotó la
                       cuota de la cuenta (free tier u otro plan) --
                       señal distinta de "sin resultados" para que el
                       frontend pueda avisar al usuario y deshabilitar
                       el botón de búsqueda web en vez de fallar en
                       silencio en cada mensaje siguiente.
    """

    answer: str
    confidence: float
    collections_used: list[str]
    reformulated: bool = False
    used_web_search: bool = False
    web_sources: list[WebSource] | None = None
    web_search_quota_exceeded: bool = False


class CollectionsResponse(BaseModel):
    """Respuesta de GET /api/v1/collections."""

    collections: list[str]
