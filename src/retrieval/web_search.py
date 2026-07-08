"""
Búsqueda web vía Tavily, usada como fuente de retrieval complementaria o
alternativa a las colecciones locales (ver QueryRequest.web_search en
src/api/schemas/chat.py y su uso en src/api/routers/chat.py).

Por qué Tavily y no golpear un motor de búsqueda directo: hacer scraping
de Google/Bing viola sus ToS y es frágil (el HTML cambia sin aviso).
Tavily está pensado específicamente para RAG/LLMs -- devuelve snippets
ya limpios, con URL, vía una API HTTP simple (mismo patrón que ya usás
para el resto de HTTP externo con `requests`, ver src/ingest/).

Este módulo NUNCA deja que un fallo acá tumbe el request completo: toda
excepción (timeout, red, JSON inválido, HTTP 4xx/5xx) se atrapa, se
loguea, y se devuelve un WebSearchOutcome con resultados vacíos -- el
router decide qué hacer con eso (ver chat.py). La única distinción que
SÍ se propaga es "se agotó la cuota" (status=QUOTA_EXCEEDED) vs.
"cualquier otro fallo" (status=ERROR): son la misma "sin resultados"
para el pipeline de generación, pero requieren UX distinta (ver
GenerationSection.tsx -- un fallo de red transitorio no debería
deshabilitar el botón "Web", pero agotar el free tier de Tavily sí
debería avisarle al usuario en vez de fallar en silencio cada vez).
"""

from __future__ import annotations

from enum import Enum

import requests
from pydantic import BaseModel

from src.config.settings import settings
from src.utils.logger import logger

TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Códigos HTTP documentados por Tavily para límites de cuenta (ver SDK
# oficial tavily-python: ambos se mapean a ForbiddenError).
#   432 = Plan Limit Exceeded (se agotaron los créditos del plan, ej. el
#         free tier de 1000/mes).
#   433 = Pay-As-You-Go Limit Exceeded (tope de gasto configurado en la
#         cuenta, para cuentas con PAYGO habilitado).
# Fuente: https://docs.tavily.com (research endpoint) y
# https://github.com/tavily-ai/tavily-n8n-node (tabla de errores).
TAVILY_QUOTA_EXCEEDED_STATUS_CODES = frozenset({432, 433})


class WebSearchResult(BaseModel):
    title: str
    url: str
    content: str  # snippet ya resumido por Tavily, no el HTML crudo


class WebSearchStatus(str, Enum):
    OK = "ok"
    # Se agotaron los créditos de la cuenta de Tavily (free tier u otro
    # plan) -- distinto de un fallo transitorio: no tiene sentido
    # reintentar hasta que cambie el mes/plan (ver
    # settings.web_search_quota_exceeded en el frontend).
    QUOTA_EXCEEDED = "quota_exceeded"
    # Cualquier otro fallo: sin API key, feature apagada, timeout, error
    # de red, JSON inválido, u otro código HTTP no-2xx.
    ERROR = "error"


class WebSearchOutcome(BaseModel):
    results: list[WebSearchResult]
    status: WebSearchStatus


def search_web(query: str, max_results: int | None = None) -> WebSearchOutcome:
    """
    Devuelve como máximo max_results snippets de la web para `query`,
    junto con un status que distingue "sin resultados por cuota agotada"
    de cualquier otro motivo (ver WebSearchStatus). Nunca lanza excepción.
    """
    if not settings.web_search_enabled:
        logger.warning("[web_search] WEB_SEARCH_ENABLED=false -- se omite la búsqueda.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    if not settings.tavily_api_key:
        logger.warning("[web_search] TAVILY_API_KEY no configurada -- se omite la búsqueda.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    effective_max_results = max_results or settings.web_search_max_results

    try:
        response = requests.post(
            TAVILY_ENDPOINT,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": effective_max_results,
                "include_answer": False,
            },
            timeout=settings.web_search_timeout,
        )
    except requests.RequestException as e:
        logger.warning(f"[web_search] Fallo la consulta a Tavily: {e}")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    if response.status_code in TAVILY_QUOTA_EXCEEDED_STATUS_CODES:
        logger.warning(
            f"[web_search] Cuota de Tavily agotada (HTTP {response.status_code}) -- "
            "revisá tu plan en https://app.tavily.com."
        )
        return WebSearchOutcome(results=[], status=WebSearchStatus.QUOTA_EXCEEDED)

    if not response.ok:
        logger.warning(
            f"[web_search] Tavily devolvio HTTP {response.status_code}: {response.text[:200]}"
        )
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    try:
        payload = response.json()
    except ValueError:
        logger.warning("[web_search] Respuesta de Tavily no es JSON valido.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    results: list[WebSearchResult] = []
    for item in payload.get("results", []):
        title = item.get("title")
        url = item.get("url")
        content = item.get("content")
        if not (title and url and content):
            continue
        results.append(WebSearchResult(title=title, url=url, content=content))

    logger.info(f"[web_search] Tavily devolvio {len(results)} resultado(s) para: {query!r}")
    return WebSearchOutcome(results=results, status=WebSearchStatus.OK)
