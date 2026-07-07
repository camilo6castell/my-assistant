"""
Schemas del dominio "config" -- descubrimiento de providers/modelos
disponibles y qué parámetros de generación soporta cada uno.

Deliberadamente de solo lectura: no existe un PATCH /config que mute
base_url/api_key/modelo en runtime. Esa es configuración de
infraestructura (vive en .env) -- exponerla por API sería un riesgo de
seguridad (cualquiera con acceso al endpoint podría redirigir el backend
a un servidor arbitrario) sin ganancia real de producto.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProviderInfo(BaseModel):
    name: str
    model: str
    supports: list[str]


class GenerationDefaults(BaseModel):
    """
    Valores actuales de .env/settings para los parámetros de generación y
    retrieval que la UI permite overridear por-conversación (ver
    GenerationOptions en src/api/schemas/chat.py).

    Son solo de lectura -- la UI los usa para mostrar "valor actual" antes
    de que el usuario decida modificarlo, y como fallback cuando el
    override de la conversación es None. No hay forma de cambiarlos vía
    API (ver docstring del módulo): cambiar esto significa editar .env.
    """

    temperature: float
    max_turns: int
    hard_top_k_initial: int
    hard_top_k_final: int
    soft_top_k_initial: int
    soft_top_k_final: int


class ProvidersResponse(BaseModel):
    """Respuesta de GET /api/v1/config/providers."""

    providers: dict[str, ProviderInfo]
    # Nombre del provider que /query y /query/agent usan para la respuesta
    # final al usuario (settings.generate_provider) -- el frontend lo usa
    # para saber cuál entrada de `providers` mirar al decidir qué controles
    # de GenerationOptions mostrar (ej. el switch de think mode).
    active_generation_provider: str
    defaults: GenerationDefaults
