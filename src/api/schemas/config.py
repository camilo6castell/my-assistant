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


class ProvidersResponse(BaseModel):
    """Respuesta de GET /api/v1/config/providers."""

    providers: dict[str, ProviderInfo]
