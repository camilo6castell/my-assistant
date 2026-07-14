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
    # Ver get_supports() en src/config/models/__init__.py -- qué
    # GenerationOptions acepta el modelo de este provider. El frontend
    # lo usa para decidir dinámicamente qué controles mostrar (ej. el
    # botón de "Pensar" solo aparece habilitado si "think_mode" está
    # acá; si no, se muestra deshabilitado con un tooltip).
    supports: list[str]
    # Ver get_default_think() en src/config/models/__init__.py -- el
    # valor de enable_thinking/think ya escrito en _MODELS[model] para
    # este modelo (None si el modelo no tiene modo de razonamiento). El
    # frontend lo usa para que el botón "Pensar" arranque reflejando el
    # comportamiento REAL del modelo antes de que el usuario lo toque,
    # en vez de arrancar apagado por defecto sin importar qué diga la
    # config del backend.
    default_think: bool | None


class ProvidersResponse(BaseModel):
    """Respuesta de GET /api/v1/config/providers."""

    providers: dict[str, ProviderInfo]
    # Nombre del provider que /query y /query/agent usan para la respuesta
    # final al usuario (settings.provider_for(LLMRole.GENERATE)) -- el
    # frontend lo usa para saber cuál entrada de `providers` mirar al
    # decidir qué controles de GenerationOptions mostrar (ej. el switch
    # de think mode). Redundante con provider_roles["generate"] de abajo,
    # pero se mantiene como campo propio para no romper el contrato que
    # ya usa el frontend.
    active_generation_provider: str
    # Mapa completo rol -> provider (ver LLMRole en src/llm/roles.py y
    # Settings.provider_for) -- de solo lectura, igual que el resto de
    # este endpoint; pensado para poder inspeccionar de un vistazo qué
    # modelo atiende cada punto del pipeline sin tener que leer .env.
    provider_roles: dict[str, str]
