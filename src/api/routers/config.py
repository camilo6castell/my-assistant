"""
Router "config" -- descubrimiento de providers y sus capacidades.

Solo lectura a propósito: ver docstring de src/api/schemas/config.py
para el razonamiento de por qué no hay un PATCH acá.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.config import ProviderInfo, ProvidersResponse
from src.llm.providers import list_provider_configs

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """
    Lista los providers configurados y qué GenerationOptions acepta
    cada uno -- el frontend usa `supports` para decidir dinámicamente
    qué controles mostrar (ej. el switch de "think mode" solo aparece
    si el provider activo lo soporta), en vez de hardcodear por nombre
    de modelo.
    """
    table = list_provider_configs()
    return ProvidersResponse(
        providers={
            name: ProviderInfo(
                name=config.name,
                model=config.model,
                supports=sorted(config.supports),
            )
            for name, config in table.items()
        }
    )
