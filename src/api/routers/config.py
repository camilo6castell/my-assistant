"""
Router "config" -- descubrimiento de providers y sus capacidades.

Solo lectura a propósito: ver docstring de src/api/schemas/config.py
para el razonamiento de por qué no hay un PATCH acá.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.config import ProviderInfo, ProvidersResponse
from src.config.models import get_default_think, get_supports
from src.config.settings import settings
from src.nlp.llm.providers import list_provider_configs
from src.nlp.llm.roles import LLMRole

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """
    Lista los providers configurados y qué GenerationOptions acepta
    cada uno -- el frontend usa `supports` para decidir dinámicamente
    qué controles mostrar (ej. el botón de "Pensar" solo aparece
    habilitado si el provider activo lo soporta, deshabilitado con
    tooltip si no), en vez de hardcodear por nombre de modelo.
    `default_think` le dice a la UI en qué estado arranca ese botón
    antes de que el usuario lo toque. `active_generation_provider` le
    dice cuál de todos es el relevante para esa decisión.

    `supports`/`default_think` salen de src/config/models/, no de un
    campo configurado a mano -- no pueden desincronizarse del
    comportamiento real (ver docstring de src/config/models/__init__.py).

    No incluye retrieval/max_turns (ver GenerationDefaults, eliminado):
    esos parámetros pasaron a ser exclusivamente de .env, sin ningún
    override ni valor "actual" que mostrarle al cliente -- ver
    GenerationOptions en src/api/schemas/chat.py.
    """
    table = list_provider_configs()
    return ProvidersResponse(
        providers={
            name: ProviderInfo(
                name=config.name,
                model=config.model,
                supports=sorted(get_supports(config.capabilities, config.model)),
                default_think=get_default_think(config.capabilities, config.model),
            )
            for name, config in table.items()
        },
        active_generation_provider=settings.provider_for(LLMRole.GENERATE),
        provider_roles={role.value: settings.provider_for(role) for role in LLMRole},
    )
