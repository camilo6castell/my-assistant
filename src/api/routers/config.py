"""
Router "config" -- descubrimiento de providers y sus capacidades.

Solo lectura a propósito: ver docstring de src/api/schemas/config.py
para el razonamiento de por qué no hay un PATCH acá.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.config import GenerationDefaults, ProviderInfo, ProvidersResponse
from src.config.models import get_model_capabilities, supports_set
from src.config.settings import settings
from src.llm.providers import list_provider_configs
from src.llm.roles import LLMRole

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """
    Lista los providers configurados y qué GenerationOptions acepta
    cada uno -- el frontend usa `supports` para decidir dinámicamente
    qué controles mostrar (ej. el switch de "think mode" solo aparece
    si el provider activo lo soporta), en vez de hardcodear por nombre
    de modelo. `active_generation_provider` le dice cuál de todos es
    el relevante para esa decisión.

    `supports` sale de ModelCapabilities (src/config/models/), no de un
    campo configurado a mano -- no puede desincronizarse del
    comportamiento real (ver docstring de src/config/models/__init__.py).
    """
    table = list_provider_configs()
    return ProvidersResponse(
        providers={
            name: ProviderInfo(
                name=config.name,
                model=config.model,
                supports=sorted(
                    supports_set(get_model_capabilities(config.capabilities, config.model))
                ),
            )
            for name, config in table.items()
        },
        active_generation_provider=settings.provider_for(LLMRole.GENERATE),
        provider_roles={role.value: settings.provider_for(role) for role in LLMRole},
        defaults=GenerationDefaults(
            temperature=settings.llm_temperature,
            max_turns=settings.max_turns,
            hard_top_k_initial=settings.hard_top_k_initial,
            hard_top_k_final=settings.hard_top_k_final,
            soft_top_k_initial=settings.soft_top_k_initial,
            soft_top_k_final=settings.soft_top_k_final,
        ),
    )
