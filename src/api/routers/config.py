"""
Router "config" -- discovery of providers and their capabilities.

Deliberately read-only: see docstring of src/api/schemas/config.py
for the reasoning behind why there is no PATCH endpoint here.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.config import ProviderInfo, ProvidersResponse
from src.config.models import get_default_think, get_supports
from src.nlp.llm.providers import list_provider_configs
from src.nlp.llm.roles import LLMRole

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """
    Lists configured providers and which GenerationOptions each one
    accepts -- the frontend uses `supports` to dynamically decide which
    controls to show (e.g. the "Think" button only appears enabled if
    the active provider supports it, disabled with tooltip otherwise),
    instead of hardcoding by model name. `default_think` tells the UI
    what state that button starts in before the user touches it.
    `active_generation_provider` tells it which of all providers is the
    relevant one for that decision.

    `supports`/`default_think` come from src/config/models/, not from a
    hand-configured field -- they cannot drift out of sync with actual
    behavior (see docstring of src/config/models/__init__.py).

    Does not include retrieval/max_turns (see GenerationDefaults,
    removed): those parameters became exclusively .env-driven, with no
    override or "current" value to show the client -- see
    GenerationOptions in src/api/schemas/chat.py.
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
        active_generation_provider=LLMRole.GENERATE.value,
    )
