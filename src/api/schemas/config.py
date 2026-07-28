"""
Schemas for the "config" domain -- discovery of available providers/models
and which generation parameters each one supports.

Deliberately read-only: there is no PATCH /config that mutates
base_url/api_key/model at runtime. That is infrastructure configuration
(lives in .env) -- exposing it via API would be a security risk (anyone
with access to the endpoint could redirect the backend to an arbitrary
server) with no real product benefit.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProviderInfo(BaseModel):
    name: str
    model: str
    # See get_supports() in src/config/models/__init__.py -- which
    # GenerationOptions this provider's model accepts. The frontend uses
    # it to dynamically decide which controls to show (e.g. the "Think"
    # button only appears enabled if "think_mode" is here; otherwise it
    # is shown disabled with a tooltip).
    supports: list[str]
    # See get_default_think() in src/config/models/__init__.py -- the
    # enable_thinking/think value already written in _MODELS[model] for
    # this model (None if the model has no reasoning mode). The frontend
    # uses it so the "Think" button starts reflecting the REAL behavior
    # of the model before the user touches it, instead of starting off
    # by default regardless of what the backend config says.
    default_think: bool | None


class ProvidersResponse(BaseModel):
    """Response for GET /api/v1/config/providers."""

    providers: dict[str, ProviderInfo]
    # Role name that /query and /query/agent use for the final response
    # to the user (LLMRole.GENERATE.value) -- the frontend uses it to
    # know which entry in `providers` to look at when deciding which
    # GenerationOptions controls to show (e.g. the think mode switch).
    # The keys of `providers` ARE role names (see docstring of
    # list_provider_configs in src/nlp/llm/providers.py), so no
    # separate role -> provider map is needed: it is the identity.
    active_generation_provider: str
