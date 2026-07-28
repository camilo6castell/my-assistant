"""
LLM provider registry (provider-per-role architecture).

Problem it solves:
  The original system had a single global OpenAI-compatible `client`,
  pointing to "whatever is in .env" (local or gemini, never both at
  once). To use two different models in the same graph (one reformulates,
  one generates) you need to be able to request "the client for role X"
  at runtime.

Design (two independent axes):
  - BACKEND: a concrete runtime -- "flm" | "ollama" | "gemini". Defines
    where to connect (base_url), with what credential (api_key), and
    what LLMClient implementation to talk to it (`client`:
    "openai_compat" | "ollama_native", see src/nlp/llm/backends/). It
    is pure transport mechanics, knows nothing about concrete models.
    Configured once in .env.providers (LLM_FLM_URL, LLM_OLLAMA_URL,
    LLM_GEMINI_URL, GEMINI_API_KEY) -- see _backend_url_table()/
    _BACKEND_CLIENT below.
  - ROLE: each pipeline step (see LLMRole in src/nlp/llm/roles.py)
    chooses, in .env.providers (LLM_ROLE_GENERATE=flm,qwen3.5:9b, etc.
    -- see Settings.role_spec()), which backend + which model serves it.
    A role's `capabilities` is directly its backend: WHICH file in
    src/config/models/ to look at to know what its model accepts
    (max_tokens, think mode and how to enable it). It is model
    description, knows nothing about HTTP.

  Both axes are independent on purpose: two roles can share a backend
  with different models (GENERATE=flm,qwen3.5:9b vs
  SUPPLEMENT=flm,qwen3.5:2b), or share a model with different backends,
  without one needing to know about the other.

  - Concrete clients are cached by (backend, base_url, api_key, client)
    -- not by role: two roles with the same backend+model end up
    sharing a single connection instead of opening one per role (see
    ProviderConfig.name, excluded from the dataclass hash/eq).
  - Adding a new backend (Claude, OpenAI, Mistral...) means adding an
    entry to _backend_url_table()/_backend_api_key_table()/
    _BACKEND_CLIENT + its file in src/config/models/ (same 4 functions
    as flm/ollama/gemini) -- you never need to touch graph.py, nodes.py
    or generate.py. Adding a new model under an existing backend is an
    entry in src/config/models/<backend>.py, nothing more.
  - Graph nodes do not import this module directly: they use
    ask_llm(..., provider=...) / ask_llm_internal(..., provider=...) in
    generate.py, which does depend on it. This way the graph never knows
    about concrete LLM clients, only about "which role to use"
    (LLMRole.X.value).

Why this instead of the previous getattr(settings, f"{prefix}_base_url"):
  That design generated an AttributeError as soon as a prefix (e.g.
  "local") did not have ALL its `<prefix>_*` fields defined in Settings
  at once -- a configuration error was only discovered on the first
  request, with a traceback that did not say which .env.providers
  variable was missing. Here each backend has its own explicit dict
  (_backend_url_table()/_backend_api_key_table()/_BACKEND_CLIENT): an
  unknown backend fails with a readable ValueError on first use, listing
  the valid backends -- never a blind getattr.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from src.config.settings import settings
from src.domain.models import LLMRole
from src.nlp.llm.backends.base import LLMClient
from src.nlp.llm.backends.ollama_native import OllamaNativeClient
from src.nlp.llm.backends.openai_compat import OpenAICompatClient
from src.utils.logger import logger


# backend -> URL configured in .env.providers (LLM_<BACKEND>_URL). A
# new backend is an entry here + the llm_<backend>_url (and, if a real
# credential is needed, <backend>_api_key) fields in Settings.
#
# Function, not a module-level dict: like _build_provider_table(), so
# the value reflects settings at call time, not at import time -- useful
# in tests that mutate settings.
def _backend_url_table() -> dict[str, str]:
    return {
        "flm": settings.llm_flm_url,
        "ollama": settings.llm_ollama_url,
        "gemini": settings.llm_gemini_url,
    }


# backend -> credential. flm/ollama are local runtimes that do not
# validate the key (the OpenAI SDK still requires a non-empty string);
# gemini does need the real one.
def _backend_api_key_table() -> dict[str, str]:
    return {
        "flm": "not-needed",
        "ollama": "not-needed",
        "gemini": settings.gemini_api_key,
    }


# backend -> which LLMClient implementation to use (see
# src/nlp/llm/backends/). "ollama" uses the native client from the
# `ollama` package -- see OllamaNativeClient docstring for why (more
# reliable `think` support than its OpenAI-compatible endpoint). This
# mapping is static (does not depend on settings), so it lives at module
# level.
_BACKEND_CLIENT: dict[str, str] = {
    "flm": "openai_compat",
    "ollama": "ollama_native",
    "gemini": "openai_compat",
}

# client_kind -> factory that builds the concrete LLMClient. Adding a
# new backend with an existing client_kind does not require touching
# this; adding a new client_kind does (+ its file in
# src/nlp/llm/backends/).
_CLIENT_FACTORIES: dict[str, Callable[[ProviderConfig], LLMClient]] = {
    "openai_compat": lambda config: OpenAICompatClient(
        base_url=config.base_url, api_key=config.api_key
    ),
    "ollama_native": lambda config: OllamaNativeClient(host=config.base_url),
}


@dataclass(frozen=True)
class ProviderConfig:
    # Name of the role that resolved this config (e.g. "generate") -- for
    # logging/display only. compare=False: two roles with the same
    # backend+model must cache the SAME LLMClient (see _client_for), so
    # `name` does not participate in the dataclass hash/eq.
    name: str = field(compare=False)
    # "flm" | "ollama" | "gemini" -- see _BACKEND_URL/_BACKEND_API_KEY.
    backend: str
    base_url: str
    api_key: str
    model: str
    # "openai_compat" | "ollama_native" -- see _CLIENT_FACTORIES.
    client: str
    # Key in src/config/models/ (e.g. "flm", "ollama", "gemini") -- which
    # file to look at for the real capabilities of `model`. Matches 1:1
    # with `backend`: they are not two independent concepts here, unlike
    # the previous design where a generic HTTP backend
    # ("openai_compat") could serve different capabilities.
    capabilities: str


def _build_provider_table() -> dict[str, ProviderConfig]:
    """
    Resolves a ProviderConfig for each LLMRole, reading backend+model
    from settings.role_spec(role) (.env.providers: LLM_ROLE_*).

    Built as a function (not at module level) so values reflect settings
    at call time, not at import time -- useful in tests that mutate
    settings.
    """
    return {role.value: _provider_config_for_role(role) for role in LLMRole}


def _provider_config_for_role(role: LLMRole) -> ProviderConfig:
    backend, model = settings.role_spec(role)
    backend_urls = _backend_url_table()
    try:
        base_url = backend_urls[backend]
        api_key = _backend_api_key_table()[backend]
        client = _BACKEND_CLIENT[backend]
    except KeyError:
        valid = ", ".join(sorted(backend_urls))
        raise ValueError(
            f"Unknown LLM backend: {backend!r} (role {role.value!r}). Valid: {valid}"
        ) from None

    if not base_url:
        raise ValueError(
            f"Backend {backend!r} (role {role.value!r}) has no URL configured "
            f"-- LLM_{backend.upper()}_URL is missing in .env.providers."
        )

    return ProviderConfig(
        name=role.value,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        client=client,
        capabilities=backend,
    )


def list_provider_configs() -> dict[str, ProviderConfig]:
    """
    Public version of _build_provider_table() for external consumers
    (e.g. src/api/routers/config.py). get_provider()/get_client() remain
    the internal path to resolve a provider by name; this is only for
    listing all at once (GET /api/v1/config/providers).

    Keys are role names ("generate", "reformulate", "review",
    "web_supplement"), not backends -- each role is its own "provider"
    in this architecture (see module docstring).
    """
    return _build_provider_table()


@cache
def _client_for(config: ProviderConfig) -> LLMClient:
    """
    One LLMClient per unique (backend, base_url, api_key, model, client)
    -- avoids recreating connections. `name` (the role) does not
    participate in the cache key (see ProviderConfig.name,
    compare=False): two roles pointing to the same backend+model share
    a client.
    """
    factory = _CLIENT_FACTORIES.get(config.client)
    if factory is None:
        raise ValueError(
            f"Unknown LLM client type: {config.client!r} "
            f"(backend {config.backend!r}). Valid: {sorted(_CLIENT_FACTORIES)}"
        )
    logger.info(
        f"[providers] Initializing LLM client | backend={config.backend} "
        f"| client={config.client} | base_url={config.base_url} | model={config.model}"
    )
    return factory(config)


def get_provider(name: str) -> ProviderConfig:
    """
    Returns the configuration for role `name` (e.g. "generate").

    Raises ValueError with the list of valid roles if `name` does not
    exist -- fails fast and explicitly instead of producing a confusing
    401 from the model API.
    """
    table = _build_provider_table()
    try:
        config = table[name]
    except KeyError as exc:
        valid = ", ".join(sorted(table))
        raise ValueError(f"Unknown LLM role: {name!r}. Valid: {valid}") from exc

    return config


def get_client(name: str) -> tuple[LLMClient, ProviderConfig]:
    """Returns (client, config) ready to use with LLMClient.complete()."""
    config = get_provider(name)
    return _client_for(config), config
