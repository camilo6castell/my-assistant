"""
Registro de proveedores LLM (arquitectura provider-per-rol).

Problema que resuelve:
  El sistema original tenía un único `client` OpenAI-compatible global,
  apuntando a "lo que esté en .env" (local O gemini, nunca ambos a la vez).
  Para usar dos modelos distintos en el mismo grafo (uno reformula, otro
  genera) hace falta poder pedir "el cliente del rol X" en runtime.

Diseño (dos ejes independientes):
  - BACKEND: un runtime concreto -- "flm" | "ollama" | "gemini". Define
    dónde conectarse (base_url), con qué credencial (api_key), y con qué
    implementación de LLMClient hablarle (`client`: "openai_compat" |
    "ollama_native", ver src/nlp/llm/backends/). Es mecánica de
    transporte pura, no sabe nada de modelos concretos. Se configura una
    sola vez en .env.providers (LLM_FLM_URL, LLM_OLLAMA_URL,
    LLM_GEMINI_URL, GEMINI_API_KEY) -- ver _backend_url_table()/
    _BACKEND_CLIENT más abajo.
  - ROL: cada punto del pipeline (ver LLMRole en src/nlp/llm/roles.py)
    elige, en .env.providers (LLM_ROL_GENERATE=flm,qwen3.5:9b, etc. --
    ver Settings.role_spec()), qué backend + qué modelo lo atiende. El
    `capabilities` de un rol es directamente su backend: QUÉ ARCHIVO de
    src/config/models/ mirar para saber qué acepta su modelo (max_tokens,
    think mode y cómo activarlo). Es descripción del modelo, no sabe
    nada de HTTP.

  Ambos ejes son independientes a propósito: dos roles pueden compartir
  backend con modelos distintos (GENERATE=flm,qwen3.5:9b vs
  SUPPLEMENT=flm,qwen3.5:2b), o modelo con backend distinto, sin que uno
  tenga que saber del otro.

  - Los clientes concretos se cachean por (backend, base_url, api_key,
    client) -- no por rol: dos roles con el mismo backend+modelo
    terminan compartiendo una única conexión en vez de abrir una por
    rol (ver ProviderConfig.name, excluido del hash/eq del dataclass).
  - Agregar un backend nuevo (Claude, OpenAI, Mistral...) es agregar una
    entrada a _backend_url_table()/_backend_api_key_table()/
    _BACKEND_CLIENT + su archivo en src/config/models/ (mismas 4
    funciones que flm/ollama/gemini) -- nunca hace falta tocar graph.py,
    nodes.py ni generate.py. Agregar un modelo nuevo bajo un backend
    existente es una entrada en src/config/models/<backend>.py, nada más.
  - Los nodos del grafo no importan este módulo directamente: usan
    ask_llm(..., provider=...) / ask_llm_internal(..., provider=...) en
    generate.py, que sí depende de él. Así el grafo nunca sabe de
    clientes LLM concretos, solo de "qué rol usar" (LLMRole.X.value).

Por qué esto y no el getattr(settings, f"{prefix}_base_url") anterior:
  Ese diseño generaba un AttributeError en cuanto un prefijo (ej.
  "local") no tenía TODOS sus campos `<prefijo>_*` definidos en Settings
  a la vez -- un error de configuración se enteraba recién en el primer
  request, con una traza que no decía qué variable de .env.providers
  faltaba. Acá cada backend tiene su propio dict explícito
  (_backend_url_table()/_backend_api_key_table()/_BACKEND_CLIENT): un
  backend desconocido falla con un ValueError legible al primer uso,
  listando los backends válidos -- nunca un getattr a ciegas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from src.config.settings import settings
from src.nlp.llm.backends.base import LLMClient
from src.nlp.llm.backends.ollama_native import OllamaNativeClient
from src.nlp.llm.backends.openai_compat import OpenAICompatClient
from src.nlp.llm.roles import LLMRole
from src.utils.logger import logger


# backend -> URL configurada en .env.providers (LLM_<BACKEND>_URL). Un
# backend nuevo es una entrada acá + los campos llm_<backend>_url (y,
# si hace falta credencial real, <backend>_api_key) en Settings.
#
# Función, no dict a nivel de módulo: igual que _build_provider_table(),
# para que el valor refleje settings en el momento de la llamada, no en
# el momento del import -- útil en tests que mutan settings.
def _backend_url_table() -> dict[str, str]:
    return {
        "flm": settings.llm_flm_url,
        "ollama": settings.llm_ollama_url,
        "gemini": settings.llm_gemini_url,
    }


# backend -> credencial. flm/ollama son runtimes locales que no validan
# la key (el SDK de OpenAI igual exige un string no vacío); gemini sí
# necesita la real.
def _backend_api_key_table() -> dict[str, str]:
    return {
        "flm": "not-needed",
        "ollama": "not-needed",
        "gemini": settings.gemini_api_key,
    }


# backend -> qué implementación de LLMClient usar (ver
# src/nlp/llm/backends/). "ollama" usa el cliente nativo del paquete
# `ollama` -- ver docstring de OllamaNativeClient para por qué (soporte
# de `think` más confiable que su endpoint OpenAI-compatible). Este
# mapeo es estático (no depende de settings), así que sí vive a nivel de
# módulo.
_BACKEND_CLIENT: dict[str, str] = {
    "flm": "openai_compat",
    "ollama": "ollama_native",
    "gemini": "openai_compat",
}

# client_kind -> factory que arma el LLMClient concreto. Agregar un
# backend nuevo con un client_kind ya existente no requiere tocar esto;
# agregar un client_kind nuevo sí (+ su archivo en src/nlp/llm/backends/).
_CLIENT_FACTORIES: dict[str, Callable[[ProviderConfig], LLMClient]] = {
    "openai_compat": lambda config: OpenAICompatClient(
        base_url=config.base_url, api_key=config.api_key
    ),
    "ollama_native": lambda config: OllamaNativeClient(host=config.base_url),
}


@dataclass(frozen=True)
class ProviderConfig:
    # Nombre del rol que resolvió esta config (ej. "generate") -- solo
    # para logging/display. compare=False: dos roles con el mismo
    # backend+modelo deben cachear al MISMO LLMClient (ver _client_for),
    # así que `name` no participa en el hash/eq del dataclass.
    name: str = field(compare=False)
    # "flm" | "ollama" | "gemini" -- ver _BACKEND_URL/_BACKEND_API_KEY.
    backend: str
    base_url: str
    api_key: str
    model: str
    # "openai_compat" | "ollama_native" -- ver _CLIENT_FACTORIES.
    client: str
    # Clave en src/config/models/ (ej. "flm", "ollama", "gemini") -- qué
    # archivo mirar para las capacidades reales de `model`. Coincide
    # 1:1 con `backend`: no son dos conceptos independientes acá, a
    # diferencia del diseño anterior donde un backend HTTP genérico
    # ("openai_compat") podía servir a capabilities distintas.
    capabilities: str


def _build_provider_table() -> dict[str, ProviderConfig]:
    """
    Resuelve un ProviderConfig por cada LLMRole, leyendo backend+modelo
    desde settings.role_spec(role) (.env.providers: LLM_ROL_*).

    Se construye en una función (no a nivel de módulo) para que los
    valores reflejen settings en el momento de la llamada, no en el
    momento del import — útil en tests que mutan settings.
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
            f"Backend LLM desconocido: {backend!r} (rol {role.value!r}). Válidos: {valid}"
        ) from None

    if not base_url:
        raise ValueError(
            f"El backend {backend!r} (rol {role.value!r}) no tiene URL configurada "
            f"-- falta LLM_{backend.upper()}_URL en .env.providers."
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
    Versión pública de _build_provider_table() para consumidores externos
    al módulo (ej. src/api/routers/config.py). get_provider()/get_client()
    siguen siendo el camino interno para resolver un provider por nombre;
    esto es solo para listar todos a la vez (GET /api/v1/config/providers).

    Las claves son nombres de rol ("generate", "reformulate", "review",
    "web_supplement"), no de backend -- cada rol es su propio "provider"
    en esta arquitectura (ver docstring del módulo).
    """
    return _build_provider_table()


@cache
def _client_for(config: ProviderConfig) -> LLMClient:
    """
    Un LLMClient por (backend, base_url, api_key, model, client) único
    -- evita recrear conexiones. `name` (el rol) no participa en el
    cache key (ver ProviderConfig.name, compare=False): dos roles
    apuntando al mismo backend+modelo comparten cliente.
    """
    factory = _CLIENT_FACTORIES.get(config.client)
    if factory is None:
        raise ValueError(
            f"Tipo de cliente LLM desconocido: {config.client!r} "
            f"(backend {config.backend!r}). Válidos: {sorted(_CLIENT_FACTORIES)}"
        )
    logger.info(
        f"[providers] Inicializando cliente LLM | backend={config.backend} "
        f"| client={config.client} | base_url={config.base_url} | model={config.model}"
    )
    return factory(config)


def get_provider(name: str) -> ProviderConfig:
    """
    Devuelve la configuración del rol `name` (ej. "generate").

    Lanza ValueError con la lista de roles válidos si `name` no existe
    -- falla rápido y explícito en vez de un 401 confuso del API del
    modelo.
    """
    table = _build_provider_table()
    try:
        config = table[name]
    except KeyError as exc:
        valid = ", ".join(sorted(table))
        raise ValueError(f"Rol LLM desconocido: {name!r}. Válidos: {valid}") from exc

    return config


def get_client(name: str) -> tuple[LLMClient, ProviderConfig]:
    """Devuelve (client, config) listos para usar en LLMClient.complete()."""
    config = get_provider(name)
    return _client_for(config), config
