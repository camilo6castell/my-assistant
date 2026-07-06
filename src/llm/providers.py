"""
Registro de proveedores LLM (arquitectura provider-per-node).

Problema que resuelve:
  El sistema original tenía un único `client` OpenAI-compatible global,
  apuntando a "lo que esté en .env" (local O gemini, nunca ambos a la vez).
  Para usar dos modelos distintos en el mismo grafo (uno reformula, otro
  genera) hace falta poder pedir "el cliente del proveedor X" en runtime.

Diseño (dos ejes independientes):
  - `client`: QUÉ IMPLEMENTACIÓN de LLMClient usar para hablar con este
    proveedor -- "openai_compat" o "ollama_native" (ver
    src/llm/backends/). Es mecánica de transporte pura, no sabe nada de
    modelos concretos.
  - `capabilities`: QUÉ ARCHIVO de src/config/models/ mirar para saber
    qué acepta el modelo de este proveedor (temperature, max_tokens,
    think mode y cómo activarlo). Es descripción del modelo, no sabe
    nada de HTTP.
  Ambos ejes son independientes a propósito: dos proveedores podrían
  compartir el mismo `client` (dos servidores OpenAI-compatible) con
  `capabilities` distintas (modelos distintos), o viceversa.

  - Los clientes concretos se cachean por ProviderConfig completo (un
    dataclass frozen y hashable) -- un cliente por proveedor, no por
    request.
  - Agregar un proveedor nuevo (Claude, OpenAI, Mistral...) es agregar
    una entrada a _PROVIDER_ENV_PREFIXES + los campos correspondientes
    en Settings, no tocar graph.py, nodes.py ni generate.py. Agregar un
    modelo nuevo bajo un backend existente es una entrada en
    src/config/models/<backend>.py, nada más.
  - Los nodos del grafo no importan este módulo directamente: usan
    ask_llm(..., provider=...) / ask_llm_internal(..., provider=...) en
    generate.py, que sí depende de él. Así el grafo nunca sabe de
    clientes LLM concretos, solo de "qué proveedor usar".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from src.config.settings import settings
from src.llm.backends.base import LLMClient
from src.llm.backends.ollama_native import OllamaNativeClient
from src.llm.backends.openai_compat import OpenAICompatClient
from src.utils.logger import logger


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    # "openai_compat" | "ollama_native" -- ver _CLIENT_FACTORIES abajo.
    client: str
    # Clave en src/config/models/ (ej. "fastflowlm", "ollama", "gemini")
    # -- qué archivo mirar para las capacidades reales de `model`.
    capabilities: str


# (nombre, prefijo en Settings) -- agregar un provider nuevo es agregar
# una tupla acá + los campos <prefijo>_base_url/_api_key/_model/_client/
# _capabilities en Settings (mismo patrón que local_*/gemini_*).
_PROVIDER_ENV_PREFIXES: tuple[str, ...] = ("local", "gemini")

# client_kind -> factory que arma el LLMClient concreto a partir de un
# ProviderConfig. Agregar un backend nuevo (ej. un cliente propio de
# Anthropic) es una entrada acá + su archivo en src/llm/backends/.
_CLIENT_FACTORIES: dict[str, Callable[[ProviderConfig], LLMClient]] = {
    "openai_compat": lambda config: OpenAICompatClient(
        base_url=config.base_url, api_key=config.api_key
    ),
    "ollama_native": lambda config: OllamaNativeClient(host=config.base_url),
}


def _build_provider_table() -> dict[str, ProviderConfig]:
    """
    Lee la configuración de cada proveedor desde settings.

    Se construye en una función (no a nivel de módulo) para que los
    valores reflejen settings en el momento de la llamada, no en el
    momento del import — útil en tests que mutan settings.
    """
    return {
        prefix: ProviderConfig(
            name=prefix,
            base_url=getattr(settings, f"{prefix}_base_url"),
            api_key=getattr(settings, f"{prefix}_api_key"),
            model=getattr(settings, f"{prefix}_model"),
            client=getattr(settings, f"{prefix}_client"),
            capabilities=getattr(settings, f"{prefix}_capabilities"),
        )
        for prefix in _PROVIDER_ENV_PREFIXES
    }


def list_provider_configs() -> dict[str, ProviderConfig]:
    """
    Versión pública de _build_provider_table() para consumidores externos
    al módulo (ej. src/api/routers/config.py). get_provider()/get_client()
    siguen siendo el camino interno para resolver un provider por nombre;
    esto es solo para listar todos a la vez (GET /api/v1/config/providers).
    """
    return _build_provider_table()


@lru_cache(maxsize=None)
def _client_for(config: ProviderConfig) -> LLMClient:
    """
    Un LLMClient por ProviderConfig único -- evita recrear conexiones.

    ProviderConfig es un dataclass frozen de solo strings, por lo tanto
    hashable: sirve directo como key de lru_cache sin descomponerlo en
    argumentos posicionales.
    """
    factory = _CLIENT_FACTORIES.get(config.client)
    if factory is None:
        raise ValueError(
            f"Tipo de cliente LLM desconocido: {config.client!r} "
            f"(provider {config.name!r}). Válidos: {sorted(_CLIENT_FACTORIES)}"
        )
    logger.info(
        f"[providers] Inicializando cliente LLM | provider={config.name} "
        f"| client={config.client} | base_url={config.base_url}"
    )
    return factory(config)


def get_provider(name: str) -> ProviderConfig:
    """
    Devuelve la configuración del proveedor `name`.

    Lanza ValueError con la lista de proveedores válidos si `name` no
    existe — falla rápido y explícito en vez de un 401 confuso del API
    del modelo.
    """
    table = _build_provider_table()
    try:
        config = table[name]
    except KeyError as exc:
        valid = ", ".join(sorted(table))
        raise ValueError(
            f"Proveedor LLM desconocido: {name!r}. Válidos: {valid}"
        ) from exc

    if not config.base_url or not config.model:
        raise ValueError(
            f"Proveedor {name!r} no está configurado "
            f"(falta {name.upper()}_BASE_URL o {name.upper()}_MODEL en .env)"
        )

    return config


def get_client(name: str) -> tuple[LLMClient, ProviderConfig]:
    """Devuelve (client, config) listos para usar en LLMClient.complete()."""
    config = get_provider(name)
    return _client_for(config), config
