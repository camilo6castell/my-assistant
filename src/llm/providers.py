"""
Registro de proveedores LLM (arquitectura provider-per-node).

Problema que resuelve:
  El sistema original tenía un único `client` OpenAI-compatible global,
  apuntando a "lo que esté en .env" (local O gemini, nunca ambos a la vez).
  Para usar dos modelos distintos en el mismo grafo (uno reformula, otro
  genera) hace falta poder pedir "el cliente del proveedor X" en runtime.

Diseño:
  - Cada proveedor es solo (base_url, api_key, model) -> un OpenAI client.
  - Los clientes se cachean (un client por proveedor, no por request).
  - Agregar un proveedor nuevo (Claude, OpenAI, Mistral...) es agregar
    una entrada al dict `_PROVIDERS`, no tocar graph.py ni nodes.py.
  - Los nodos del grafo no importan providers.py directamente: usan
    ask_llm(..., provider=...) / ask_llm_internal(..., provider=...) en
    generate.py, que sí depende de este módulo. Así el grafo nunca sabe
    de OpenAI clients, solo de "qué proveedor usar".

Por qué no LangChain ChatModels aquí: el resto del proyecto ya habla
directo con el SDK `openai` (ver llm/client.py original). Mantener esa
misma interfaz para los providers nuevos evita una migración innecesaria
y dos formas distintas de llamar al LLM conviviendo en el código.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from src.config.settings import settings
from src.utils.logger import logger


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


def _build_provider_table() -> dict[str, ProviderConfig]:
    """
    Lee la configuración de cada proveedor desde settings.

    Se construye en una función (no a nivel de módulo) para que los
    valores reflejen settings en el momento de la llamada, no en el
    momento del import — útil en tests que mutan settings.
    """
    return {
        "local": ProviderConfig(
            name="local",
            base_url=settings.local_base_url,
            api_key=settings.local_api_key,
            model=settings.local_model,
        ),
        "gemini": ProviderConfig(
            name="gemini",
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        ),
        # Para agregar un proveedor nuevo:
        # "claude": ProviderConfig(
        #     name="claude",
        #     base_url=settings.claude_base_url,
        #     api_key=settings.claude_api_key,
        #     model=settings.claude_model,
        # ),
    }


@lru_cache(maxsize=None)
def _client_for(base_url: str, api_key: str) -> OpenAI:
    """Un OpenAI client por (base_url, api_key) único — evita recrear conexiones."""
    logger.info(f"[providers] Inicializando cliente LLM | base_url={base_url}")
    return OpenAI(base_url=base_url, api_key=api_key)


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


def get_client(name: str) -> tuple[OpenAI, ProviderConfig]:
    """Devuelve (client, config) listos para usar en chat.completions.create()."""
    config = get_provider(name)
    client = _client_for(config.base_url, config.api_key)
    return client, config
