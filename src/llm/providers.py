"""
Registro de proveedores LLM (arquitectura provider-per-node).

Problema que resuelve:
  El sistema original tenía un único `client` OpenAI-compatible global,
  apuntando a "lo que esté en .env" (local O gemini, nunca ambos a la vez).
  Para usar dos modelos distintos en el mismo grafo (uno reformula, otro
  genera) hace falta poder pedir "el cliente del proveedor X" en runtime.

Diseño:
  - Cada proveedor es (base_url, api_key, model, supports, think_param)
    -> un OpenAI client. Todo, incluyendo qué GenerationOptions acepta
    (supports) y cómo activa el modo de razonamiento (think_param), sale
    de settings/.env -- nada en este archivo menciona "FastFlowLM",
    "Ollama" ni "qwen3" por nombre. Cambiar de runtime o de modelo es
    editar .env.providers, no tocar código.
  - Los clientes se cachean (un client por proveedor, no por request).
  - Agregar un proveedor nuevo (Claude, OpenAI, Mistral...) es agregar
    una entrada a _PROVIDER_ENV_PREFIXES + los campos correspondientes
    en Settings, no tocar graph.py, nodes.py ni generate.py.
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
    # Qué GenerationOptions (src/api/schemas/chat.py) acepta este
    # provider/modelo. temperature y max_tokens son universales en
    # cualquier endpoint OpenAI-compatible; think_mode y extra NO -- ver
    # think_param abajo y el docstring de GenerationOptions.extra.
    supports: frozenset[str] = frozenset({"temperature", "max_tokens"})
    # Nombre del campo que este runtime espera en el payload para activar
    # el modo de razonamiento. None/"" = no soportado (y "think_mode" no
    # debería estar en `supports` en ese caso). El mecanismo exacto varía
    # por runtime -- por eso es data, no código: no hay forma de
    # introspectar automáticamente un servidor OpenAI-compatible
    # arbitrario para saber qué espera.
    think_param: str | None = None


# (nombre, prefijo en Settings) -- agregar un provider nuevo es agregar
# una tupla acá + los campos <prefijo>_base_url/_api_key/_model/_supports/
# _think_param en Settings (mismo patrón que local_*/gemini_*).
_PROVIDER_ENV_PREFIXES: tuple[str, ...] = ("local", "gemini")


def _parse_supports(raw: str) -> frozenset[str]:
    """'temperature,max_tokens,think_mode' -> frozenset(...), tolerante a espacios/vacíos."""
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


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
            supports=_parse_supports(getattr(settings, f"{prefix}_supports")),
            think_param=getattr(settings, f"{prefix}_think_param", "") or None,
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
