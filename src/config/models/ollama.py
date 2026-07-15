"""
Configuración de modelos servidos vía el cliente nativo de Ollama.

Mismo patrón que src/config/models/fastflowlm.py -- ver ese docstring
para el razonamiento general. La diferencia es de FORMA, no de
filosofía: acá build_kwargs() arma el dict que espera
`ollama.Client().chat(**kwargs)` (model/messages/think/options), en vez
del shape OpenAI-compatible -- cada backend decide su propio shape de
salida, sus callers (src/llm/backends/*.py) no necesitan traducir nada.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.nlp.llm.backends.base import ChatTurn
from src.utils.logger import logger

_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-r1": {
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
            "top_p": 0.95,
            "top_k": 20,
        },
        "think": False,
    },
}

_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-r1": 32_768,
}


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f"Modelo Ollama no soportado: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Ventana de contexto en tokens, o None si no está documentada (ver _CONTEXT_WINDOWS)."""
    _lookup(model_name)
    return _CONTEXT_WINDOWS.get(model_name)


def supports_thinking(model_name: str) -> bool:
    return "think" in _lookup(model_name)


def default_think(model_name: str) -> bool | None:
    cfg = _lookup(model_name)
    if "think" not in cfg:
        return None
    return bool(cfg["think"])


def supports_max_tokens(model_name: str) -> bool:
    return "num_predict" in _lookup(model_name).get("options", {})


def build_kwargs(
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = deepcopy(_lookup(model_name))
    options = kwargs.setdefault("options", {})

    if max_tokens is not None:
        options["num_predict"] = max_tokens

    if think is not None:
        if "think" not in kwargs:
            raise ValueError(f"El modelo '{model_name}' no tiene modo de razonamiento configurado.")
        kwargs["think"] = think

    if extra:
        # El cliente nativo de Ollama no tiene un passthrough genérico
        # tipo extra_body -- solo entiende kwargs propios (model,
        # messages, think, options, format...). Nunca hubo un lugar
        # conocido donde meter claves arbitrarias, así que se ignoran
        # con warning, igual que hacía antes OllamaNativeClient.
        logger.warning(
            f"[config.models.ollama] Campos 'extra' sin mapeo conocido para "
            f"'{model_name}', se ignoran: {sorted(extra)}"
        )

    kwargs["model"] = model_name
    kwargs["messages"] = messages
    return kwargs
