"""
Configuración de modelos Gemini, vía su endpoint OpenAI-compatible.

Mismo patrón que src/config/models/fastflowlm.py -- ver ese docstring
para el razonamiento general. Gemini (vía este endpoint) no expone hoy
un equivalente a enable_thinking/think, así que supports_thinking()
siempre es False acá.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.nlp.llm.backends.base import ChatTurn

_MODELS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash-lite": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 0.95,
    },
}


_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.5-flash-lite": 1_000_000,
}


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f"Modelo Gemini no soportado: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Ventana de contexto en tokens, o None si no está documentada (ver _CONTEXT_WINDOWS)."""
    _lookup(model_name)
    return _CONTEXT_WINDOWS.get(model_name)


def supports_thinking(model_name: str) -> bool:
    _lookup(model_name)  # valida que el modelo exista
    return False


def default_think(model_name: str) -> bool | None:
    _lookup(model_name)  # valida que el modelo exista
    return None


def supports_max_tokens(model_name: str) -> bool:
    return "max_tokens" in _lookup(model_name)


def build_kwargs(
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if think is not None:
        raise ValueError(
            f"El modelo '{model_name}' (Gemini) no tiene modo de razonamiento configurado."
        )

    kwargs = deepcopy(_lookup(model_name))
    kwargs["model"] = model_name
    kwargs["messages"] = messages

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)

    return kwargs
