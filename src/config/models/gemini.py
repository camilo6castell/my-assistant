"""
Configuration for Gemini models, via their OpenAI-compatible endpoint.

Same pattern as src/config/models/fastflowlm.py -- see that docstring
for the general rationale. Gemini (via this endpoint) does not expose an
equivalent to enable_thinking/think today, so supports_thinking() is
always False here.
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
        raise ValueError(f"Unsupported Gemini model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Context window in tokens, or None if not documented (see _CONTEXT_WINDOWS)."""
    _lookup(model_name)
    return _CONTEXT_WINDOWS.get(model_name)


def supports_thinking(model_name: str) -> bool:
    _lookup(model_name)  # validate that the model exists
    return False


def default_think(model_name: str) -> bool | None:
    _lookup(model_name)  # validate that the model exists
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
            f"Model '{model_name}' (Gemini) does not have a reasoning mode configured."
        )

    kwargs = deepcopy(_lookup(model_name))
    kwargs["model"] = model_name
    kwargs["messages"] = messages

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)

    return kwargs
