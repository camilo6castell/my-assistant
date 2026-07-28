"""
Configuration for models served via the native Ollama client.

Same pattern as src/config/models/fastflowlm.py -- see that docstring
for the general rationale. The difference is in form, not philosophy:
here build_kwargs() assembles the dict that `ollama.Client().chat(**kwargs)`
expects (model/messages/think/options), instead of the OpenAI-compatible
shape -- each backend decides its own output shape, and its callers
(src/llm/backends/*.py) do not need to translate anything.
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
    "qwen3.5:2b": {
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
            "top_p": 1.0,
            "top_k": 20,
        },
        "think": False,
    },
}

_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-r1": 32_768,
    "qwen3.5:2b": 32_768,
}


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unsupported Ollama model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Context window in tokens, or None if not documented (see _CONTEXT_WINDOWS)."""
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
            raise ValueError(f"Model '{model_name}' does not have a reasoning mode configured.")
        kwargs["think"] = think

    if extra:
        # The native Ollama client has no generic passthrough like
        # extra_body -- it only understands its own kwargs (model,
        # messages, think, options, format...). There was never a known
        # place to put arbitrary keys, so they are ignored with a
        # warning, same as OllamaNativeClient did before.
        logger.warning(
            f"[config.models.ollama] 'extra' fields with no known mapping for "
            f"'{model_name}', ignored: {sorted(extra)}"
        )

    kwargs["model"] = model_name
    kwargs["messages"] = messages
    return kwargs
