"""
Configuration for FastFlowLM models (backend "flm" in .env.providers) --
an OpenAI-compatible backend.

_MODELS is the SINGLE source of truth: a dict per model with exactly the
kwargs that `client.chat.completions.create(**kwargs)` expects. To
tune the behavior of a specific model (temperature, top_p, whether it
thinks by default...) edit this file and this file only -- you never need
to touch build_kwargs() for that.

Everything here is pure functions over `model_name` + `_MODELS[model_name]`,
with no shared mutable state: build_kwargs() always deepcopies from
_MODELS before applying overrides, so two concurrent requests for the
same model can never stomp on each other's override (unlike the previous
design where a single FastFlowLMModelConfig per process mutated its own
_config via set_thinking()).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.nlp.llm.backends.base import ChatTurn

_MODELS: dict[str, dict[str, Any]] = {
    "qwen3.5:9b": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    },
    "gpt-oss:20b": {
        "timeout": 600,
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "qwen3.5:2b": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "enable_thinking": False,
        },
    },
}

_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen3.5:9b": 32_768,
    "gpt-oss:20b": 128_000,
    "qwen3.5:2b": 32_768,
}


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unsupported FastFlowLM model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Context window in tokens, or None if not documented (see _CONTEXT_WINDOWS)."""
    _lookup(model_name)  # validate that the model exists
    return _CONTEXT_WINDOWS.get(model_name)


def supports_thinking(model_name: str) -> bool:
    extra = _lookup(model_name).get("extra_body", {})
    if "enable_thinking" in extra:
        return True
    return "enable_thinking" in extra.get("chat_template_kwargs", {})


def default_think(model_name: str) -> bool | None:
    """enable_thinking value ALREADY written in _MODELS -- None if not applicable."""
    if not supports_thinking(model_name):
        return None
    extra = _lookup(model_name).get("extra_body", {})
    if "enable_thinking" in extra:
        return bool(extra["enable_thinking"])
    return bool(extra.get("chat_template_kwargs", {}).get("enable_thinking"))


def supports_max_tokens(model_name: str) -> bool:
    return "max_tokens" in _lookup(model_name)


def _set_thinking(extra_body: dict[str, Any], enabled: bool) -> None:
    """Mutate an extra_body ALREADY COPIED (see build_kwargs) -- never the original."""
    if "enable_thinking" in extra_body:
        extra_body["enable_thinking"] = enabled
    chat_kwargs = extra_body.get("chat_template_kwargs")
    if isinstance(chat_kwargs, dict) and "enable_thinking" in chat_kwargs:
        chat_kwargs["enable_thinking"] = enabled


def build_kwargs(
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble the dict ready for `client.chat.completions.create(**kwargs)`.

    Does not accept `temperature`: it always stays as written in
    _MODELS[model_name] -- there is no per-request override for that; it
    is this file's sole responsibility (edit the model entry here if it
    needs changing).

    The remaining `None` values mean "use whatever is already in
    _MODELS[model_name]" -- that is why, for example, qwen3.5:9b ships
    with thinking OFF by default without this module needing a separate
    "default" concept: it is already written in its _MODELS entry.

    Raises ValueError if `think` is requested for a model without
    reasoning support -- the caller (src/llm/generate.py) already
    validates this against `supports_thinking()` before reaching here,
    so under normal circulation this error should never fire.
    """
    kwargs = deepcopy(_lookup(model_name))
    kwargs["model"] = model_name
    kwargs["messages"] = messages

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if think is not None:
        if not supports_thinking(model_name):
            raise ValueError(f"Model '{model_name}' does not have a reasoning mode configured.")
        extra_body = kwargs.setdefault("extra_body", {})
        _set_thinking(extra_body, think)

    if extra:
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.update(extra)

    return kwargs
