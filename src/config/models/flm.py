from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from src.config.models import get_backend_data
from src.nlp.llm.backends.base import ChatTurn

_DATA = get_backend_data("flm")


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return cast("dict[str, Any]", _DATA[model_name])
    except KeyError:
        raise ValueError(f"Unsupported FastFlowLM model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    _lookup(model_name)
    return cast("int | None", _DATA[model_name].get("context_window"))


def supports_thinking(model_name: str) -> bool:
    extra = _lookup(model_name)["kwargs"].get("extra_body", {})
    if "enable_thinking" in extra:
        return True
    return "enable_thinking" in extra.get("chat_template_kwargs", {})


def default_think(model_name: str) -> bool | None:
    if not supports_thinking(model_name):
        return None
    extra = _lookup(model_name)["kwargs"].get("extra_body", {})
    if "enable_thinking" in extra:
        return bool(extra["enable_thinking"])
    return bool(extra.get("chat_template_kwargs", {}).get("enable_thinking"))


def supports_max_tokens(model_name: str) -> bool:
    return "max_tokens" in _lookup(model_name)["kwargs"]


def max_tokens(model_name: str) -> int | None:
    kwargs = _lookup(model_name)["kwargs"]
    return kwargs.get("max_tokens")


def _set_thinking(extra_body: dict[str, Any], enabled: bool) -> None:
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
    kwargs = deepcopy(cast("dict[str, Any]", _lookup(model_name)["kwargs"]))
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
