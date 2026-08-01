from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from src.config.models import get_backend_data
from src.nlp.llm.backends.base import ChatTurn

_DATA = get_backend_data("gemini")


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return cast("dict[str, Any]", _DATA[model_name])
    except KeyError:
        raise ValueError(f"Unsupported Gemini model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    _lookup(model_name)
    return cast("int | None", _DATA[model_name].get("context_window"))


def supports_thinking(model_name: str) -> bool:
    _lookup(model_name)
    return False


def default_think(model_name: str) -> bool | None:
    _lookup(model_name)
    return None


def supports_max_tokens(model_name: str) -> bool:
    return "max_tokens" in _lookup(model_name)["kwargs"]


def max_tokens(model_name: str) -> int | None:
    kwargs = _lookup(model_name)["kwargs"]
    return cast("int | None", kwargs.get("max_tokens"))


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

    kwargs = deepcopy(cast("dict[str, Any]", _lookup(model_name)["kwargs"]))
    kwargs["model"] = model_name
    kwargs["messages"] = messages

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)

    return kwargs
