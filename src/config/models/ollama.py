from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from src.config.models import get_backend_data
from src.nlp.llm.backends.base import ChatTurn
from src.utils.logger import logger

_DATA = get_backend_data("ollama")


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return cast("dict[str, Any]", _DATA[model_name])
    except KeyError:
        raise ValueError(f"Unsupported Ollama model: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    _lookup(model_name)
    return cast("int | None", _DATA[model_name].get("context_window"))


def supports_thinking(model_name: str) -> bool:
    return "think" in _lookup(model_name)["kwargs"]


def default_think(model_name: str) -> bool | None:
    cfg = _lookup(model_name)["kwargs"]
    if "think" not in cfg:
        return None
    return bool(cfg["think"])


def supports_max_tokens(model_name: str) -> bool:
    return "num_predict" in _lookup(model_name)["kwargs"].get("options", {})


def max_tokens(model_name: str) -> int | None:
    options = _lookup(model_name)["kwargs"].get("options", {})
    return cast("int | None", options.get("num_predict"))


def build_kwargs(
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = deepcopy(cast("dict[str, Any]", _lookup(model_name)["kwargs"]))
    options = kwargs.setdefault("options", {})

    ctx = context_window(model_name)
    if ctx is not None:
        options["num_ctx"] = ctx

    if max_tokens is not None:
        options["num_predict"] = max_tokens

    if think is not None:
        if "think" not in kwargs:
            raise ValueError(f"Model '{model_name}' does not have a reasoning mode configured.")
        kwargs["think"] = think

    if extra:
        logger.warning(
            f"[config.models.ollama] 'extra' fields with no known mapping for "
            f"'{model_name}', ignored: {sorted(extra)}"
        )

    kwargs["model"] = model_name
    kwargs["messages"] = messages
    return kwargs
