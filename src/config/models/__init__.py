"""
Per-model capabilities registry — single source of truth is `models.json`
in this directory, shared with the TypeScript frontend.

Each backend (flm, ollama, gemini) has its own module that reads from the
JSON and provides builder/query functions. This module is only a
DISPATCHER — it does not know the shape of any kwargs dict.

Adding a new model = an entry in models.json. Adding a new backend =
a new file here + a new key in models.json + a new line in _registry().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

from src.nlp.llm.backends.base import ChatTurn

_MODELS_JSON_PATH = Path(__file__).resolve().parent / "models.json"


def _load_models_data() -> dict[str, Any]:
    with open(_MODELS_JSON_PATH) as f:
        return cast("dict[str, Any]", json.load(f))


_MODELS_DATA: dict[str, Any] = _load_models_data()


def get_backend_data(backend: str) -> dict[str, Any]:
    """Return the entire models dict for a given backend."""
    try:
        return cast("dict[str, Any]", _MODELS_DATA[backend])
    except KeyError:
        valid = ", ".join(sorted(_MODELS_DATA))
        raise ValueError(f"Unknown capabilities backend: {backend!r}. Valid: {valid}") from None


class _ModelBackend(Protocol):
    """
    Structural protocol that each backend file
    (flm.py/gemini.py/ollama.py) must satisfy to register here --
    a module with these functions matches this Protocol without needing
    to inherit anything or an explicit cast.
    """

    def build_kwargs(
        self,
        model_name: str,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = None,
        think: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def supports_thinking(self, model_name: str) -> bool: ...
    def default_think(self, model_name: str) -> bool | None: ...
    def supports_max_tokens(self, model_name: str) -> bool: ...
    def context_window(self, model_name: str) -> int | None: ...


def _registry() -> dict[str, _ModelBackend]:
    from src.config.models import flm, gemini, ollama

    return {
        "flm": flm,
        "ollama": ollama,
        "gemini": gemini,
    }


def _module(capabilities_key: str) -> _ModelBackend:
    registry = _registry()
    try:
        return registry[capabilities_key]
    except KeyError:
        valid = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown capabilities backend: {capabilities_key!r}. Valid: {valid}"
        ) from None


def build_kwargs(
    capabilities_key: str,
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _module(capabilities_key).build_kwargs(
        model_name,
        messages,
        max_tokens=max_tokens,
        think=think,
        extra=extra,
    )


def get_supports(capabilities_key: str, model_name: str) -> frozenset[str]:
    mod = _module(capabilities_key)
    names: set[str] = set()
    if mod.supports_max_tokens(model_name):
        names.add("max_tokens")
    if mod.supports_thinking(model_name):
        names.add("think_mode")
    names.add("extra")
    return frozenset(names)


def get_context_window(capabilities_key: str, model_name: str) -> int | None:
    return _module(capabilities_key).context_window(model_name)


def get_default_think(capabilities_key: str, model_name: str) -> bool | None:
    return _module(capabilities_key).default_think(model_name)
