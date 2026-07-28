"""
Per-model capabilities registry -- replaces the old
`ProviderConfig.supports`/`think_param`/`default_think` (plain strings
in .env) with a JSON dict per model, one per backend, in its own file:

    src/config/models/flm.py
    src/config/models/ollama.py
    src/config/models/gemini.py

The filename/key of each backend here ("flm", "ollama", "gemini") maps
1:1 to the backend alias used in .env.providers
(LLM_FLM_URL, EMBEDDER_OLLAMA_URL, LLM_ROL_GENERATE=flm,..., EMBEDDER=
ollama,..., etc. -- see src/config/settings.py) and the `capabilities`
of each ProviderConfig in src/nlp/llm/providers.py. This is intentional:
the same string identifies the backend across all three layers, so adding
a new backend never requires inventing a different alias per layer.

Why this and not .env:
  Previously it was possible to declare "think_mode" in LOCAL_SUPPORTS
  and forget to configure LOCAL_THINK_PARAM -- they would drift out of
  sync because they were two independent strings. Here `get_supports()`
  is DERIVED directly from `_MODELS[model]` (the same structure
  build_kwargs() uses to assemble the real request), so it cannot drift:
  if a model has no thinking field in its dict, "think_mode" simply does
  not appear in `supports`.

  Same criterion for temperature: it is NOT a per-request override
  (build_kwargs() does not accept a `temperature` parameter) -- it is a
  fixed property of _MODELS[model]["temperature"] in the backend's file.
  To change it, edit that file; there is no other place (.env,
  GenerationOptions, or a UI slider) that can override it.

Adding a new model = an entry in the `_MODELS` dict of that backend's
file. Adding a new backend = a new file here (with the same 4 functions:
build_kwargs, supports_thinking, default_think, supports_max_tokens) +
a new line in `_registry()` + a new client in src/llm/backends/ --
you never need to touch generate.py, providers.py, or the routers.

This module is only a DISPATCHER to the correct backend file -- it does
not know the shape of any kwargs dict, nor does it mutate anything. Each
function here simply forwards to `<backend>.<same_function>(model_name, ...)`.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.nlp.llm.backends.base import ChatTurn


class _ModelBackend(Protocol):
    """
    Structural protocol that each backend file
    (fastflowlm.py/gemini.py/ollama.py) must satisfy to register here --
    a module with these functions matches this Protocol without needing
    to inherit anything or an explicit cast (structural typing: mypy
    compares the module's real signature against this). It gives real
    typing to `_module(...).build_kwargs(...)` instead of degrading to
    `Any` the way returning `ModuleType` would.
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
    # Lazy import (inside the function) so that adding a new file in
    # this folder requires touching nothing here except this line.
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
    """
    Assemble the kwargs dict READY to pass to the backend SDK
    (`**kwargs` directly, no further transformation in
    src/llm/backends/*.py). The exact shape is decided by each backend
    module -- OpenAI-compatible (fastflowlm/gemini) returns flat kwargs;
    ollama returns {model, messages, think, options}.

    Does not accept `temperature` on purpose: it always stays as written
    in _MODELS[model_name]["temperature"] in the backend file -- there is
    no per-request override for that (see module docstring).
    """
    return _module(capabilities_key).build_kwargs(
        model_name,
        messages,
        max_tokens=max_tokens,
        think=think,
        extra=extra,
    )


def get_supports(capabilities_key: str, model_name: str) -> frozenset[str]:
    """
    GenerationOptions that this model accepts -- used by
    GET /api/v1/config/providers (which controls to show in the UI) and
    by request validation in the chat router.

    Does not include "temperature": it is not a GenerationOption, it is
    a fixed property of the model (see module docstring).

    Conservative fallback (empty frozenset) if the backend or model is
    not registered, instead of a 500 -- it is logged as a normal
    exception higher up the stack if the caller does not expect it; here
    it is enough not to break provider discovery over a misconfigured
    model.
    """
    mod = _module(capabilities_key)
    names: set[str] = set()
    if mod.supports_max_tokens(model_name):
        names.add("max_tokens")
    if mod.supports_thinking(model_name):
        names.add("think_mode")
    # "extra" (passthrough) is always offered as opt-in -- each backend
    # decides what to do with it (fastflowlm/gemini merge it into
    # extra_body, ollama ignores it with a warning; see build_kwargs in
    # each module).
    names.add("extra")
    return frozenset(names)


def get_context_window(capabilities_key: str, model_name: str) -> int | None:
    """
    Context window (in tokens) of the active model, or None if not
    documented in its backend file -- in that case
    check_context_fit() (src/llm/context_guard.py) does not block the
    request, it just lets it through without checking (fail-open:
    preferable to breaking the app for a model with that data not yet
    filled in).
    """
    return _module(capabilities_key).context_window(model_name)


def get_default_think(capabilities_key: str, model_name: str) -> bool | None:
    """
    The thinking value ALREADY written in _MODELS[model_name] for this
    model -- None if the model has no reasoning mode at all. Used by
    GET /api/v1/config/providers so the "Think" button in the UI starts
    reflecting the model's real behavior (see ProviderInfo.default_think)
    instead of starting in a fixed state unrelated to the actual config.
    """
    return _module(capabilities_key).default_think(model_name)
