"""
Configuración de modelos FastFlowLM -- un backend OpenAI-compatible.

_MODELS es la ÚNICA fuente de verdad: un dict por modelo con exactamente
los kwargs que espera `client.chat.completions.create(**kwargs)`. Para
tocar el comportamiento de un modelo puntual (temperatura, top_p, si
piensa por defecto...) se edita acá, nada más -- build_kwargs() nunca
hace falta tocarlo para eso.

Todo acá son funciones puras sobre `model_name` + `_MODELS[model_name]`,
sin estado mutable compartido: build_kwargs() siempre deepcopy-ea desde
_MODELS antes de aplicar overrides, así que dos requests concurrentes
para el mismo modelo nunca pueden pisarse un override del otro (a
diferencia del diseño anterior, donde un único FastFlowLMModelConfig por
proceso mutaba su propio _config con set_thinking()).
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


# Ventana de contexto por modelo, en tokens -- usada por el guard de
# contexto (src/llm/context_guard.py). Deliberadamente SEPARADA de
# _MODELS: build_kwargs() hace deepcopy(_lookup(model_name)) y lo
# entrega casi tal cual como kwargs al SDK, así que meter
# "context_window" ahí lo filtraría al payload real y rompería la
# llamada con un kwarg desconocido.
#
# TODO(Alejandro): confirmar estos valores contra tu build real de
# FastFlowLM -- son el contexto NOMINAL de cada modelo base, pero
# FastFlowLM en NPU puede correr con una ventana efectiva menor según
# cómo hayas compilado/cuantizado el modelo. Hasta confirmarlos, el
# guard usa estos placeholders.
_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen3.5:9b": 32_768,
    "gpt-oss:20b": 32_768,
    "qwen3.5:2b": 32_768,
}


def _lookup(model_name: str) -> dict[str, Any]:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f"Modelo FastFlowLM no soportado: {model_name!r}") from None


def context_window(model_name: str) -> int | None:
    """Ventana de contexto en tokens, o None si no está documentada (ver _CONTEXT_WINDOWS)."""
    _lookup(model_name)  # valida que el modelo exista
    return _CONTEXT_WINDOWS.get(model_name)


def supports_thinking(model_name: str) -> bool:
    extra = _lookup(model_name).get("extra_body", {})
    if "enable_thinking" in extra:
        return True
    return "enable_thinking" in extra.get("chat_template_kwargs", {})


def default_think(model_name: str) -> bool | None:
    """Valor de enable_thinking YA escrito en _MODELS -- None si no aplica."""
    if not supports_thinking(model_name):
        return None
    extra = _lookup(model_name).get("extra_body", {})
    if "enable_thinking" in extra:
        return bool(extra["enable_thinking"])
    return bool(extra.get("chat_template_kwargs", {}).get("enable_thinking"))


def supports_max_tokens(model_name: str) -> bool:
    return "max_tokens" in _lookup(model_name)


def _set_thinking(extra_body: dict[str, Any], enabled: bool) -> None:
    """Muta un extra_body YA COPIADO (ver build_kwargs) -- nunca el original."""
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
    Arma el dict listo para `client.chat.completions.create(**kwargs)`.

    No acepta `temperature`: siempre queda tal cual está en
    _MODELS[model_name] -- no hay override por-request para eso, es
    responsabilidad exclusiva de este archivo (editar la entrada del
    modelo acá si hace falta cambiarla).

    Los demás `None` significan "usar lo que ya está en
    _MODELS[model_name]" -- por eso, por ejemplo, qwen3.5:9b viene con
    thinking OFF por defecto sin que este módulo necesite un concepto
    separado de "default": ya está escrito en su entrada de _MODELS.

    Lanza ValueError si se pide `think` para un modelo sin soporte de
    razonamiento -- el caller (src/llm/generate.py) ya valida esto contra
    `supports_thinking()` antes de llegar acá, así que en circulación
    normal este error nunca debería dispararse.
    """
    kwargs = deepcopy(_lookup(model_name))
    kwargs["model"] = model_name
    kwargs["messages"] = messages

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if think is not None:
        if not supports_thinking(model_name):
            raise ValueError(f"El modelo '{model_name}' no tiene modo de razonamiento configurado.")
        extra_body = kwargs.setdefault("extra_body", {})
        _set_thinking(extra_body, think)

    if extra:
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.update(extra)

    return kwargs
