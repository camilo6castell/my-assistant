"""
Contrato común entre implementaciones de cliente LLM.

generate.py arma el dict de kwargs YA RESUELTO (model/messages/
max_tokens/think/extra ya traducidos al shape que ese backend espera --
la temperatura no pasa por acá, viaja fija dentro de ese mismo dict
desde _MODELS[model], ver src/config/models/build_kwargs()) y se lo
entrega a cualquier LLMClient sin saber qué hay detrás -- OpenAI SDK,
cliente nativo de Ollama, lo que sea. Cada implementación solo tiene que
desempacar ese dict contra su propio SDK (`**kwargs`); no vuelve a tocar
la mecánica de qué campo significa qué.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict


class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    """Cualquier forma de completar un chat: OpenAI-compatible, Ollama nativo, etc."""

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        """
        `kwargs` ya viene armado por src.config.models.build_kwargs() en
        el shape nativo de este backend. Devuelve el texto de la
        respuesta, o None si vino vacía.
        """
        ...
