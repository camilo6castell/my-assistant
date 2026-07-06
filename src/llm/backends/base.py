"""
Contrato común entre implementaciones de cliente LLM.

generate.py arma un CompletionRequest ya resuelto (temperatura efectiva,
extra_fields con think ya traducido según ModelCapabilities) y se lo
entrega a cualquier LLMClient sin saber qué hay detrás -- OpenAI SDK,
cliente nativo de Ollama, lo que sea. Cada implementación decide cómo
mapear `extra_fields` a su propio transporte; generate.py nunca conoce
esa mecánica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    messages: list[ChatTurn]
    model: str
    timeout: float
    temperature: float | None = None
    max_tokens: int | None = None
    # Campos ya resueltos por generate.py contra ModelCapabilities (ej.
    # {"think": False} o {"think": "low"}), más lo que el cliente haya
    # mandado en GenerationOptions.extra. El backend solo sabe DÓNDE
    # meter cada clave en su propio protocolo -- nunca decide si el
    # modelo lo soporta, eso ya se resolvió antes de llegar acá.
    extra_fields: dict[str, bool | str | int | float] | None = None


class LLMClient(Protocol):
    """Cualquier forma de completar un chat: OpenAI-compatible, Ollama nativo, etc."""

    def complete(self, request: CompletionRequest) -> str | None:
        """Devuelve el texto de la respuesta, o None si vino vacía."""
        ...
