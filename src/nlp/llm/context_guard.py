"""
Guard preventivo de límite de contexto -- corre ANTES de llamar al LLM,
tanto en /query (src/api/routers/chat.py) como en /task/query
(src/api/routers/task.py).

No trunca ni reescribe nada automáticamente: si el request no entra,
lanza ContextLimitExceeded con el detalle exacto para que el router lo
traduzca a un 413 y el frontend se lo muestre al usuario ANTES de
esperar una respuesta que iba a fallar (o peor, a devolver una
respuesta cortada a mitad de frase porque el modelo se quedó sin
espacio para terminarla).
"""

from __future__ import annotations

from src.cli.types import TurnMemory
from src.config.models import get_context_window
from src.nlp.llm.providers import get_client
from src.utils.tokens import estimate_tokens

# Margen de seguridad sobre el estimado de tokens -- tiktoken/heurística
# nunca va a matchear el tokenizador real exacto del modelo activo; este
# colchón absorbe ese margen de error sin bloquear requests que en la
# práctica entrarían igual.
SAFETY_MARGIN_RATIO = 0.10

# Tokens reservados para la respuesta cuando el request no especifica
# max_tokens explícito -- piso conservador para no dejar al modelo sin
# espacio para responder aunque el prompt entre justo en la ventana.
DEFAULT_OUTPUT_RESERVE = 1024


class ContextLimitExceeded(Exception):
    """
    Se lanza cuando el request estimado no entra en la ventana de
    contexto del modelo activo. El router la captura y la traduce a un
    HTTPException 413 vía as_detail().
    """

    def __init__(self, *, estimated_tokens: int, limit: int, model: str) -> None:
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.model = model
        super().__init__(
            f"El request estimado ({estimated_tokens} tokens) supera el límite "
            f"utilizable ({limit} tokens) del modelo '{model}'."
        )

    def as_detail(self) -> dict[str, int | str]:
        return {
            "error": "context_limit_exceeded",
            "estimated_tokens": self.estimated_tokens,
            "limit": self.limit,
            "model": self.model,
        }


def check_context_fit(
    *,
    system_prompt: str,
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None,
) -> None:
    """
    Estima el tamaño total del request (system + historial + prompt) y
    lo compara contra la ventana de contexto del modelo activo, menos lo
    reservado para la respuesta. Lanza ContextLimitExceeded si no entra;
    no devuelve nada si entra (o si el modelo activo no tiene
    context_window documentado -- ver más abajo).

    Fail-open: si get_context_window() devuelve None (el modelo activo
    todavía no tiene su ventana de contexto completada en
    src/config/models/<backend>.py), esta función no bloquea nada --
    preferible a romper requests por un dato de configuración
    incompleto. Completar ese valor es lo que activa el guard para ese
    modelo.
    """
    _, config = get_client(provider)
    limit = get_context_window(config.capabilities, config.model)
    if limit is None:
        return

    text_parts = [system_prompt, prompt]
    for turn in chat_memory:
        text_parts.append(turn.user)
        text_parts.append(turn.assistant)

    estimated = sum(estimate_tokens(part) for part in text_parts)
    estimated = int(estimated * (1 + SAFETY_MARGIN_RATIO))

    reserve = max_tokens or DEFAULT_OUTPUT_RESERVE
    usable = limit - reserve

    if estimated > usable:
        raise ContextLimitExceeded(estimated_tokens=estimated, limit=usable, model=config.model)
