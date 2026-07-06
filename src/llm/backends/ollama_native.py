"""
Cliente nativo para Ollama, usando el paquete oficial `ollama`
(pip install ollama) en vez de su endpoint OpenAI-compatible.

Por qué nativo y no OpenAI-compat: el endpoint /v1/chat/completions de
Ollama tiene soporte inconsistente y no del todo documentado para
`think`/`reasoning_effort` (hay un issue abierto del proyecto donde
think=true directamente no aplica para varios modelos). El cliente
nativo (`ollama.Client().chat(..., think=...)`) sí lo soporta de forma
directa y tipada -- ver ollama._types.ChatResponse en el paquete.

El import de `ollama` es perezoso (dentro de __init__, no a nivel de
módulo): así, alguien que solo usa providers "openai_compat" (FastFlowLM,
Gemini) nunca necesita tener el paquete `ollama` instalado -- es una
dependencia opcional, no una obligación de todo el proyecto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from src.llm.backends.base import CompletionRequest
from src.utils.logger import logger

if TYPE_CHECKING:
    import ollama

_ThinkParam = bool | Literal["low", "medium", "high"] | None


class OllamaNativeClient:
    def __init__(self, host: str) -> None:
        import ollama  # lazy: ver docstring del módulo

        self._client = ollama.Client(host=host)

    def complete(self, request: CompletionRequest) -> str | None:
        options: dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        extra = dict(request.extra_fields or {})
        # "think" es el único campo que el cliente nativo de Ollama
        # entiende como kwarg propio (bool o nivel "low"/"medium"/"high")
        # -- lo sacamos de extra_fields y lo pasamos posta, no como un
        # passthrough genérico. Cualquier otra clave en extra_fields no
        # tiene un lugar conocido en este cliente y se ignora con un
        # warning en vez de fallar silenciosamente.
        think = cast(_ThinkParam, extra.pop("think", None))
        if extra:
            logger.warning(
                f"[ollama_native] Campos extra sin mapeo conocido, se ignoran: "
                f"{sorted(extra)}"
            )

        try:
            response = self._client.chat(
                model=request.model,
                messages=request.messages,
                think=think,
                options=options or None,
            )
            content = response.message.content
            return content.strip() if content else None
        except Exception:
            # A diferencia de openai-python, ollama-python no envuelve
            # errores de transporte (servidor caído, timeout) en su
            # propia jerarquía -- ResponseError/RequestError no cubren
            # esos casos. Capturar amplio acá mantiene el mismo contrato
            # que OpenAICompatClient: nunca lanza, el caller decide el
            # fallback.
            logger.exception("[ollama_native] Error consultando LLM")
            return None
