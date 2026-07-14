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

from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    pass


class OllamaNativeClient:
    """
    Transporte puro, igual que OpenAICompatClient -- no decide qué
    significa cada campo, solo desempaca `kwargs` (ya armado por
    src.config.models.ollama.build_kwargs(), shape
    {model, messages, think, options}) contra `ollama.Client().chat()`.
    """

    def __init__(self, host: str) -> None:
        import ollama  # lazy: ver docstring del módulo

        self._client = ollama.Client(host=host)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        try:
            response = self._client.chat(**kwargs)
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
