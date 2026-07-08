"""
Cliente para cualquier servidor OpenAI-compatible (FastFlowLM, Gemini vía
su endpoint compat, y en general cualquier runtime que hable el protocolo
de /v1/chat/completions).
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI, OpenAIError

from src.llm.backends.base import CompletionRequest
from src.utils.logger import logger


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, request: CompletionRequest) -> str | None:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "timeout": request.timeout,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        # extra_body inyecta claves top-level adicionales en el JSON del
        # request -- así es como se le pasan campos propios de un runtime
        # (ej. "think") que el SDK de OpenAI no conoce nativamente.
        if request.extra_fields:
            kwargs["extra_body"] = request.extra_fields

        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError:
            logger.exception("[openai_compat] Error consultando LLM")
            return None

        # Algunos runtimes OpenAI-compatible (ej. FastFlowLM sobre NPU)
        # pueden devolver HTTP 200 con un body sin 'choices' cuando el
        # proceso de inferencia falla internamente a mitad de generación
        # -- no es un error que el SDK de OpenAI reconozca como tal (no
        # es un OpenAIError), así que indexar choices[0] a ciegas
        # crashea con un TypeError sin manejar y tumba el request entero
        # con un 500. Tratarlo como respuesta vacía es consistente con
        # el resto del contrato de complete() (ver base.py: "None si
        # vino vacía") y deja que el caller haga el fallback normal
        # ("El modelo no devolvio respuesta.", ver generate.py) en vez
        # de propagar una excepción no manejada hasta el endpoint.
        if not response.choices:
            # Antes solo logueábamos que faltaba 'choices', sin decir por
            # qué -- eso deja a ciegas justo el caso que más hace falta
            # diagnosticar (ver conversación: FastFlowLM devolviendo esto
            # consistentemente para un prompt de RAG, sin lanzar ningún
            # error HTTP). El SDK de OpenAI modela la respuesta como un
            # objeto pydantic, así que volcamos el JSON completo -- puede
            # traer 'usage' (¿completion_tokens=0? sugiere que el server
            # cortó la generación antes de emitir texto), un 'error'
            # embebido que el runtime metió fuera del schema estándar, o
            # finish_reason -- cualquiera de esos acota mucho más la causa
            # que "no vinieron choices".
            try:
                raw_body = response.model_dump_json()
            except Exception:
                raw_body = repr(response)
            logger.warning(
                "[openai_compat] La respuesta del servidor no trae 'choices' "
                "(posible fallo interno del runtime, ej. timeout o abortado a "
                "mitad de generación) -- se trata como respuesta vacia. "
                f"Body completo: {raw_body[:2000]}"
            )
            return None

        content = response.choices[0].message.content
        return str(content).strip() if content else None
