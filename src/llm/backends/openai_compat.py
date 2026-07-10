"""
Cliente para cualquier servidor OpenAI-compatible (FastFlowLM, Gemini vía
su endpoint compat, y en general cualquier runtime que hable el protocolo
de /v1/chat/completions).
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI, OpenAIError

from src.utils.logger import logger


class OpenAICompatClient:
    """
    Transporte puro: no sabe nada de modelos concretos ni de
    FastFlowLM/Gemini en particular -- sirve a cualquier provider cuyo
    ProviderConfig.client sea "openai_compat" (ver src/llm/providers.py).
    `kwargs` ya viene armado por src.config.models.build_kwargs() con el
    modelo, los mensajes, y cualquier override ya resuelto.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
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
