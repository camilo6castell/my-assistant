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
            content = response.choices[0].message.content
            return str(content).strip() if content else None
        except OpenAIError:
            logger.exception("[openai_compat] Error consultando LLM")
            return None
