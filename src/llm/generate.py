"""
Consulta al LLM local vía cliente OpenAI-compatible.
Timeout y temperatura se leen de settings/env.
"""

from __future__ import annotations

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from src.chat.types import TurnMemory
from src.config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
from src.llm.client import client
from src.utils.logger import logger

SYSTEM_PROMPT: str = """
Eres un asistente RAG especializado en responder
usando únicamente el contexto proporcionado.

Reglas:

- Prioriza el contenido recuperado.
- No inventes información.
- Si el contexto no contiene suficiente información,
  dilo explícitamente.
- Responde de forma clara y estructurada.
- Cuando sea posible, conecta conceptos relacionados.
"""


def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
) -> list[ChatCompletionMessageParam]:

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for turn in chat_memory:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": prompt})

    return messages


def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
) -> str:

    logger.info(
        f"Consultando LLM | model={LLM_MODEL} "
        f"| timeout={LLM_TIMEOUT}s | temp={LLM_TEMPERATURE}"
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=build_messages(prompt=prompt, chat_memory=chat_memory),
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )

        content = response.choices[0].message.content

        if not content:
            logger.warning("El modelo devolvió respuesta vacía.")
            return "El modelo no devolvió respuesta."

        return content.strip()

    except OpenAIError as e:
        logger.exception("Error consultando LLM")
        return f"Error consultando modelo: {e}"
