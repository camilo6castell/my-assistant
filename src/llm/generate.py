"""
Consulta al LLM local via cliente OpenAI-compatible.

El historial de conversacion viaja aqui como mensajes estructurados
user/assistant — es el unico lugar donde se incluye. El prompt no
contiene un bloque HISTORIAL para evitar redundancia.

MAX_TURNS limita cuantos turnos se envian para proteger la ventana
de contexto del modelo.
"""

from __future__ import annotations

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from src.chat.types import TurnMemory
from src.config.settings import settings
from src.llm.providers import get_client
from src.utils.logger import logger

SYSTEM_PROMPT: str = """
Eres un asistente RAG especializado en responder
usando unicamente el contexto proporcionado.

Reglas:

- Prioriza el contenido recuperado.
- No inventes informacion.
- Si el contexto no contiene suficiente informacion,
  dilo explicitamente.
- Responde de forma clara y estructurada.
- Cuando sea posible, conecta conceptos relacionados.
"""


def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
) -> list[ChatCompletionMessageParam]:
    """
    Construye el array de mensajes para la API.

    Estructura:
      [system] -> instrucciones base
      [user / assistant] x MAX_TURNS -> historial reciente (ventana deslizante)
      [user] -> prompt actual (contexto recuperado + pregunta)
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # TurnMemory es BaseModel: acceso por atributo (.user, .assistant)
    for turn in chat_memory[-settings.max_turns :]:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": prompt})

    return messages


def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str | None = None,
) -> str:

    provider_name = provider or settings.generate_provider
    client, config = get_client(provider_name)

    logger.info(
        f"Consultando LLM | provider={provider_name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s | temp={settings.llm_temperature}"
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=build_messages(prompt=prompt, chat_memory=chat_memory),
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

        content = response.choices[0].message.content

        if not content:
            logger.warning("El modelo devolvio respuesta vacia.")
            return "El modelo no devolvio respuesta."

        return str(content).strip()

    except OpenAIError as e:
        logger.exception("Error consultando LLM")
        return f"Error consultando modelo: {e}"


REFORMULATION_SYSTEM_PROMPT: str = """
Eres un experto en recuperación de información semántica (RAG).
Tu única tarea es reformular preguntas para mejorar el recall en búsquedas vectoriales.
No respondas la pregunta. No añadas explicaciones. Devuelve solo la pregunta reformulada.
"""


def ask_llm_internal(prompt: str, provider: str | None = None) -> str:
    """
    Llamada al LLM para operaciones internas del grafo (ej: reformulación de queries).

    Diferencias respecto a ask_llm():
      - Usa REFORMULATION_SYSTEM_PROMPT en lugar del system prompt RAG.
      - No acepta chat_memory: las operaciones internas no son turnos del usuario.
      - El log distingue la llamada como "[internal]" para facilitar el debug.
      - provider por defecto = settings.reformulate_provider (gemini),
        independiente del proveedor que use generate_node.
    """
    provider_name = provider or settings.reformulate_provider
    client, config = get_client(provider_name)

    logger.info(
        f"[internal] Consultando LLM | provider={provider_name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s | temp={settings.llm_temperature}"
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

        content = response.choices[0].message.content

        if not content:
            logger.warning("[internal] El modelo devolvio respuesta vacia.")
            return prompt  # fallback: devuelve la pregunta original sin cambios

        return str(content).strip()

    except OpenAIError as e:
        logger.exception("[internal] Error consultando LLM")
        return prompt  # fallback: devuelve la pregunta original sin cambios
