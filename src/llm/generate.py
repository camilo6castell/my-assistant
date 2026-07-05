"""
Consulta al LLM local via cliente OpenAI-compatible.

El historial de conversacion viaja aqui como mensajes estructurados
user/assistant — es el unico lugar donde se incluye. El prompt no
contiene un bloque HISTORIAL para evitar redundancia.

MAX_TURNS limita cuantos turnos se envian para proteger la ventana
de contexto del modelo.

Overrides de generación (temperature/max_tokens/think_mode/extra):
  Son parámetros *por-request*, nunca mutan `settings` ni ProviderConfig.
  Si `ask_llm()`/`ask_llm_internal()` mutaran un Settings global para
  aplicar un override, el cambio de una conversación se filtraría a
  todas las conversaciones concurrentes del proceso — un bug de
  concurrencia clásico. Cuando vienen en None, se usa el default de
  settings/.env; el contrato completo vive en GenerationOptions
  (src/api/schemas/chat.py).

think_mode es genérico a propósito: el nombre del campo que cada runtime
espera en el payload (ej. "think" en FastFlowLM/Ollama) es
ProviderConfig.think_param, configurable por .env (LOCAL_THINK_PARAM,
GEMINI_THINK_PARAM...) -- este módulo no conoce ni le importa qué
runtime hay detrás de cada provider.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from src.chat.types import TurnMemory
from src.config.settings import settings
from src.llm.providers import ProviderConfig, get_client
from src.utils.logger import logger
from src.prompts.builder import build_system_prompt, build_reformulation_system_prompt


def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
    system_prompt: str = build_system_prompt(),
) -> list[ChatCompletionMessageParam]:
    """
    Construye el array de mensajes para la API.

    Estructura:
      [system] -> instrucciones base
      [user / assistant] x MAX_TURNS -> historial reciente (ventana deslizante)
      [user] -> prompt actual (contexto recuperado + pregunta)

    chat_memory vacío (caso de ask_llm_internal, que no tiene turnos de
    usuario) simplemente no agrega nada entre system y el prompt actual.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
    ]

    # TurnMemory es BaseModel: acceso por atributo (.user, .assistant)
    for turn in chat_memory[-settings.max_turns :]:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": prompt})

    return messages


def _apply_think_mode(
    extra_body: dict[str, Any], think_mode: bool, config: ProviderConfig
) -> None:
    """
    Activa/desactiva el modo de razonamiento usando el nombre de campo
    declarado por el provider (config.think_param), sin conocer nada
    sobre el runtime real detrás de él.

    El router (src/api/routers/chat.py) ya valida think_mode contra
    ProviderConfig.supports antes de llegar acá, así que este error solo
    dispararía si algo llama a ask_llm()/ask_llm_internal() directamente
    (ej. un script) pasando think_mode para un provider mal configurado.
    """
    if not config.think_param:
        raise ValueError(
            f"El provider '{config.name}' no tiene think_param configurado "
            f"-- ver {config.name.upper()}_THINK_PARAM en .env.providers."
        )
    # extra_body[config.think_param] = think_mode


def _complete(
    *,
    messages: list[ChatCompletionMessageParam],
    provider_name: str,
    log_prefix: str,
    temperature: float | None,
    max_tokens: int | None,
    think_mode: bool | None,
    extra: dict[str, Any] | None,
) -> str | None:
    """
    Llamada compartida a chat.completions.create().

    Devuelve None (nunca lanza) si el modelo respondió vacío o si la
    llamada falló con un OpenAIError -- cada función pública decide su
    propio fallback (ask_llm devuelve un mensaje de error visible al
    usuario; ask_llm_internal devuelve el prompt original sin cambios).
    """
    client, config = get_client(provider_name)
    effective_temp = settings.llm_temperature if temperature is None else temperature

    logger.info(
        f"{log_prefix}Consultando LLM | provider={provider_name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s | temp={effective_temp}"
        + (f" | max_tokens={max_tokens}" if max_tokens is not None else "")
        + (f" | think_mode={think_mode}" if think_mode is not None else "")
    )

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": effective_temp,
        "timeout": settings.llm_timeout,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # extra_body inyecta claves top-level adicionales en el JSON del
    # request -- así es como se le pasan campos propios de un runtime
    # (ej. "think") que el SDK de OpenAI no conoce nativamente.
    extra_body: dict[str, Any] = {}
    if think_mode is not None:
        _apply_think_mode(extra_body, think_mode, config)
    if extra:
        extra_body.update(extra)
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if not content:
            logger.warning(f"{log_prefix}El modelo devolvio respuesta vacia.")
            return None

        return str(content).strip()

    except OpenAIError:
        logger.exception(f"{log_prefix}Error consultando LLM")
        return None


def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    think_mode: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    provider_name = provider or settings.generate_provider
    messages = build_messages(prompt=prompt, chat_memory=chat_memory)

    content = _complete(
        messages=messages,
        provider_name=provider_name,
        log_prefix="",
        temperature=temperature,
        max_tokens=max_tokens,
        think_mode=think_mode,
        extra=extra,
    )

    return content if content is not None else "El modelo no devolvio respuesta."


def ask_llm_internal(
    prompt: str,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Llamada al LLM para operaciones internas del grafo (ej: reformulación de queries).

    Diferencias respecto a ask_llm():
      - Usa REFORMULATION_SYSTEM_PROMPT en lugar del system prompt RAG.
      - No acepta chat_memory: las operaciones internas no son turnos del usuario.
      - El log distingue la llamada como "[internal]" para facilitar el debug.
      - provider por defecto = settings.reformulate_provider (gemini),
        independiente del proveedor que use generate_node.
      - No acepta think_mode/extra: las tareas internas (reformular,
        revisar) son de una sola pasada sobre texto corto, no se
        benefician de razonamiento extendido ni de parámetros avanzados.
    """
    provider_name = provider or settings.reformulate_provider
    messages = build_messages(
        prompt=prompt, chat_memory=[], system_prompt=build_reformulation_system_prompt()
    )

    content = _complete(
        messages=messages,
        provider_name=provider_name,
        log_prefix="[internal] ",
        temperature=temperature,
        max_tokens=max_tokens,
        think_mode=None,
        extra=None,
    )

    # Fallback: devuelve el prompt original sin cambios si el modelo
    # falló o respondió vacío -- mejor no reformular que romper el flujo.
    return content if content is not None else prompt
