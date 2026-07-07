"""
Consulta al LLM configurado, sin importar qué backend hay detrás.

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

think_mode se resuelve acá contra ModelCapabilities (src/config/models/)
-- este módulo traduce el bool lógico "pensar sí/no" al valor real que
el modelo espera (True/False, o un nivel "low"/"high"...) y arma un
CompletionRequest ya resuelto. El LLMClient (src/llm/backends/) que
efectivamente lo manda no necesita saber nada de esto.
"""

from __future__ import annotations

from src.chat.types import TurnMemory
from src.config.models import get_model_capabilities
from src.config.settings import settings
from src.llm.backends.base import ChatTurn, CompletionRequest
from src.llm.providers import ProviderConfig, get_client
from src.prompts.builder import build_reformulation_system_prompt, build_system_prompt
from src.utils.logger import logger

ExtraFields = dict[str, bool | str | int | float]


def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
    system_prompt: str = build_system_prompt(),
    max_turns: int | None = None,
) -> list[ChatTurn]:
    """
    Construye el array de mensajes para la API.

    Estructura:
      [system] -> instrucciones base
      [user / assistant] x max_turns -> historial reciente (ventana deslizante)
      [user] -> prompt actual (contexto recuperado + pregunta)

    chat_memory vacío (caso de ask_llm_internal, que no tiene turnos de
    usuario) simplemente no agrega nada entre system y el prompt actual.

    max_turns: override por-request (ver GenerationOptions en
    src/api/schemas/chat.py). None usa settings.max_turns -- mismo
    contrato que temperature/max_tokens, nunca muta settings.
    """
    effective_max_turns = settings.max_turns if max_turns is None else max_turns

    messages: list[ChatTurn] = [
        {"role": "system", "content": system_prompt},
    ]

    # TurnMemory es BaseModel: acceso por atributo (.user, .assistant)
    for turn in chat_memory[-effective_max_turns:]:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": prompt})

    return messages


def _resolve_think(
    think_mode: bool | None, config: ProviderConfig
) -> tuple[bool | str | None, str | None]:
    """
    Traduce el bool lógico "pensar sí/no" (o None = sin override) al
    valor real que este modelo espera, según ModelCapabilities.

    Devuelve (valor, nombre_del_campo). (None, None) si no hay nada que
    aplicar -- ni override, ni default configurado, ni ThinkMapping en
    absoluto para este modelo.

    Lanza ValueError si se pidió un valor explícito y el modelo no tiene
    ThinkMapping -- el router (src/api/routers/chat.py) ya valida esto
    contra `supports` antes de llegar acá, así que este error solo
    dispararía si algo llama a ask_llm() directamente sin pasar por la
    validación del endpoint.
    """
    caps = get_model_capabilities(config.capabilities, config.model)

    if caps.think is None:
        if think_mode is not None:
            raise ValueError(
                f"El modelo '{config.model}' (provider '{config.name}') no tiene "
                "modo de razonamiento configurado -- ver src/config/models/"
                f"{config.capabilities}.py."
            )
        return None, None

    # Sin override explícito, cae al default del modelo (ThinkMapping.default)
    # en vez de omitir el campo -- omitirlo deja que el modelo use SU
    # propio default (Qwen3 viene con thinking ON), que es justo lo que
    # hace imposible "apagarlo" si nunca se manda el campo explícitamente.
    effective = caps.think.default if think_mode is None else think_mode
    if effective is None:
        return None, None

    return (caps.think.on if effective else caps.think.off), caps.think.param


def _complete(
    *,
    messages: list[ChatTurn],
    provider_name: str,
    log_prefix: str,
    temperature: float | None,
    max_tokens: int | None,
    think_mode: bool | None,
    extra: ExtraFields | None,
) -> str | None:
    """
    Arma un CompletionRequest resuelto y se lo entrega al LLMClient del
    provider -- este módulo no sabe (ni necesita saber) si eso termina
    hablando con OpenAI, FastFlowLM u Ollama nativo.

    Devuelve None (nunca lanza) si el modelo respondió vacío o si la
    llamada falló -- cada función pública decide su propio fallback
    (ask_llm devuelve un mensaje de error visible al usuario;
    ask_llm_internal devuelve el prompt original sin cambios).
    """
    client, config = get_client(provider_name)
    caps = get_model_capabilities(config.capabilities, config.model)
    effective_temp = settings.llm_temperature if temperature is None else temperature

    think_value, think_param = _resolve_think(think_mode, config)

    extra_fields: ExtraFields = {}
    if think_param is not None and think_value is not None:
        extra_fields[think_param] = think_value
    if extra:
        extra_fields.update(extra)

    logger.info(
        f"{log_prefix}Consultando LLM | provider={provider_name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s"
        + (f" | temp={effective_temp}" if caps.supports_temperature else "")
        + (f" | max_tokens={max_tokens}" if max_tokens is not None else "")
        + (f" | think={think_value}" if think_value is not None else "")
    )

    request = CompletionRequest(
        messages=messages,
        model=config.model,
        timeout=settings.llm_timeout,
        temperature=effective_temp if caps.supports_temperature else None,
        max_tokens=max_tokens if caps.supports_max_tokens else None,
        extra_fields=extra_fields or None,
    )

    content = client.complete(request)
    if not content:
        logger.warning(f"{log_prefix}El modelo devolvio respuesta vacia.")
        return None

    # Log diagnóstico para el bug reportado de mensajes que llegan
    # incompletos al frontend (a veces faltan los primeros ~16
    # caracteres). No hay slicing en este módulo ni en el camino
    # request->JSONResponse->frontend (revisado), así que si el corte ya
    # está presente ACÁ (longitud/preview más corto de lo esperado, o el
    # preview empieza a mitad de palabra) el origen es el provider/modelo
    # o el cliente HTTP (OpenAI SDK / ollama-python), no este código. Si
    # el contenido llega completo hasta acá pero el frontend lo muestra
    # incompleto, el problema está en el tramo API->navegador (red,
    # proxy, o el estado de React), no en la generación.
    logger.info(
        f"{log_prefix}Respuesta del LLM | len={len(content)} "
        f"| preview={content[:40]!r}"
    )
    return content


def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    think_mode: bool | None = None,
    extra: ExtraFields | None = None,
    max_turns: int | None = None,
) -> str:
    provider_name = provider or settings.generate_provider
    messages = build_messages(prompt=prompt, chat_memory=chat_memory, max_turns=max_turns)

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
      - Usa el system prompt de reformulación en lugar del de RAG.
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
