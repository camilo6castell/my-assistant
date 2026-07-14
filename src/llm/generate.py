"""
Consulta al LLM configurado, sin importar qué backend hay detrás.

El historial de conversacion viaja aqui como mensajes estructurados
user/assistant — es el unico lugar donde se incluye. El prompt no
contiene un bloque HISTORIAL para evitar redundancia.

MAX_TURNS limita cuantos turnos se envian para proteger la ventana
de contexto del modelo.

Overrides de generación (max_tokens/think_mode/extra):
  Son parámetros *por-request*, nunca mutan `settings` ni ProviderConfig.
  Si `ask_llm()`/`ask_llm_internal()` mutaran un Settings global para
  aplicar un override, el cambio de una conversación se filtraría a
  todas las conversaciones concurrentes del proceso — un bug de
  concurrencia clásico. Cuando vienen en None, se usa el default de
  settings/.env; el contrato completo vive en GenerationOptions
  (src/api/schemas/chat.py).

  La temperatura NO es uno de estos overrides -- es una propiedad fija
  de cada modelo, definida en src/config/models/<backend>.py
  (_MODELS[model]["temperature"]) y nunca pisada acá. Antes existía
  settings.llm_temperature (.env) que se aplicaba SIEMPRE como default
  cuando no venía override, tapando silenciosamente el valor real del
  modelo (ej. qwen3.5:9b configurado con temperature=0.0 en su
  _MODELS pero mostrando 0.2 igual, porque _complete() lo pisaba antes
  de llegar a build_kwargs). Eliminado a propósito: si algún día hace
  falta un override real de temperatura por-request, agregarlo de nuevo
  acá explícitamente en vez de reintroducir un default silencioso.

think_mode se valida acá contra `src.config.models.get_supports()` (el
modelo activo tiene o no tiene modo de razonamiento) y se lo pasa tal
cual a `src.config.models.build_kwargs()`, que arma el dict final ya en
el shape nativo del backend (extra_body para OpenAI-compatible, kwarg
`think` para Ollama). El LLMClient (src/llm/backends/) que efectivamente
lo manda no necesita saber nada de esa mecánica. think_mode=None NO es
"apagado" -- es "sin override", y build_kwargs() deja intacto lo que ya
esté escrito en _MODELS[model] para ese modelo (ver
src/config/models/fastflowlm.py).
"""

from __future__ import annotations

from typing import Any

from src.chat.types import TurnMemory
from src.config.models import build_kwargs, get_supports
from src.config.settings import settings
from src.llm.backends.base import ChatTurn
from src.llm.providers import ProviderConfig, get_client
from src.prompts.builder import build_system_prompt
from src.utils.logger import logger

ExtraFields = dict[str, bool | str | int | float]


def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
    system_prompt: str | None = None,
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

    system_prompt: None resuelve a build_system_prompt() (comportamiento
    RAG por defecto) en el momento de la llamada, no al importar el
    módulo -- antes este parámetro tenía build_system_prompt() como
    default posicional, evaluado una sola vez al cargar generate.py.
    Era inofensivo porque build_system_prompt() es puro/determinístico,
    pero es el antipattern clásico de default mutable/evaluado-al-
    importar en Python, y además impedía pasar un system prompt
    distinto sin tocar la firma. ask_llm() ahora expone ese override
    (ver su propio docstring) para el caso sin ninguna fuente de
    contexto (_answer_raw() en src/api/routers/chat.py), que llama con
    system_prompt="" para omitir el system prompt de RAG por completo.

    max_turns: override por-request (ver GenerationOptions en
    src/api/schemas/chat.py). None usa settings.max_turns -- mismo
    contrato que max_tokens, nunca muta settings.
    """
    effective_max_turns = settings.max_turns if max_turns is None else max_turns
    effective_system_prompt = system_prompt if system_prompt is not None else build_system_prompt()

    # system_prompt="" (distinto de None) es "sin system prompt en
    # absoluto" -- caso _answer_raw() en src/api/routers/chat.py, donde
    # no hay ninguna fuente de contexto y el usuario controla el rol/las
    # reglas/la tarea íntegramente desde su propio mensaje. No se manda
    # un mensaje "system" vacío: se omite del todo.
    messages: list[ChatTurn] = []
    if effective_system_prompt:
        messages.append({"role": "system", "content": effective_system_prompt})

    # TurnMemory es BaseModel: acceso por atributo (.user, .assistant)
    for turn in chat_memory[-effective_max_turns:]:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": prompt})

    return messages


def _validate_think(think_mode: bool | None, config: ProviderConfig) -> None:
    """
    Si vino un override explícito de think_mode, valida que el modelo
    activo lo soporte -- ver `get_supports()` en src/config/models/.

    think_mode=None (sin override) nunca necesita validación: significa
    "dejar el default que ya está escrito en _MODELS[model] para este
    modelo" (ver src/config/models/fastflowlm.py -- Qwen3 ya trae
    enable_thinking=False como parte de su config base, así que "sin
    override" nunca deja el campo sin mandar).

    Lanza ValueError si se pidió un valor explícito y el modelo no
    soporta think_mode -- el router (src/api/routers/chat.py) ya valida
    esto contra `supports` antes de llegar acá, así que este error solo
    dispararía si algo llama a ask_llm() directamente sin pasar por la
    validación del endpoint.
    """
    if think_mode is None:
        return

    supports = get_supports(config.capabilities, config.model)
    if "think_mode" not in supports:
        raise ValueError(
            f"El modelo '{config.model}' (provider '{config.name}') no tiene "
            "modo de razonamiento configurado -- ver src/config/models/"
            f"{config.capabilities}.py."
        )


def _dump_request_for_debug(kwargs: dict[str, Any]) -> None:
    """
    Escribe el body EXACTO que se le manda al provider (los mismos
    kwargs que arma src.config.models.build_kwargs(), en el shape nativo
    de ese backend) a ./debug_last_llm_request.json en la raíz del
    proyecto -- listo para:

        curl -v --max-time 300 http://<base_url>/chat/completions \\
          -H "Content-Type: application/json" \\
          -d @debug_last_llm_request.json

    (Para backends "ollama_native" el shape no es directamente el body
    HTTP de /v1/chat/completions -- sirve igual para inspeccionar qué se
    mandó, aunque el curl de arriba solo aplica tal cual a
    "openai_compat".)

    Se sobreescribe en cada llamada (solo importa el último request) y
    solo se activa con settings.llm_debug_dump=True -- nunca corre en
    uso normal. Cualquier fallo al escribir se loguea y se ignora: es un
    diagnóstico opcional, nunca debe romper una consulta real al LLM.
    """
    import json

    # `kwargs["messages"]` es list[ChatTurn] en la práctica (build_kwargs
    # siempre lo pone ahí), pero el tipo declarado es dict[str, Any] --
    # se narrowea acá con isinstance en vez de asumirlo, así el cálculo
    # de tamaño nunca revienta si algún backend nuevo cambia el shape.
    raw_messages = kwargs.get("messages", [])
    messages: list[dict[str, Any]] = raw_messages if isinstance(raw_messages, list) else []

    try:
        with open("debug_last_llm_request.json", "w", encoding="utf-8") as f:
            json.dump(kwargs, f, ensure_ascii=False, indent=2)
        chars = sum(len(m["content"]) for m in messages if isinstance(m, dict) and "content" in m)
        logger.info(
            "[debug] Request volcado a ./debug_last_llm_request.json "
            f"({chars} caracteres de mensajes)"
        )
    except OSError as e:
        logger.warning(f"[debug] No se pudo volcar el request para debug: {e}")


def _complete(
    *,
    messages: list[ChatTurn],
    provider_name: str,
    log_prefix: str,
    max_tokens: int | None,
    think_mode: bool | None,
    extra: ExtraFields | None,
) -> str | None:
    """
    Arma los kwargs resueltos (vía src.config.models.build_kwargs) y se
    los entrega al LLMClient del provider -- este módulo no sabe (ni
    necesita saber) si eso termina hablando con OpenAI, FastFlowLM u
    Ollama nativo.

    No recibe `temperature`: nunca se pasa un override acá, así que
    build_kwargs() siempre deja intacto el valor que ya está en
    _MODELS[model] para el modelo activo (ver docstring del módulo).

    Devuelve None (nunca lanza) si el modelo respondió vacío o si la
    llamada falló -- cada función pública decide su propio fallback
    (ask_llm devuelve un mensaje de error visible al usuario;
    ask_llm_internal devuelve el prompt original sin cambios).
    """
    client, config = get_client(provider_name)
    supports = get_supports(config.capabilities, config.model)

    _validate_think(think_mode, config)

    kwargs = build_kwargs(
        config.capabilities,
        config.model,
        messages,
        max_tokens=max_tokens if "max_tokens" in supports else None,
        think=think_mode,
        extra=extra,
    )

    logger.info(
        f"{log_prefix}Consultando LLM | provider={provider_name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s"
        + (f" | max_tokens={max_tokens}" if max_tokens is not None else "")
        + (f" | think={think_mode}" if think_mode is not None else "")
    )

    # Diagnóstico opt-in (ver settings.llm_debug_dump / LLM_DEBUG_DUMP en
    # .env): vuelca el request EXACTO que se le manda al provider a un
    # archivo, listo para reproducir con curl sin adivinar tamaño de
    # prompt ni reconstruir los kwargs a mano. Pensado para casos como
    # "el modelo local devuelve vacío solo con prompts de RAG grandes" --
    # sin esto, aislar si es tamaño/contenido del prompt implica
    # reconstruir el payload real a ojo.
    if settings.llm_debug_dump:
        _dump_request_for_debug(kwargs)

    content = client.complete(kwargs)
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
    logger.info(f"{log_prefix}Respuesta del LLM | len={len(content)} | preview={content[:40]!r}")
    return content


def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None = None,
    think_mode: bool | None = None,
    extra: ExtraFields | None = None,
    max_turns: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    provider es obligatorio y siempre debe venir de
    settings.provider_for(LLMRole.GENERATE) -- ver src/llm/roles.py. Este
    módulo ya no elige un default por su cuenta: el único lugar donde se
    decide "qué modelo genera la respuesta" es Settings.provider_for(),
    para que no haya una segunda fuente de verdad silenciosa.

    system_prompt: override opcional del system prompt -- None usa
    build_system_prompt() (comportamiento RAG por defecto, con
    grounding/citación contra contexto recuperado). "" (string vacío,
    distinto de None) omite el mensaje "system" por completo -- ver
    build_messages() más arriba y _answer_raw() en
    src/api/routers/chat.py, el caso sin ninguna colección/archivo
    efímero/web_search activo: el usuario controla rol/reglas/tarea
    desde su propio mensaje, sin nada del servidor de por medio.
    """
    messages = build_messages(
        prompt=prompt,
        chat_memory=chat_memory,
        system_prompt=system_prompt,
        max_turns=max_turns,
    )

    content = _complete(
        messages=messages,
        provider_name=provider,
        log_prefix="",
        max_tokens=max_tokens,
        think_mode=think_mode,
        extra=extra,
    )

    return content if content is not None else "El modelo no devolvio respuesta."


def ask_llm_internal(
    system_prompt: str,
    prompt: str,
    provider: str,
    max_tokens: int | None = None,
) -> str | None:
    """
    Llamada al LLM para operaciones internas del grafo (ej: reformulación de queries).
    """
    messages = build_messages(prompt=prompt, chat_memory=[], system_prompt=system_prompt)

    content = _complete(
        messages=messages,
        provider_name=provider,
        log_prefix="[internal] ",
        max_tokens=max_tokens,
        think_mode=None,
        extra=None,
    )

    return content

    # DECISIÓN (dejar comentado, no borrar): variante descartada de
    # ask_llm_internal con un parámetro could_be_none explícito en vez
    # del fallback fijo "devolver el prompt sin cambios". Se conserva
    # como referencia del patrón (fallback configurable por caller) por
    # si hace falta en otro internal call que no pueda usar el mismo
    # fallback que reformular query.
    # if content is not None:
    #     return content
    # if content is None and could_be_none:
    #     return None
    # if content is None and not could_be_none:
    #     return prompt


# DECISIÓN (dejar comentado, no borrar): se decidió no usar esta
# variante de ask_llm_internal en el flujo actual de web-supplement (ver
# WEB_SUPPLEMENT_SENTINEL en src/prompts/builder.py para el contexto
# completo), pero el patrón -- devolver None en un fallo en vez de hacer
# eco del prompt, para callers donde el prompt es scaffolding interno y
# no algo seguro de mostrarle al usuario -- es reutilizable. Se conserva
# como referencia de implementación en vez de borrarla.
# def ask_llm_supplement(
#     prompt: str,
#     system_prompt: str,
#     provider: str,
#     max_tokens: int | None = None,
# ) -> str | None:
#     """
#     Variante de ask_llm_internal() para tareas internas de una sola
#     pasada donde un fallo NO debe hacer eco del prompt como fallback.

#     Por qué esto necesita existir aparte de ask_llm_internal(): ahí el
#     fallback "devolver el prompt sin cambios" es seguro porque el prompt
#     ES la pregunta del usuario (reformular query). Acá el prompt es
#     scaffolding interno arbitrario (ej. la respuesta ya generada +
#     fragmentos de una búsqueda web, ver build_web_supplement_prompt() en
#     src/prompts/builder.py) -- si el modelo falla y este caller hiciera
#     el mismo fallback, ese scaffolding completo terminaría pegado en la
#     respuesta que ve el usuario. Por eso devuelve None en vez de prompt:
#     fuerza al caller a decidir explícitamente qué hacer ante un fallo
#     (típicamente: omitir el complemento en silencio, ver
#     src/api/routers/chat.py).

#     provider es obligatorio -- debe venir de
#     settings.provider_for(LLMRole.WEB_SUPPLEMENT).
#     """
#     messages = build_messages(
#         prompt=prompt, chat_memory=[], system_prompt=system_prompt
#     )

#     return _complete(
#         messages=messages,
#         provider_name=provider,
#         log_prefix="[internal:web_supplement] ",
#         max_tokens=max_tokens,
#         think_mode=None,
#         extra=None,
#     )
