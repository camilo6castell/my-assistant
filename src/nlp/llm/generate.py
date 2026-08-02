"""
Queries the configured LLM regardless of which backend is behind it.

Conversation history travels here as structured user/assistant messages
-- this is the only place where it is included. The prompt does not
contain a HISTORY block to avoid redundancy.

MAX_TURNS limits how many turns are sent to protect the model's
context window.

Generation overrides (max_tokens/think_mode/extra):
  These are *per-request* parameters that never mutate `settings` or
  ProviderConfig. If `ask_llm()`/`ask_llm_internal()` mutated a global
  Settings to apply an override, one conversation's change would leak to
  all concurrent conversations in the process -- a classic concurrency
  bug. When they come as None, the default from settings/.env is used;
  the full contract lives in GenerationOptions
  (src/api/schemas/chat.py).

  Temperature is NOT one of these overrides -- it is a fixed property
  of each model, defined in src/config/models/<backend>.py
  (_MODELS[model]["temperature"]) and never overwritten here. There
  previously existed settings.llm_temperature (.env) which was applied
  ALWAYS as a default when no override was provided, silently masking
  the model's real value (e.g. qwen3.5:4b configured with
  temperature=0.0 in _MODELS but still showing 0.2 because _complete()
  overwrote it before build_kwargs was reached). Removed on purpose: if
  a real per-request temperature override is ever needed, add it back
  here explicitly instead of reintroducing a silent default.

think_mode is validated here against `src.config.models.get_supports()`
(whether the active model has a reasoning mode or not) and is passed
as-is to `src.config.models.build_kwargs()`, which builds the final
dict already in the backend's native shape (extra_body for
OpenAI-compatible, `think` kwarg for Ollama). The LLMClient
(src/llm/backends/) that actually sends it does not need to know about
this mechanism. think_mode=None does NOT mean "off" -- it means "no
override", and build_kwargs() leaves intact what is already written in
_MODELS[model] for that model (see src/config/models/flm.py).
"""

from __future__ import annotations

from typing import Any

from src.config.models import build_kwargs, get_supports
from src.config.settings import settings
from src.domain.models import TurnMemory
from src.nlp.llm.backends.base import ChatTurn
from src.nlp.llm.providers import ProviderConfig, get_client_for, get_provider
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
    Builds the message array for the API.

    Structure:
      [system] -> base instructions
      [user / assistant] x max_turns -> recent history (sliding window)
      [user] -> current prompt (retrieved context + question)

    Empty chat_memory (the ask_llm_internal case, which has no user
    turns) simply adds nothing between system and the current prompt.

    system_prompt: None resolves to build_system_prompt() (default RAG
    behavior) at call time, not at module import -- previously this
    parameter had build_system_prompt() as a positional default,
    evaluated once when generate.py was loaded. This was harmless
    because build_system_prompt() is pure/deterministic, but it is the
    classic Python antipattern of a mutable/import-time-evaluated
    default, and it also prevented passing a different system prompt
    without changing the signature. ask_llm() now exposes that override
    (see its own docstring) for the case with no context source at all
    (_answer_raw() in src/api/routers/chat.py), which calls with
    system_prompt="" to omit the RAG system prompt entirely.

    max_turns: internal parameter, now without any caller that
    overrides it -- see GenerationOptions in src/api/schemas/chat.py,
    which no longer has this field (it became exclusively server-side
    configuration). None (the only value that arrives today) uses
    settings.max_turns. It is kept as a function parameter rather than
    removing it entirely because it remains a reasonable internal
    building block (sliding history window) that another internal
    caller might need to adjust without changing the signature; what was
    removed was the path that exposed it as a per-request API override.
    """
    effective_max_turns = settings.max_turns if max_turns is None else max_turns
    effective_system_prompt = system_prompt if system_prompt is not None else build_system_prompt()

    # system_prompt="" (distinct from None) means "no system prompt at
    # all" -- the _answer_raw() case in src/api/routers/chat.py, where
    # there is no context source and the user controls role/rules/task
    # entirely from their own message. An empty "system" message is not
    # sent: it is omitted entirely.
    messages: list[ChatTurn] = []
    if effective_system_prompt:
        messages.append({"role": "system", "content": effective_system_prompt})

    # TurnMemory is a BaseModel: attribute access (.user, .assistant)
    for turn in chat_memory[-effective_max_turns:]:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": prompt})

    return messages


def _validate_think(think_mode: bool | None, config: ProviderConfig) -> None:
    """
    If an explicit think_mode override was provided, validates that the
    active model supports it -- see `get_supports()` in
    src/config/models/.

    think_mode=None (no override) never needs validation: it means "use
    the default already written in _MODELS[model] for this model" (see
    src/config/models/flm.py -- Qwen3 already ships with
    enable_thinking=False as part of its base config, so "no override"
    never leaves the field unset).

    Raises ValueError if an explicit value was requested and the model
    does not support think_mode -- the router
    (src/api/routers/chat.py) already validates this against `supports`
    before reaching here, so this error would only fire if something
    calls ask_llm() directly without going through endpoint validation.
    """
    if think_mode is None:
        return

    supports = get_supports(config.capabilities, config.model)
    if "think_mode" not in supports:
        raise ValueError(
            f"Model '{config.model}' (provider '{config.name}') does not have "
            "a reasoning mode configured -- see src/config/models/"
            f"{config.capabilities}.py."
        )


def _dump_request_for_debug(kwargs: dict[str, Any]) -> None:
    """
    Writes the EXACT body sent to the provider (the same kwargs built
    by src.config.models.build_kwargs(), in that backend's native shape)
    to ./debug_last_llm_request.json at the project root -- ready for:

        curl -v --max-time 300 http://<base_url>/chat/completions \
          -H "Content-Type: application/json" \
          -d @debug_last_llm_request.json

    (For "ollama_native" backends the shape is not directly the HTTP
    body of /v1/chat/completions -- it is still useful for inspecting
    what was sent, even though the curl above only applies as-is to
    "openai_compat".)

    Overwritten on each call (only the last request matters) and only
    activated with settings.llm_debug_dump=True -- never runs in normal
    usage. Any write failure is logged and ignored: it is an optional
    diagnostic that should never break a real LLM query.
    """
    import json

    # `kwargs["messages"]` is list[ChatTurn] in practice (build_kwargs
    # always puts it there), but the declared type is dict[str, Any] --
    # narrowed here with isinstance instead of assumed, so the size
    # calculation never blows up if a new backend changes the shape.
    raw_messages = kwargs.get("messages", [])
    messages: list[dict[str, Any]] = raw_messages if isinstance(raw_messages, list) else []

    try:
        with open("debug_last_llm_request.json", "w", encoding="utf-8") as f:
            json.dump(kwargs, f, ensure_ascii=False, indent=2)
        chars = sum(len(m["content"]) for m in messages if isinstance(m, dict) and "content" in m)
        logger.info(
            f"[debug] Request dumped to ./debug_last_llm_request.json ({chars} message characters)"
        )
    except OSError as e:
        logger.warning(f"[debug] Could not dump request for debug: {e}")


def _complete_once(
    *,
    config: ProviderConfig,
    messages: list[ChatTurn],
    log_prefix: str,
    max_tokens: int | None,
    think_mode: bool | None,
    extra: ExtraFields | None,
) -> str | None:
    """
    Single attempt against one ProviderConfig (primary or fallback):
    builds the resolved kwargs via src.config.models.build_kwargs and
    hands them to that config's cached LLMClient -- this module does not
    know (nor need to know) whether that ends up talking to OpenAI,
    FastFlowLM, or native Ollama.

    Returns None if the model responded empty, raises if the call failed.
    """
    client = get_client_for(config)
    supports = get_supports(config.capabilities, config.model)

    kwargs = build_kwargs(
        config.capabilities,
        config.model,
        messages,
        max_tokens=max_tokens if "max_tokens" in supports else None,
        think=think_mode,
        extra=extra,
    )

    logger.info(
        f"{log_prefix}Querying LLM | provider={config.name} | model={config.model} "
        f"| timeout={settings.llm_timeout}s"
        + (f" | max_tokens={max_tokens}" if max_tokens is not None else "")
        + (f" | think={think_mode}" if think_mode is not None else "")
    )

    # Opt-in diagnostic (see settings.llm_debug_dump / LLM_DEBUG_DUMP in
    # .env): dumps the EXACT request sent to the provider to a file,
    # ready to reproduce with curl without guessing prompt size or
    # reconstructing kwargs by hand. Designed for cases like "local
    # model returns empty only with large RAG prompts" -- without this,
    # isolating whether it is prompt size/content requires reconstructing
    # the actual payload by eye.
    if settings.llm_debug_dump:
        _dump_request_for_debug(kwargs)

    return client.complete(kwargs)


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
    Runs the primary provider for `provider_name`; if it errors or
    responds empty AND a fallback is configured for that role
    (LLM_ROL_*_FALLBACK in .env.providers), retries once on the fallback
    -- a degraded path that drops the primary's think_mode override (the
    fallback model may not support it). If both fail, the original
    primary exception is re-raised (no silent swallow); if the primary
    failed with no fallback configured, it raises as before.

    Returns None (never raises) only when the model(s) responded empty
    -- each public function decides its own fallback (ask_llm returns an
    error message visible to the user; ask_llm_internal returns the
    original prompt unchanged).
    """
    config = get_provider(provider_name)
    _validate_think(think_mode, config)

    first_exc: Exception | None = None
    try:
        content = _complete_once(
            config=config,
            messages=messages,
            log_prefix=log_prefix,
            max_tokens=max_tokens,
            think_mode=think_mode,
            extra=extra,
        )
    except Exception as exc:
        first_exc = exc
        content = None

    if not content and config.fallback is not None:
        logger.warning(
            f"{log_prefix}Primary LLM {'errored' if first_exc else 'returned empty'} "
            f"(provider={config.name} | model={config.model})"
            + (f" | error={first_exc!r}" if first_exc is not None else "")
            + f" -- retrying on fallback model={config.fallback.model}"
        )
        try:
            content = _complete_once(
                config=config.fallback,
                messages=messages,
                log_prefix=f"{log_prefix}[fallback] ",
                max_tokens=max_tokens,
                think_mode=None,
                extra=extra,
            )
        except Exception as fallback_exc:
            logger.error(
                f"{log_prefix}Fallback LLM also failed (model={config.fallback.model}): "
                f"{fallback_exc!r}"
            )
            if first_exc is not None:
                raise first_exc from None
            raise fallback_exc from None
        if content:
            logger.warning(
                f"{log_prefix}FALLBACK USED -- answer came from "
                f"backend={config.fallback.backend} | model={config.fallback.model} "
                f"(primary backend={config.backend} | model={config.model} unavailable)"
            )
    elif not content and first_exc is not None:
        # No fallback configured: keep the previous behavior of
        # propagating the provider error.
        raise first_exc

    if not content:
        logger.warning(f"{log_prefix}Model returned empty response.")
        return None

    # Diagnostic log for the reported bug of messages arriving
    # incomplete at the frontend (sometimes missing the first ~16
    # characters). There is no slicing in this module or in the
    # request->JSONResponse->frontend path (reviewed), so if the
    # truncation is already present HERE (length/preview shorter than
    # expected, or preview starting mid-word), the source is the
    # provider/model or the HTTP client (OpenAI SDK / ollama-python),
    # not this code. If the content arrives complete here but the
    # frontend displays it incomplete, the problem is in the
    # API->browser segment (network, proxy, or React state), not in
    # generation.
    logger.info(f"{log_prefix}LLM response | len={len(content)} | preview={content[:40]!r}")
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
    provider is required and must always come from
    LLMRole.GENERATE.value -- see src/nlp/llm/roles.py. This module no
    longer picks a default on its own: the only place where "which model
    generates the response" is decided is Settings.role_spec(), so there
    is no second silent source of truth.

    system_prompt: optional system prompt override -- None uses
    build_system_prompt() (default RAG behavior, with grounding/citation
    against retrieved context). "" (empty string, distinct from None)
    omits the "system" message entirely -- see build_messages() above
    and _answer_raw() in src/api/routers/chat.py, the case with no
    active collection/ephemeral file/web_search: the user controls
    role/rules/task from their own message, with nothing from the
    server in between.
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

    return content if content is not None else "Model did not return a response."


def ask_llm_internal(
    system_prompt: str,
    prompt: str,
    provider: str,
    max_tokens: int | None = None,
) -> str | None:
    """
    LLM call for internal graph operations (e.g. query reformulation).
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
