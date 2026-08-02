"""
Preventive context-limit guard -- runs BEFORE calling the LLM, both in
/query (src/api/routers/chat.py) and in the LangGraph generate_node
(src/graph/nodes.py).

It does not truncate or rewrite anything automatically: if the request
does not fit, it raises ContextLimitExceeded with the exact details so
the router can translate it to a 413 and the frontend can show it to
the user BEFORE waiting for a response that would have failed (or
worse, returned a response cut off mid-sentence because the model ran
out of space to finish it).
"""

from __future__ import annotations

from src.config.models import get_context_window
from src.config.settings import settings
from src.domain.models import TurnMemory
from src.nlp.llm.providers import get_provider
from src.utils.tokens import estimate_tokens

# Safety margin over the token estimate -- tiktoken/heuristic will
# never match the active model's exact real tokenizer; this cushion
# absorbs that error margin without blocking requests that would fit in
# practice anyway.
SAFETY_MARGIN_RATIO = settings.context_guard_safety_margin

# Tokens reserved for the response when the request does not specify
# max_tokens explicitly -- a conservative floor so the model is never
# left without space to answer even if the prompt fits exactly in the
# window.
DEFAULT_OUTPUT_RESERVE = settings.context_guard_output_reserve


class ContextLimitExceeded(Exception):
    """
    Raised when the estimated request does not fit in the active model's
    context window. The router catches it and translates it to an
    HTTPException 413 via as_detail().
    """

    def __init__(self, *, estimated_tokens: int, limit: int, model: str) -> None:
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.model = model
        super().__init__(
            f"Estimated request ({estimated_tokens} tokens) exceeds the usable "
            f"limit ({limit} tokens) of model '{model}'."
        )

    def as_detail(self) -> dict[str, int | str]:
        return {
            "error": "context_limit_exceeded",
            "estimated_tokens": self.estimated_tokens,
            "limit": self.limit,
            "model": self.model,
        }


def check_context_fit(
    *,
    system_prompt: str,
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None,
) -> None:
    """
    Estimates the total request size (system + history + prompt) and
    compares it against the active model's context window minus the
    reserve for the response. Raises ContextLimitExceeded if it does not
    fit; returns nothing if it fits (or if the active model has no
    documented context_window -- see below).

    Fail-open: if get_context_window() returns None (the active model
    still has no context window completed in
    src/config/models/<backend>.py), this function blocks nothing --
    preferable to breaking requests over incomplete configuration data.
    Filling in that value is what activates the guard for that model.
    """
    config = get_provider(provider)
    # NOTE: get_provider, NOT get_client -- this guard only needs the
    # model's context_window (config.capabilities/config.model), never a
    # live connection. get_client would CONSTRUCT the LLM client here,
    # which requires valid credentials: with e.g. GEMINI_API_KEY unset,
    # the OpenAI SDK raises at construction time and a token-estimation
    # check would 500 the request before the LLM fallback in
    # src/nlp/llm/generate.py ever ran. The guard must stay fail-open on
    # transport concerns; only generate.py should care about them.
    limit = get_context_window(config.capabilities, config.model)
    if limit is None:
        return

    text_parts = [system_prompt, prompt]
    for turn in chat_memory:
        text_parts.append(turn.user)
        text_parts.append(turn.assistant)

    estimated = sum(estimate_tokens(part) for part in text_parts)
    estimated = int(estimated * (1 + SAFETY_MARGIN_RATIO))

    reserve = max_tokens or DEFAULT_OUTPUT_RESERVE
    usable = limit - reserve

    if estimated > usable:
        raise ContextLimitExceeded(estimated_tokens=estimated, limit=usable, model=config.model)
