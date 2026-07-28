"""
Schemas for the "chat" domain -- the application's regular functions:
ask a question, list available collections.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    """
    Optional generation overrides for a single call to
    POST /query or POST /query/agent.

    Temperature is NOT one of these fields -- it is the exclusive
    responsibility of each file in src/config/models/ (_MODELS[model]),
    just like top_p, presence_penalty, etc. It is not a per-request
    override: it is a model property, edited in its backend JSON and
    applied to all calls to that model. If at some point a real
    per-request temperature override needs to be exposed again,
    see the history of this file before reinventing the wheel (the
    previous design with Settings.llm_temperature silently overwrote
    the _MODELS value on every request; see src/llm/generate.py).

    think_mode: named field because it is a common concept across
    most reasoning provider/model combos (designed for the "Think"
    button in the UI). None means "use the default already written
    in _MODELS[model] for this model" -- not "use 0.2 from a .env"
    as happened before with temperature.

    extra: generic escape hatch for any parameter that the backend
    does NOT explicitly model (top_p, presence_penalty, a custom flag
    from a new provider...). It is sent as-is to the provider via the
    OpenAI client's extra_body -- without validating its content,
    because by definition it can be anything a specific provider
    understands. That is why it requires the provider to declare
    "extra" in its `supports` (see ProviderConfig in
    src/llm/providers.py): it is an explicit opt-in, not "any JSON
    passes to any server".

    Does not include max_turns or top_k_initial/top_k_final: the chat
    history window and retrieval (see src/retrieval/search.py) have
    become exclusively server-side configuration (settings.max_turns,
    settings.soft_top_k_*/hard_top_k_*, see .env) -- there is no
    per-request override for either. If they ever need to be
    re-exposed, see the history of this file before reinventing the
    wheel: they existed here and were deliberately removed, not omitted
    by accident.

    None of these values mutate Settings or ProviderConfig -- they are
    per-request parameters, not server state (see the docstring of
    src/llm/generate.py for why). The server validates each field
    against ProviderConfig.supports before using it and returns 400 if
    the active provider does not support it.
    """

    think_mode: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    extra: dict[str, Any] | None = None


class WebSource(BaseModel):
    """Web source cited in a response (see QueryResponse.web_sources)."""

    title: str
    url: str


class QueryRequest(BaseModel):
    """
    Payload for POST /query and POST /query/agent.

    collections:    tokens with the same syntax as the CLI, e.g.
                    ["sociologia", "psicoanalisis/Freud_Suenos"].
    mode:           "SOFT" (default) | "HARD"
    chat_history:   previous conversation turns. Each turn is a dict
                    {"user": "...", "assistant": "..."}. The client is
                    responsible for maintaining and re-sending the history
                    -- the API is stateless by design.
    conversation_id: opaque id that the client generates once per
                    conversation (e.g. crypto.randomUUID() in the
                    frontend). Required if the conversation has
                    ephemeral files (POST /api/v1/files) or ad-hoc
                    attachments (POST /api/v1/attachments) -- can be
                    omitted if neither is present.
    generation:     optional temperature/tokens/think mode overrides.
    web_search:     if True, complements (or replaces, if there are no
                    collections/files) the context with a web search
                    via Tavily -- see _run_web_search in
                    src/api/routers/chat.py for details on the two
                    modes. Requires settings.web_search_enabled=True;
                    otherwise the router returns 400. Best-effort: if
                    the web search fails and there ARE collections/files,
                    the request does not fail because of it (see
                    QueryResponse.used_web_search).

    Without `collections`, without ephemeral files, and with
    web_search=False: this is NOT an error (unlike a previous version
    of this schema that required at least one context source). The
    router responds directly with the LLM, without any system prompt or
    retrieval -- see _answer_raw() in src/api/routers/chat.py. This is
    the expected mode for standalone questions or for attaching a
    single file (POST /api/v1/attachments) without any selected
    collection: the user is responsible for giving the model its
    role/rules/task in their own message.
    """

    question: str = Field(..., min_length=1)
    collections: list[str] = Field(default_factory=list)
    mode: str = Field(default="SOFT", pattern="^(SOFT|HARD)$")
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str | None = None
    generation: GenerationOptions | None = None
    web_search: bool = False


class QueryResponse(BaseModel):
    """
    Response for POST /query and POST /query/agent.

    reformulated:     only relevant in /query/agent. True if the graph
                       reformulated the query before generating.
    used_web_search:   what ACTUALLY happened, not what was requested --
                       False if web_search=True was requested but the
                       search returned no usable results (see
                       src/retrieval/web_search.py), even though the rest
                       of the response was still generated with the
                       locally available context.
    web_sources:       web sources actually used (title + URL), for the
                       frontend to display as citations. None if
                       used_web_search is False.
    web_search_quota_exceeded: True if Tavily returned that the account
                       quota was exhausted (free tier or other plan) --
                       a distinct signal from "no results" so the
                       frontend can warn the user and disable the web
                       search button instead of failing silently on
                       every subsequent message.
    """

    answer: str
    confidence: float
    collections_used: list[str]
    reformulated: bool = False
    used_web_search: bool = False
    web_sources: list[WebSource] | None = None
    web_search_quota_exceeded: bool = False


class CollectionsResponse(BaseModel):
    """Response for GET /api/v1/collections."""

    collections: list[str]
