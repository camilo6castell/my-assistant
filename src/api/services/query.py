"""
Query service -- business logic for /query and /query/agent endpoints.

Extracted from src/api/routers/chat.py to separate HTTP concerns from
domain logic. The router handles validation and response formatting;
this service handles the actual RAG pipeline execution.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.api.schemas.chat import GenerationOptions, QueryRequest, WebSource
from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager, LoadedCollection
from src.domain.models import LLMRole, TurnMemory
from src.nlp.llm.context_guard import check_context_fit
from src.nlp.llm.generate import ask_llm, ask_llm_internal
from src.prompts.builder import (
    build_prompt,
    build_web_supplement_prompt,
    build_web_supplement_system_prompt,
    inject_attachments,
)
from src.retrieval.web_search import WebSearchResult, WebSearchStatus, search_web
from src.utils.logger import logger

# ======================================================
# HELPERS
# ======================================================


def resolve_collections(tokens: list[str]) -> list[LoadedCollection]:
    """
    Loads the persisted collections requested in the request.

    Each request loads its own collections from disk via a local
    ContextManager -- not the deps.py singleton, which is only used for
    listing (GET /collections). Empty tokens (from a request that only
    uses ephemeral files) return an empty list without error.

    Raises ValueError if any token does not match a persisted collection.
    """
    if not tokens:
        return []

    cm = ContextManager()
    not_found: list[str] = []

    for token in tokens:
        matches = cm.activate(token)
        if not matches:
            not_found.append(token)

    if not_found:
        raise ValueError(f"Collections not found: {not_found}")

    return cm.get_loaded_collections()


def resolve_ephemeral(
    conversation_id: str | None, store: EphemeralStore
) -> LoadedCollection | None:
    """Ephemeral collection for the conversation, if it has uploaded files."""
    if conversation_id is None:
        return None
    return store.get_collection(conversation_id)


def build_chat_memory(history: list[dict[str, str]]) -> list[TurnMemory]:
    """
    Converts the request's chat_history into TurnMemory for the LLM.

    Silently ignores malformed entries (missing 'user' or 'assistant')
    so that partial client history does not break the request.
    """
    memory: list[TurnMemory] = []
    for turn in history:
        if "user" in turn and "assistant" in turn:
            memory.append(TurnMemory(user=turn["user"], assistant=turn["assistant"]))
    return memory


def fetch_attachments(conversation_id: str | None, store: AttachmentStore) -> list[tuple[str, str]]:
    """Pending ad-hoc attachments for this conversation (see src/context/attachments.py)."""
    return store.list_contents(conversation_id)


def consume_attachments(conversation_id: str | None, store: AttachmentStore) -> None:
    """
    Deletes the ad-hoc attachments for a conversation after processing a
    query -- they are single-use (see AttachmentStore docstring). Called
    at the end of /query and /query/agent, in all three branches
    (raw/web-only/RAG) equally: attachments apply to any of them.
    No-op if conversation_id is None or there were no attachments.
    """
    if conversation_id is not None:
        store.remove_conversation(conversation_id)


def _web_context_chunks(results: list[WebSearchResult]) -> list[str]:
    """
    Same format as local chunks (SOURCE/.../text) so that build_prompt()
    is identical in both cases.
    """
    return [f"SOURCE: {r.title}\nURL: {r.url}\n\n{r.content}" for r in results]


def _web_sources_from_results(results: list[WebSearchResult]) -> list[WebSource]:
    return [WebSource(title=r.title, url=r.url) for r in results]


def no_context_detail(web_status: WebSearchStatus | None) -> dict[str, Any] | str:
    """
    Builds the `detail` for the 422 "no context source" response. If the
    web was the only possible source and failed specifically due to
    exhausted quota, returns a structured dict with
    web_search_quota_exceeded=True instead of a plain string -- the
    frontend distinguishes it from "no results for this particular
    question" to warn the user and disable the web search button.
    """
    if web_status == WebSearchStatus.QUOTA_EXCEEDED:
        return {
            "message": "No context source available: the web search quota "
            "(Tavily) has been exhausted and there are no collections or "
            "files as an alternative source.",
            "web_search_quota_exceeded": True,
        }
    return (
        "No context source available: no collections, no ephemeral files, and the "
        "web search returned no usable results."
    )


def answer_web_only(
    request: QueryRequest,
    chat_memory: list[TurnMemory],
    attachments: list[tuple[str, str]],
) -> tuple[str, list[WebSource]]:
    """
    Case A (see QueryRequest.web_search): no collections/ephemeral files,
    the web search IS the context. Shared by /query and /query/agent.

    Why /query/agent does not run its graph (retrieve -> evaluate ->
    reformulate -> review) in this case: that pipeline is designed
    around local vector retrieval characteristics -- confidence measures
    chunk_index spread / source dominance over FAISS results (see
    src/retrieval/search.py), something
    that has no analogue in web search results. Forcing web results
    through that logic would require redesigning the agent's notion of
    "confidence"; instead, when there are no collections, both endpoints
    share this same simple path -- the agent mode adds nothing different
    here because there is no local retrieval to evaluate or correct.

    attachments: see inject_attachments() in src/prompts/builder.py --
    orthogonal to web context, prepended to the question regardless.

    Raises ValueError if the web search returns no usable results --
    there is no other context source to fall back on.
    """
    outcome = search_web(request.question)
    if not outcome.results:
        raise ValueError(no_context_detail(outcome.status))

    prompt = build_prompt(
        context_chunks=_web_context_chunks(outcome.results),
        question=inject_attachments(request.question, attachments),
        mode=request.mode,
    )
    answer = ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=LLMRole.GENERATE.value,
        **generation_kwargs(request.generation),
    )
    return answer, _web_sources_from_results(outcome.results)


def answer_raw(
    request: QueryRequest,
    chat_memory: list[TurnMemory],
    attachments: list[tuple[str, str]],
) -> str:
    """
    No context source case: no collections, no ephemeral collection, and
    no web_search. Previously this combination returned 422 (see the
    prior version of QueryRequest._require_some_context_source, now
    removed); now it is a valid and expected mode -- standalone
    questions, or a single attached file (see src/context/attachments.py)
    without any selected collection.

    Unlike answer_web_only() and the normal RAG case, this does NOT use
    build_system_prompt() -- the LLM responds directly to the user's
    question (plus attachments, if any) WITHOUT any system prompt (see
    build_messages() in src/llm/generate.py: system_prompt="" omits the
    "system" message entirely). The user is responsible for giving the
    model its role/rules/task in their own message -- this mode does not
    impose any RAG/grounding/citation framework.

    Shared by /query and /query/agent -- neither gains anything from
    running its grounding/review pipeline when there is nothing to
    anchor the response to (same argument as answer_web_only, see its
    docstring).

    May raise ContextLimitExceeded (src/nlp/llm/context_guard.py).
    """
    prompt = inject_attachments(request.question, attachments)
    gen_kwargs = generation_kwargs(request.generation)

    check_context_fit(
        system_prompt="",
        prompt=prompt,
        chat_memory=chat_memory,
        provider=LLMRole.GENERATE.value,
        max_tokens=gen_kwargs["max_tokens"],
    )

    return ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=LLMRole.GENERATE.value,
        system_prompt="",
        **gen_kwargs,
    )


def supplement_with_web(
    question: str,
    answer: str,
    generation: GenerationOptions | None,
) -> tuple[str, list[WebSource], bool]:
    """
    Attempts to complement `answer` (already generated from local context)
    with a web search -- plan case B: normal RAG first, the web only
    appends a paragraph at the end if it truly adds something new.
    Shared by /query and /query/agent -- in both, the response already
    came out of the corresponding pipeline (linear or graph) before
    reaching here, so this step is identical for both.

    Returns (answer, web_sources, quota_exceeded). Completely silent
    best-effort for any failure THAT IS NOT quota exhaustion: the main
    response is never modified or blocked because of this. The quota
    flag IS propagated (even if the result is "no change to the
    response") so the endpoint can warn the frontend -- see
    QueryResponse.web_search_quota_exceeded.
    """
    try:
        outcome = search_web(question)
        if outcome.status == WebSearchStatus.QUOTA_EXCEEDED:
            return answer, [], True
        if not outcome.results:
            return answer, [], False

        supplement_prompt = build_web_supplement_prompt(
            question=question,
            answer=answer,
            web_chunks=_web_context_chunks(outcome.results),
        )
        gen_kwargs = generation_kwargs(generation)
        supplement = ask_llm_internal(
            prompt=supplement_prompt,
            system_prompt=build_web_supplement_system_prompt(),
            provider=LLMRole.WEB_SUPPLEMENT.value,
            max_tokens=gen_kwargs["max_tokens"],
        )

        if supplement is None:
            return answer, [], False

        return (
            f"{answer}\n\n{supplement}",
            _web_sources_from_results(outcome.results),
            False,
        )
    except Exception as e:
        logger.warning(f"[web_search] Web supplement failed (skipped): {e}")
        return answer, [], False


class _GenerationKwargs(TypedDict):
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None


def generation_kwargs(generation: GenerationOptions | None) -> _GenerationKwargs:
    """
    Normalizes a GenerationOptions (or None) into a dict with always-
    present keys -- avoids repeating "generation.X if generation else
    None" in every endpoint. The return TypedDict allows
    `ask_llm(**generation_kwargs(...))` to validate against ask_llm's
    real signature instead of losing precision with dict[str, object].
    `temperature` does not live here: it is not an ask_llm parameter,
    it is a fixed property of each model (see src/config/models/).
    `max_turns` either: it is no longer a per-request override (see
    GenerationOptions in src/api/schemas/chat.py) -- ask_llm() without
    that kwarg uses settings.max_turns always, via its own internal
    default.
    """
    if generation is None:
        return {
            "max_tokens": None,
            "think_mode": None,
            "extra": None,
        }
    return {
        "max_tokens": generation.max_tokens,
        "think_mode": generation.think_mode,
        "extra": generation.extra,
    }
