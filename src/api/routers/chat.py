"""
"chat" router -- the application's regular functions: ask a
question (linear pipeline or LangGraph agent) and list collections.

Stateless design (see docstring of src/api/app.py): the client re-sends
collections and chat_history with every request. The only partial
exception is conversation_id, which only matters if that conversation
has attached ephemeral files (POST /api/v1/files) -- in that case they
are merged with the persisted collections before retrieval.
"""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph

from src.api.deps import (
    get_attachment_store,
    get_context_manager,
    get_ephemeral_store,
    get_rag_graph,
)
from src.api.schemas.chat import (
    CollectionsResponse,
    QueryRequest,
    QueryResponse,
    WebSource,
)
from src.api.services.query import (
    answer_raw,
    answer_web_only,
    build_chat_memory,
    consume_attachments,
    fetch_attachments,
    generation_kwargs,
    resolve_collections,
    resolve_ephemeral,
    supplement_with_web,
)
from src.config.models import get_supports
from src.config.settings import settings
from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager
from src.graph.state import RAGState
from src.nlp.llm.context_guard import ContextLimitExceeded, check_context_fit
from src.nlp.llm.generate import ask_llm
from src.nlp.llm.providers import get_provider
from src.nlp.llm.roles import LLMRole
from src.prompts.builder import (
    build_prompt,
    build_system_prompt,
    inject_attachments,
)
from src.retrieval.search import format_context_chunks, search
from src.utils.logger import logger

router = APIRouter(tags=["chat"])


# ======================================================
# VALIDATION (HTTP-specific, stays in router)
# ======================================================


def _validate_web_search(request: QueryRequest) -> None:
    """
    Rejects with 400 if web_search=True was requested but the global
    kill-switch is off (see settings.web_search_enabled) -- fail-fast
    before resolving collections or touching the LLM, same criterion as
    _validate_generation_options.
    """
    if request.web_search and not settings.web_search_enabled:
        raise HTTPException(
            status_code=400,
            detail="web_search=True was requested, but WEB_SEARCH_ENABLED "
            "is set to False in the server configuration.",
        )


def _validate_generation_options(request: QueryRequest) -> None:
    """
    400 if `request.generation` requests think_mode and/or extra but the
    provider that will actually generate the response
    (LLMRole.GENERATE.value) does not support them -- see
    get_supports() in src/config/models/. Fail-fast here, before
    resolving collections or touching the LLM, same criterion as
    _validate_web_search.

    Only validates against the GENERATE provider: it is the only role
    that the UI exposes as a user-configurable override (see
    GenerationSection.tsx) -- reformulate/review/web_supplement do not
    receive think_mode/extra (see ask_llm_internal/ask_llm_supplement).
    """
    generation = request.generation
    if generation is None:
        return

    provider_name = LLMRole.GENERATE.value
    config = get_provider(provider_name)
    supports = get_supports(config.capabilities, config.model)

    if generation.think_mode is not None and "think_mode" not in supports:
        raise HTTPException(
            status_code=400,
            detail=f"The model '{config.model}' (provider '{provider_name}') does "
            "not have reasoning mode (think_mode) configured.",
        )
    if generation.extra and "extra" not in supports:
        raise HTTPException(
            status_code=400,
            detail=f"The provider '{provider_name}' does not accept the 'extra' field.",
        )


# ======================================================
# ENDPOINTS
# ======================================================


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    context_manager: ContextManager = Depends(get_context_manager),
) -> CollectionsResponse:
    """Lists all persisted collections available on disk."""
    return CollectionsResponse(collections=context_manager.list_all())


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
    attachment_store: AttachmentStore = Depends(get_attachment_store),
) -> QueryResponse:
    """
    Linear pipeline: retrieve -> generate. No adaptive retrieval, prioritizes speed.

    Three possible cases, evaluated in this order (see helpers
    shared with /query/agent):

      Case raw -- no collections/ephemeral files and no web_search:
      there is no RAG context source at all. Responds directly with
      the LLM, without a system prompt (see answer_raw()). Ad-hoc
      attachments (see src/context/attachments.py), if any, ARE
      injected -- they do not count as a "collection", but they are
      context regardless.

      Case A -- no local sources but web_search=True: the web search
      IS the context (see answer_web_only()). If it returns no
      results, 422 (nothing to respond with).

      Case B -- local sources present: the RAG pipeline runs unchanged
      and generates `answer` first; the web is only attempted AFTER
      as an optional, best-effort complement that can never degrade or
      block the already-generated response (see
      supplement_with_web()).
    """
    logger.info(
        f"[api] POST /query | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    _validate_generation_options(request)

    try:
        collections = resolve_collections(request.collections)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    ephemeral = resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    attachments = fetch_attachments(request.conversation_id, attachment_store)

    chat_memory = build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    try:
        if not collections and not request.web_search:
            # Case raw -- see answer_raw().
            answer = answer_raw(request, chat_memory, attachments)
            confidence = 0.0  # not applicable without retrieval, see answer_raw

        elif not collections:
            # Case A -- no collections/ephemeral, but web_search=True.
            answer, web_sources = answer_web_only(request, chat_memory, attachments)
            confidence = 0.0  # not applicable to web results, see answer_web_only
            used_web_search = True

        else:
            # Normal case (identical to behavior before this feature).
            # top_k_initial/top_k_final are no longer per-request overrides (see
            # GenerationOptions in src/api/schemas/chat.py) -- search()
            # resolves both internally against settings.soft_top_k_*/
            # hard_top_k_* based on request.mode.
            results, confidence = search(
                question=request.question,
                mode=request.mode,
                collections=collections,
            )

            if not results:
                raise HTTPException(
                    status_code=422,
                    detail="No relevant context found for the given question and collections.",
                )

            context_chunks = format_context_chunks(results)

            prompt = build_prompt(
                context_chunks=context_chunks,
                question=inject_attachments(request.question, attachments),
                mode=request.mode,
            )

            gen_kwargs = generation_kwargs(request.generation)
            check_context_fit(
                system_prompt=build_system_prompt(),
                prompt=prompt,
                chat_memory=chat_memory,
                provider=LLMRole.GENERATE.value,
                max_tokens=gen_kwargs["max_tokens"],
            )

            answer = ask_llm(
                prompt=prompt,
                chat_memory=chat_memory,
                provider=LLMRole.GENERATE.value,
                **gen_kwargs,
            )

            # Case B: the main response is already complete above; this
            # can only append a paragraph at the end, never replace it.
            if request.web_search:
                answer, web_sources, quota_exceeded = supplement_with_web(
                    question=request.question,
                    answer=answer,
                    generation=request.generation,
                )
                used_web_search = bool(web_sources)
    except ContextLimitExceeded as e:
        raise HTTPException(status_code=413, detail=e.as_detail()) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=e.args[0]) from e

    # Single-use ad-hoc attachments -- see AttachmentStore docstring.
    consume_attachments(request.conversation_id, attachment_store)

    # See equivalent comment in src/llm/generate.py _complete(): diagnostic
    # log for the incomplete messages bug in the frontend. If this length
    # already matches the one logged in _complete(), the response came out
    # complete from this endpoint and the truncation happens in the
    # API->browser segment.
    logger.info(
        f"[api] POST /query | respondiendo | answer_len={len(answer)} "
        f"| used_web_search={used_web_search}"
    )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        collections_used=[c["collection_name"] for c in collections],
        used_web_search=used_web_search,
        web_sources=web_sources or None,
        web_search_quota_exceeded=quota_exceeded,
    )


@router.post("/query/agent", response_model=QueryResponse)
async def query_agent(
    request: QueryRequest,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
    attachment_store: AttachmentStore = Depends(get_attachment_store),
    rag_graph: CompiledStateGraph[RAGState] = Depends(get_rag_graph),
) -> QueryResponse:
    """
    LangGraph pipeline: retrieve -> evaluate -> [reformulate ->] generate -> review -> [correct].

    Same three cases as /query (see its docstring), integrated without
    touching the graph for the two that skip it:

      Case raw -- no collections/ephemeral and no web_search: uses
      answer_raw(), same as /query. The entire graph is skipped -- with
      no retrieved context there is nothing for review/correct to
      evaluate.

      Case A -- no collections/files, web_search=True: uses
      answer_web_only(), the same simple path as /query. See its
      docstring for why (confidence/reformulate/review are designed
      around local vector retrieval, with no meaningful analogue for
      web results).

      Case B -- collections present: the graph runs exactly as before
      this feature (retrieve -> evaluate -> ... -> review ->
      [correct]), finishes, and ONLY THEN is the web complement
      attempted on the `answer` already reviewed/corrected by the graph
      -- the same supplement_with_web() used by /query, best-effort and
      cannot alter what the graph already decided. Ad-hoc attachments, if
      any, travel in RAGState.attachments and generate_node injects them
      into the final prompt (see src/graph/nodes.py).
    """
    logger.info(
        f"[api] POST /query/agent | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    _validate_generation_options(request)

    try:
        collections = resolve_collections(request.collections)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    ephemeral = resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    attachments = fetch_attachments(request.conversation_id, attachment_store)

    chat_memory = build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    try:
        if not collections and not request.web_search:
            # Case raw -- see answer_raw().
            answer = answer_raw(request, chat_memory, attachments)
            confidence = 0.0
            reformulated = False

        elif not collections:
            # Case A -- see answer_web_only docstring.
            answer, web_sources = answer_web_only(request, chat_memory, attachments)
            confidence = 0.0
            reformulated = False
            used_web_search = True

        else:
            # Normal case: the graph runs exactly as before this feature --
            # neither graph/state.py (except the new attachments field) nor
            # graph/nodes.py changed their retrieval/review logic.
            # top_k_initial/top_k_final/max_turns no longer live in RAGState
            # (see its docstring in src/graph/state.py): retrieve_node and
            # generate_node resolve them internally against settings, without
            # per-request overrides.
            gen_kwargs = generation_kwargs(request.generation)

            initial_state: RAGState = {
                "question": request.question,
                "mode": request.mode,
                "collections": collections,
                "chat_memory": chat_memory,
                "results": [],
                "confidence": 0.0,
                "reformulated": False,
                "answer": "",
                "review_passed": False,
                "review_feedback": "",
                "review_attempts": 0,
                "max_tokens": gen_kwargs["max_tokens"],
                "think_mode": gen_kwargs["think_mode"],
                "extra": gen_kwargs["extra"],
                "attachments": attachments,
            }

            # CompiledStateGraph.invoke() is typed in the library as
            # `dict[str, Any] | Any` (not the generic StateT), so an explicit
            # cast is more honest here than blindly ignoring the error: it
            # documents the exact point where LangGraph's precision ends and
            # ours begins.
            #
            # ContextLimitExceeded can escape from generate_node (see its
            # docstring in src/graph/nodes.py) -- it is translated to 413
            # here, same as in /query.
            final_state = cast(RAGState, rag_graph.invoke(initial_state))

            answer = final_state["answer"]

            if not answer:
                raise HTTPException(
                    status_code=422,
                    detail="No relevant context found for the given question and collections.",
                )

            confidence = final_state["confidence"]
            reformulated = final_state["reformulated"]

            # Case B: the graph has already finished (including review/correct);
            # this can only append a paragraph at the end, never reopens the
            # graph's review cycle or modifies what it already decided.
            if request.web_search:
                answer, web_sources, quota_exceeded = supplement_with_web(
                    question=request.question,
                    answer=answer,
                    generation=request.generation,
                )
                used_web_search = bool(web_sources)
    except ContextLimitExceeded as e:
        raise HTTPException(status_code=413, detail=e.as_detail()) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=e.args[0]) from e

    # Single-use ad-hoc attachments -- see AttachmentStore docstring.
    consume_attachments(request.conversation_id, attachment_store)

    logger.info(
        f"[api] POST /query/agent | respondiendo | answer_len={len(answer)} "
        f"| used_web_search={used_web_search}"
    )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        collections_used=[c["collection_name"] for c in collections],
        reformulated=reformulated,
        used_web_search=used_web_search,
        web_sources=web_sources or None,
        web_search_quota_exceeded=quota_exceeded,
    )
