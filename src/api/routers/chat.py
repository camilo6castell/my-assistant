"""
Router "chat" -- las funciones regulares de la aplicación: hacer una
pregunta (pipeline lineal o agente LangGraph) y listar colecciones.

Diseño stateless (ver docstring de src/api/app.py): el cliente reenvía
collections y chat_history en cada request. La única excepción parcial
es conversation_id, que solo importa si esa conversación tiene archivos
efímeros adjuntos (POST /api/v1/files) -- en ese caso se mergean con las
colecciones persistidas antes de hacer retrieval.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph

from src.api.deps import get_context_manager, get_ephemeral_store, get_rag_graph
from src.api.schemas.chat import (
    CollectionsResponse,
    GenerationOptions,
    QueryRequest,
    QueryResponse,
)
from src.chat.types import TurnMemory
from src.config.settings import settings
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager, LoadedCollection
from src.config.models import get_model_capabilities, supports_set
from src.graph.state import RAGState
from src.llm.generate import ask_llm
from src.llm.providers import get_provider
from src.prompts.builder import build_prompt
from src.retrieval.search import search
from src.utils.logger import logger

router = APIRouter(tags=["chat"])


# ======================================================
# HELPERS
# ======================================================


def _resolve_collections(tokens: list[str]) -> list[LoadedCollection]:
    """
    Carga las colecciones persistidas pedidas en el request.

    Cada request carga sus propias colecciones desde disco vía un
    ContextManager local a esta función -- no el singleton de deps.py,
    que solo se usa para listar (GET /collections). Tokens vacíos (caso
    de un request que solo usa archivos efímeros) devuelven lista vacía
    sin error.
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
        raise HTTPException(
            status_code=404,
            detail=f"Collections not found: {not_found}",
        )

    return cm.get_loaded_collections()


def _resolve_ephemeral(
    conversation_id: str | None, store: EphemeralStore
) -> LoadedCollection | None:
    """Colección efímera de la conversación, si tiene archivos subidos."""
    if conversation_id is None:
        return None
    return store.get_collection(conversation_id)


def _build_chat_memory(history: list[dict[str, str]]) -> list[TurnMemory]:
    """
    Convierte el chat_history del request en TurnMemory para el LLM.

    Ignora silenciosamente entradas malformadas (sin 'user' o 'assistant')
    para no romper el request si el cliente envía historial parcial.
    """
    memory: list[TurnMemory] = []
    for turn in history:
        if "user" in turn and "assistant" in turn:
            memory.append(TurnMemory(user=turn["user"], assistant=turn["assistant"]))
    return memory


def _validate_generation_options(request: QueryRequest) -> None:
    """
    Rechaza con 400 las opciones de generación que el provider activo
    (settings.generate_provider, el que escribe la respuesta final) no
    soporta -- en vez de aceptarlas en silencio y que el cliente crea
    que se aplicaron cuando no pasó nada.
    """
    if request.generation is None:
        return

    provider = get_provider(settings.generate_provider)
    supports = supports_set(get_model_capabilities(provider.capabilities, provider.model))
    requested = {
        name
        for name, value in (
            ("temperature", request.generation.temperature),
            ("max_tokens", request.generation.max_tokens),
            ("think_mode", request.generation.think_mode),
            ("extra", request.generation.extra),
        )
        if value is not None
    }
    unsupported = requested - supports

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El modelo '{provider.model}' (provider '{provider.name}') no soporta: "
                f"{sorted(unsupported)}. Soportados: {sorted(supports)}."
            ),
        )


def _resolve_top_k(request: QueryRequest) -> tuple[int | None, int | None]:
    """
    Resuelve (top_k_initial, top_k_final) del request contra los defaults
    de settings para request.mode, y valida que final <= initial -- sin
    esto, search() (src/retrieval/search.py) devolvería un slice
    resultados[:top_k_final] más ancho que lo que realmente se recuperó
    en el índice, silenciosamente inútil en vez de un error claro.

    Devuelve los overrides tal cual vinieron (posiblemente None) para
    pasarlos directo a search()/RAGState -- la resolución final contra
    settings ya la hace search() internamente; acá solo se valida la
    combinación antes de llegar ahí.
    """
    generation = request.generation
    top_k_initial = generation.top_k_initial if generation else None
    top_k_final = generation.top_k_final if generation else None

    if request.mode == "SOFT":
        default_initial = settings.soft_top_k_initial
        default_final = settings.soft_top_k_final
    else:
        default_initial = settings.hard_top_k_initial
        default_final = settings.hard_top_k_final

    effective_initial = default_initial if top_k_initial is None else top_k_initial
    effective_final = default_final if top_k_final is None else top_k_final

    if effective_final > effective_initial:
        raise HTTPException(
            status_code=400,
            detail=(
                f"top_k_final ({effective_final}) no puede ser mayor que "
                f"top_k_initial ({effective_initial}) para el modo {request.mode}."
            ),
        )

    return top_k_initial, top_k_final


class _GenerationKwargs(TypedDict):
    temperature: float | None
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None
    max_turns: int | None


def _generation_kwargs(generation: GenerationOptions | None) -> _GenerationKwargs:
    """
    Normaliza un GenerationOptions (o None) a un dict con las claves
    siempre presentes -- evita repetir "generation.X if generation else
    None" varias veces en cada endpoint. El TypedDict de retorno permite
    que `ask_llm(**_generation_kwargs(...))` se valide contra la firma
    real de ask_llm en vez de perder precisión con dict[str, object].
    top_k_initial/top_k_final no viven acá porque no son parámetros de
    ask_llm -- se resuelven aparte con _resolve_top_k() y van a search()
    /RAGState.
    """
    if generation is None:
        return {
            "temperature": None,
            "max_tokens": None,
            "think_mode": None,
            "extra": None,
            "max_turns": None,
        }
    return {
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
        "think_mode": generation.think_mode,
        "extra": generation.extra,
        "max_turns": generation.max_turns,
    }


# ======================================================
# ENDPOINTS
# ======================================================


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    context_manager: ContextManager = Depends(get_context_manager),
) -> CollectionsResponse:
    """Lista todas las colecciones persistidas disponibles en disco."""
    return CollectionsResponse(collections=context_manager.list_all())


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> QueryResponse:
    """Pipeline lineal: retrieve → generate. Sin adaptive retrieval, prioriza velocidad."""
    logger.info(
        f"[api] POST /query | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r}"
    )

    _validate_generation_options(request)
    top_k_initial, top_k_final = _resolve_top_k(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    if not collections:
        raise HTTPException(
            status_code=422,
            detail="No context source available: no collections resolved and no "
            "ephemeral files for this conversation_id.",
        )

    chat_memory = _build_chat_memory(request.chat_history)

    results, confidence = search(
        question=request.question,
        mode=request.mode,
        collections=collections,
        top_k_initial=top_k_initial,
        top_k_final=top_k_final,
    )

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No relevant context found for the given question and collections.",
        )

    context_chunks = [
        f"SOURCE: {r.source}\nCOLLECTION: {r.collection}\nPAGE: {r.page}\n\n{r.text}"
        for r in results
    ]

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=request.question,
        mode=request.mode,
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        **_generation_kwargs(request.generation),
    )

    # Ver comentario equivalente en src/llm/generate.py _complete(): log
    # diagnóstico para el bug de mensajes incompletos en el frontend. Si
    # esta longitud ya coincide con la logueada en _complete(), la
    # respuesta salió completa de este endpoint y el corte ocurre en el
    # tramo API->navegador.
    logger.info(f"[api] POST /query | respondiendo | answer_len={len(answer)}")

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        collections_used=[c["collection_name"] for c in collections],
    )


@router.post("/query/agent", response_model=QueryResponse)
async def query_agent(
    request: QueryRequest,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
    rag_graph: CompiledStateGraph[RAGState] = Depends(get_rag_graph),
) -> QueryResponse:
    """Pipeline LangGraph: retrieve → evaluate → [reformulate →] generate → review → [correct]."""
    logger.info(
        f"[api] POST /query/agent | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r}"
    )

    _validate_generation_options(request)
    top_k_initial, top_k_final = _resolve_top_k(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    if not collections:
        raise HTTPException(
            status_code=422,
            detail="No context source available: no collections resolved and no "
            "ephemeral files for this conversation_id.",
        )

    chat_memory = _build_chat_memory(request.chat_history)
    gen_kwargs = _generation_kwargs(request.generation)

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
        "temperature": gen_kwargs["temperature"],
        "max_tokens": gen_kwargs["max_tokens"],
        "think_mode": gen_kwargs["think_mode"],
        "extra": gen_kwargs["extra"],
        "max_turns": gen_kwargs["max_turns"],
        "top_k_initial": top_k_initial,
        "top_k_final": top_k_final,
    }

    # CompiledStateGraph.invoke() está tipado en la librería como
    # `dict[str, Any] | Any` (no como el StateT genérico), así que un
    # cast explícito es más honesto acá que ignorar el error a ciegas:
    # documenta justo el punto donde termina la precisión de LangGraph
    # y empieza la nuestra.
    final_state = cast(RAGState, rag_graph.invoke(initial_state))

    answer = final_state["answer"]

    if not answer:
        raise HTTPException(
            status_code=422,
            detail="No relevant context found for the given question and collections.",
        )

    logger.info(f"[api] POST /query/agent | respondiendo | answer_len={len(answer)}")

    return QueryResponse(
        answer=answer,
        confidence=final_state["confidence"],
        collections_used=[c["collection_name"] for c in collections],
        reformulated=final_state["reformulated"],
    )
