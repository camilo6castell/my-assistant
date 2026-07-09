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
    WebSource,
)
from src.chat.types import TurnMemory
from src.config.settings import settings
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager, LoadedCollection
from src.graph.state import RAGState
from src.llm.generate import ask_llm, ask_llm_supplement
from src.llm.roles import LLMRole
from src.prompts.builder import (
    WEB_SUPPLEMENT_SENTINEL,
    build_prompt,
    build_web_supplement_prompt,
    build_web_supplement_system_prompt,
)
from src.retrieval.search import search
from src.retrieval.web_search import WebSearchResult, WebSearchStatus, search_web
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


def _require_context_source(
    collections: list[LoadedCollection], request: QueryRequest
) -> None:
    """
    422 si no hay ninguna fuente de contexto posible: ni colecciones/
    archivos ni web_search=True. Compartido por /query y /query/agent --
    idéntico chequeo, antes solo vivía en /query.
    """
    if not collections and not request.web_search:
        raise HTTPException(
            status_code=422,
            detail="No context source available: no collections resolved and no "
            "ephemeral files for this conversation_id.",
        )


def _validate_web_search(request: QueryRequest) -> None:
    """
    Rechaza con 400 si se pidió web_search=True pero el kill-switch
    global está apagado (ver settings.web_search_enabled) -- fail-fast
    antes de resolver colecciones o tocar el LLM, mismo criterio que
    _validate_generation_options.
    """
    if request.web_search and not settings.web_search_enabled:
        raise HTTPException(
            status_code=400,
            detail="web_search=True fue solicitado, pero WEB_SEARCH_ENABLED "
            "está en False en la configuración del servidor.",
        )


def _web_context_chunks(results: list[WebSearchResult]) -> list[str]:
    """Mismo formato que los chunks locales (SOURCE/.../texto) para que build_prompt() sea idéntico en ambos casos."""
    return [f"SOURCE: {r.title}\nURL: {r.url}\n\n{r.content}" for r in results]


def _web_sources_from_results(results: list[WebSearchResult]) -> list[WebSource]:
    return [WebSource(title=r.title, url=r.url) for r in results]


def _no_context_detail(web_status: WebSearchStatus | None) -> dict[str, Any] | str:
    """
    Arma el `detail` del 422 "sin fuente de contexto". Si la web era la
    única fuente posible y falló específicamente por cuota agotada,
    devuelve un dict estructurado con web_search_quota_exceeded=True en
    vez de un string plano -- el frontend lo distingue de "no había
    resultados para esta pregunta puntual" para poder avisar al usuario
    y deshabilitar el botón de búsqueda web (ver apiErrorMessage() en
    ui/src/lib/api/client.ts).
    """
    if web_status == WebSearchStatus.QUOTA_EXCEEDED:
        return {
            "message": "No context source available: se agotó la cuota de búsqueda "
            "web (Tavily) y no hay colecciones ni archivos como fuente alternativa.",
            "web_search_quota_exceeded": True,
        }
    return (
        "No context source available: no collections, no ephemeral files, and the "
        "web search returned no usable results."
    )


def _answer_web_only(
    request: QueryRequest,
    chat_memory: list[TurnMemory],
) -> tuple[str, list[WebSource]]:
    """
    Caso A (ver QueryRequest.web_search): sin colecciones/archivos, la
    búsqueda web ES el contexto. Compartido por /query y /query/agent.

    Por qué /query/agent no corre su grafo (retrieve -> evaluate ->
    reformulate -> review) en este caso: ese pipeline está diseñado
    alrededor de características del retrieval vectorial local --
    confidence mide spread de chunk_index / dominancia de fuente sobre
    resultados de FAISS (ver settings.confidence_limit y
    src/retrieval/search.py), algo que no tiene un análogo con
    resultados de una búsqueda web. Forzar resultados web a través de
    esa lógica exigiría rediseñar la noción de "confidence" del agente;
    en cambio, cuando no hay colecciones, ambos endpoints comparten este
    mismo camino simple -- el modo agente no aporta nada distinto acá
    porque no hay retrieval local que evaluar o corregir.

    Lanza HTTPException(422) si la búsqueda web no devuelve resultados
    utilizables -- no hay ninguna otra fuente de contexto a la que caer.
    """
    outcome = search_web(request.question)
    if not outcome.results:
        raise HTTPException(status_code=422, detail=_no_context_detail(outcome.status))

    prompt = build_prompt(
        context_chunks=_web_context_chunks(outcome.results),
        question=request.question,
        mode=request.mode,
    )
    answer = ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=settings.provider_for(LLMRole.GENERATE),
        **_generation_kwargs(request.generation),
    )
    return answer, _web_sources_from_results(outcome.results)


def _supplement_with_web(
    question: str,
    answer: str,
    generation: GenerationOptions | None,
) -> tuple[str, list[WebSource], bool]:
    """
    Intenta complementar `answer` (ya generada desde contexto local) con
    una búsqueda web -- caso B del plan: RAG normal primero, la web solo
    agrega un párrafo al final si de verdad aporta algo nuevo. Compartido
    por /query y /query/agent -- en ambos, la respuesta ya salió del
    pipeline correspondiente (lineal o grafo) antes de llegar acá, así
    que este paso es idéntico para los dos.

    Devuelve (answer, web_sources, quota_exceeded). Best-effort total y
    silencioso para cualquier fallo QUE NO sea cuota agotada: la
    respuesta principal nunca se modifica ni se bloquea por esto. El
    flag de cuota SÍ se propaga (aunque el resultado sea "sin cambios en
    la respuesta") para que el endpoint pueda avisarle al frontend --
    ver QueryResponse.web_search_quota_exceeded.
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
        gen_kwargs = _generation_kwargs(generation)
        supplement = ask_llm_supplement(
            prompt=supplement_prompt,
            system_prompt=build_web_supplement_system_prompt(),
            provider=settings.provider_for(LLMRole.WEB_SUPPLEMENT),
            temperature=gen_kwargs["temperature"],
            max_tokens=gen_kwargs["max_tokens"],
        )

        if supplement is None:
            return answer, [], False

        supplement = supplement.strip()
        if not supplement or supplement == WEB_SUPPLEMENT_SENTINEL:
            return answer, [], False

        return (
            f"{answer}\n\n{supplement}",
            _web_sources_from_results(outcome.results),
            False,
        )
    except Exception as e:
        logger.warning(f"[web_search] Fallo el complemento web (se omite): {e}")
        return answer, [], False


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
    """
    Pipeline lineal: retrieve → generate. Sin adaptive retrieval, prioriza velocidad.

    web_search=True tiene dos comportamientos distintos según haya o no
    colecciones/archivos (ver QueryRequest.web_search y los helpers
    _answer_web_only / _supplement_with_web, compartidos con /query/agent):

      Caso A -- sin fuentes locales: la búsqueda web ES el contexto.
      Si no devuelve resultados, 422 (no hay nada con qué responder).

      Caso B -- con fuentes locales: el pipeline RAG corre sin cambios
      y genera `answer` primero; la web solo se intenta DESPUÉS, como un
      complemento opcional y best-effort que nunca puede degradar ni
      bloquear la respuesta ya generada.
    """
    logger.info(
        f"[api] POST /query | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    top_k_initial, top_k_final = _resolve_top_k(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    _require_context_source(collections, request)

    chat_memory = _build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    if not collections:
        # Caso A. _require_context_source ya garantiza que si llegamos
        # acá (sin collections ni ephemeral), web_search es True.
        answer, web_sources = _answer_web_only(request, chat_memory)
        confidence = 0.0  # no aplica a resultados web, ver _answer_web_only
        used_web_search = True

    else:
        # Caso normal (idéntico al comportamiento previo a esta feature).
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
            provider=settings.provider_for(LLMRole.GENERATE),
            **_generation_kwargs(request.generation),
        )

        # Caso B: la respuesta principal ya está completa arriba; esto
        # solo puede agregarle un párrafo al final, nunca reemplazarla.
        if request.web_search:
            answer, web_sources, quota_exceeded = _supplement_with_web(
                question=request.question,
                answer=answer,
                generation=request.generation,
            )
            used_web_search = bool(web_sources)

    # Ver comentario equivalente en src/llm/generate.py _complete(): log
    # diagnóstico para el bug de mensajes incompletos en el frontend. Si
    # esta longitud ya coincide con la logueada en _complete(), la
    # respuesta salió completa de este endpoint y el corte ocurre en el
    # tramo API->navegador.
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
    rag_graph: CompiledStateGraph[RAGState] = Depends(get_rag_graph),
) -> QueryResponse:
    """
    Pipeline LangGraph: retrieve → evaluate → [reformulate →] generate → review → [correct].

    web_search=True se integra sin tocar el grafo (ver graph/state.py,
    graph/nodes.py -- ninguno de los dos cambió):

      Caso A -- sin colecciones/archivos: usa _answer_web_only(), el
      mismo camino simple que /query. El grafo entero se salta -- ver el
      docstring de _answer_web_only para por qué (confidence/reformulate
      /review están diseñados en torno al retrieval vectorial local, sin
      un análogo significativo para resultados web).

      Caso B -- con colecciones: el grafo corre exactamente igual que
      antes de esta feature (retrieve -> evaluate -> ... -> review ->
      [correct]), termina, y SOLO DESPUÉS se intenta el complemento web
      sobre el `answer` ya revisado/corregido por el grafo -- mismo
      _supplement_with_web() que usa /query, best-effort y no puede
      alterar lo que el grafo ya decidió.
    """
    logger.info(
        f"[api] POST /query/agent | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    top_k_initial, top_k_final = _resolve_top_k(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    _require_context_source(collections, request)

    chat_memory = _build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    if not collections:
        # Caso A -- ver docstring de _answer_web_only.
        answer, web_sources = _answer_web_only(request, chat_memory)
        confidence = 0.0
        reformulated = False
        used_web_search = True

    else:
        # Caso normal: el grafo corre exactamente igual que antes de
        # esta feature -- ni graph/state.py ni graph/nodes.py cambiaron.
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

        confidence = final_state["confidence"]
        reformulated = final_state["reformulated"]

        # Caso B: el grafo ya terminó (incluyendo review/correct); esto
        # solo puede agregarle un párrafo al final, nunca reabre el ciclo
        # de revisión del grafo ni modifica lo que ya decidió.
        if request.web_search:
            answer, web_sources, quota_exceeded = _supplement_with_web(
                question=request.question,
                answer=answer,
                generation=request.generation,
            )
            used_web_search = bool(web_sources)

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
