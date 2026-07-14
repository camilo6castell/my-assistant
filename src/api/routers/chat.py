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

from src.api.deps import (
    get_attachment_store,
    get_context_manager,
    get_ephemeral_store,
    get_rag_graph,
)
from src.api.schemas.chat import (
    CollectionsResponse,
    GenerationOptions,
    QueryRequest,
    QueryResponse,
    WebSource,
)
from src.cli.types import TurnMemory
from src.config.models import get_supports
from src.config.settings import settings
from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager, LoadedCollection
from src.graph.state import RAGState
from src.nlp.llm.context_guard import ContextLimitExceeded, check_context_fit
from src.nlp.llm.generate import ask_llm, ask_llm_internal
from src.nlp.llm.providers import get_provider
from src.nlp.llm.roles import LLMRole
from src.prompts.builder import (
    build_prompt,
    build_system_prompt,
    build_web_supplement_prompt,
    build_web_supplement_system_prompt,
    inject_attachments,
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


def _fetch_attachments(
    conversation_id: str | None, store: AttachmentStore
) -> list[tuple[str, str]]:
    """Adjuntos ad-hoc pendientes de esta conversación (ver src/context/attachments.py)."""
    return store.list_contents(conversation_id)


def _consume_attachments(conversation_id: str | None, store: AttachmentStore) -> None:
    """
    Borra los adjuntos ad-hoc de la conversación después de procesar una
    query -- son de un solo uso (ver docstring de AttachmentStore). Se
    llama al final de /query y /query/agent, en las tres ramas
    (raw/web-only/RAG) por igual: los adjuntos aplican a cualquiera.
    No-op si conversation_id es None o no había adjuntos.
    """
    if conversation_id is not None:
        store.remove_conversation(conversation_id)


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


def _validate_generation_options(request: QueryRequest) -> None:
    """
    400 si `request.generation` pide think_mode y/o extra pero el
    provider que efectivamente va a generar la respuesta
    (settings.provider_for(LLMRole.GENERATE)) no los soporta -- ver
    get_supports() en src/config/models/. Fail-fast acá, antes de
    resolver colecciones o tocar el LLM, mismo criterio que
    _validate_web_search.

    Solo valida contra el provider de GENERATE: es el único rol que la
    UI expone como override configurable por el usuario (ver
    GenerationSection.tsx) -- reformulate/review/web_supplement no
    reciben think_mode/extra (ver ask_llm_internal/ask_llm_supplement).
    """
    generation = request.generation
    if generation is None:
        return

    provider_name = settings.provider_for(LLMRole.GENERATE)
    config = get_provider(provider_name)
    supports = get_supports(config.capabilities, config.model)

    if generation.think_mode is not None and "think_mode" not in supports:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{config.model}' (provider '{provider_name}') no "
            "tiene modo de razonamiento (think_mode) configurado.",
        )
    if generation.extra and "extra" not in supports:
        raise HTTPException(
            status_code=400,
            detail=f"El provider '{provider_name}' no acepta el campo 'extra'.",
        )


def _web_context_chunks(results: list[WebSearchResult]) -> list[str]:
    """
    Mismo formato que los chunks locales (SOURCE/.../texto) para que build_prompt()
    sea idéntico en ambos casos.
    """
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
    attachments: list[tuple[str, str]],
) -> tuple[str, list[WebSource]]:
    """
    Caso A (ver QueryRequest.web_search): sin colecciones/archivos
    efímeros, la búsqueda web ES el contexto. Compartido por /query y
    /query/agent.

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

    attachments: ver inject_attachments() en src/prompts/builder.py --
    ortogonales al contexto web, se anteponen a la pregunta igual.

    Lanza HTTPException(422) si la búsqueda web no devuelve resultados
    utilizables -- no hay ninguna otra fuente de contexto a la que caer.
    """
    outcome = search_web(request.question)
    if not outcome.results:
        raise HTTPException(status_code=422, detail=_no_context_detail(outcome.status))

    prompt = build_prompt(
        context_chunks=_web_context_chunks(outcome.results),
        question=inject_attachments(request.question, attachments),
        mode=request.mode,
    )
    answer = ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=settings.provider_for(LLMRole.GENERATE),
        **_generation_kwargs(request.generation),
    )
    return answer, _web_sources_from_results(outcome.results)


def _answer_raw(
    request: QueryRequest,
    chat_memory: list[TurnMemory],
    attachments: list[tuple[str, str]],
) -> str:
    """
    Caso sin ninguna fuente de contexto: ni colecciones, ni colección
    efímera, ni web_search. Antes esta combinación devolvía 422 (ver
    versión previa de QueryRequest._require_some_context_source, ya
    eliminada); ahora es un modo válido y esperado -- preguntas sueltas,
    o un archivo puntual adjunto (ver src/context/attachments.py) sin
    ninguna colección elegida.

    A diferencia de _answer_web_only() y del caso RAG normal, acá NO se
    usa build_system_prompt() -- el LLM responde directo a la pregunta
    del usuario (más los adjuntos, si los hay) SIN ningún system prompt
    (ver build_messages() en src/llm/generate.py: system_prompt=""
    omite el mensaje "system" del todo). El usuario es responsable de
    darle rol/reglas/tarea al modelo en su propio mensaje -- este modo
    no le impone ningún marco de RAG/grounding/citación.

    Compartido por /query y /query/agent -- ninguno de los dos gana nada
    corriendo su pipeline de grounding/review cuando no hay nada contra
    qué anclar la respuesta (mismo argumento que _answer_web_only, ver
    su docstring).
    """
    prompt = inject_attachments(request.question, attachments)
    generation_kwargs = _generation_kwargs(request.generation)

    try:
        check_context_fit(
            system_prompt="",
            prompt=prompt,
            chat_memory=chat_memory,
            provider=settings.provider_for(LLMRole.GENERATE),
            max_tokens=generation_kwargs["max_tokens"],
        )
    except ContextLimitExceeded as e:
        raise HTTPException(status_code=413, detail=e.as_detail()) from e

    return ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=settings.provider_for(LLMRole.GENERATE),
        system_prompt="",
        **generation_kwargs,
    )


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
        print(outcome)
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
        supplement = ask_llm_internal(
            prompt=supplement_prompt,
            system_prompt=build_web_supplement_system_prompt(),
            provider=settings.provider_for(LLMRole.WEB_SUPPLEMENT),
            max_tokens=gen_kwargs["max_tokens"],
        )

        if supplement is None:
            return answer, [], False

        # DECISIÓN (dejar comentado, no borrar): se decidió no filtrar el
        # supplement por el sentinel WEB_SUPPLEMENT_SENTINEL en el flujo
        # actual -- ver el comentario extenso junto a su definición en
        # src/prompts/builder.py para el porqué se conserva como
        # referencia en vez de borrarse.
        # supplement = supplement.strip()
        # if not supplement or supplement == WEB_SUPPLEMENT_SENTINEL:
        #     return answer, [], False

        return (
            f"{answer}\n\n{supplement}",
            _web_sources_from_results(outcome.results),
            False,
        )
    except Exception as e:
        logger.warning(f"[web_search] Fallo el complemento web (se omite): {e}")
        return answer, [], False


class _GenerationKwargs(TypedDict):
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None


def _generation_kwargs(generation: GenerationOptions | None) -> _GenerationKwargs:
    """
    Normaliza un GenerationOptions (o None) a un dict con las claves
    siempre presentes -- evita repetir "generation.X if generation else
    None" varias veces en cada endpoint. El TypedDict de retorno permite
    que `ask_llm(**_generation_kwargs(...))` se valide contra la firma
    real de ask_llm en vez de perder precisión con dict[str, object].
    `temperature` no vive acá: no es un parámetro de ask_llm, es una
    propiedad fija de cada modelo (ver src/config/models/). `max_turns`
    tampoco: dejó de ser un override por-request (ver GenerationOptions
    en src/api/schemas/chat.py) -- ask_llm() sin ese kwarg usa
    settings.max_turns siempre, vía su propio default interno.
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
    attachment_store: AttachmentStore = Depends(get_attachment_store),
) -> QueryResponse:
    """
    Pipeline lineal: retrieve → generate. Sin adaptive retrieval, prioriza velocidad.

    Tres casos posibles, evaluados en este orden (ver helpers
    compartidos con /query/agent):

      Caso raw -- sin colecciones/archivos efímeros y sin web_search:
      no hay ninguna fuente de contexto RAG. Responde directo con el
      LLM, sin system prompt (ver _answer_raw()). Los archivos adjuntos
      ad-hoc (ver src/context/attachments.py), si los hay, SÍ se
      inyectan -- no cuentan como "colección", pero son contexto igual.

      Caso A -- sin fuentes locales pero con web_search=True: la
      búsqueda web ES el contexto (ver _answer_web_only()). Si no
      devuelve resultados, 422 (no hay nada con qué responder).

      Caso B -- con fuentes locales: el pipeline RAG corre sin cambios
      y genera `answer` primero; la web solo se intenta DESPUÉS, como un
      complemento opcional y best-effort que nunca puede degradar ni
      bloquear la respuesta ya generada (ver _supplement_with_web()).
    """
    logger.info(
        f"[api] POST /query | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    _validate_generation_options(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    attachments = _fetch_attachments(request.conversation_id, attachment_store)

    chat_memory = _build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    if not collections and not request.web_search:
        # Caso raw -- ver _answer_raw().
        answer = _answer_raw(request, chat_memory, attachments)
        confidence = 0.0  # no aplica sin retrieval, ver _answer_raw

    elif not collections:
        # Caso A -- sin colecciones/efímeros, pero con web_search=True.
        answer, web_sources = _answer_web_only(request, chat_memory, attachments)
        confidence = 0.0  # no aplica a resultados web, ver _answer_web_only
        used_web_search = True

    else:
        # Caso normal (idéntico al comportamiento previo a esta feature).
        # top_k_initial/top_k_final ya no son overrides por-request (ver
        # GenerationOptions en src/api/schemas/chat.py) -- search()
        # resuelve ambos internamente contra settings.soft_top_k_*/
        # hard_top_k_* según request.mode.
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

        context_chunks = [
            f"SOURCE: {r.source}\nCOLLECTION: {r.collection}\nPAGE: {r.page}\n\n{r.text}"
            for r in results
        ]

        prompt = build_prompt(
            context_chunks=context_chunks,
            question=inject_attachments(request.question, attachments),
            mode=request.mode,
        )

        generation_kwargs = _generation_kwargs(request.generation)
        try:
            check_context_fit(
                system_prompt=build_system_prompt(),
                prompt=prompt,
                chat_memory=chat_memory,
                provider=settings.provider_for(LLMRole.GENERATE),
                max_tokens=generation_kwargs["max_tokens"],
            )
        except ContextLimitExceeded as e:
            raise HTTPException(status_code=413, detail=e.as_detail()) from e

        answer = ask_llm(
            prompt=prompt,
            chat_memory=chat_memory,
            provider=settings.provider_for(LLMRole.GENERATE),
            **generation_kwargs,
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

    # Adjuntos ad-hoc de un solo uso -- ver docstring de AttachmentStore.
    _consume_attachments(request.conversation_id, attachment_store)

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
    attachment_store: AttachmentStore = Depends(get_attachment_store),
    rag_graph: CompiledStateGraph[RAGState] = Depends(get_rag_graph),
) -> QueryResponse:
    """
    Pipeline LangGraph: retrieve → evaluate → [reformulate →] generate → review → [correct].

    Mismos tres casos que /query (ver su docstring), integrados sin
    tocar el grafo para los dos que lo saltan:

      Caso raw -- sin colecciones/efímeros y sin web_search: usa
      _answer_raw(), igual que /query. El grafo entero se salta -- sin
      contexto recuperado no hay nada que el review/correct pueda
      evaluar.

      Caso A -- sin colecciones/archivos, con web_search=True: usa
      _answer_web_only(), el mismo camino simple que /query. Ver su
      docstring para por qué (confidence/reformulate/review están
      diseñados en torno al retrieval vectorial local, sin un análogo
      significativo para resultados web).

      Caso B -- con colecciones: el grafo corre exactamente igual que
      antes de esta feature (retrieve -> evaluate -> ... -> review ->
      [correct]), termina, y SOLO DESPUÉS se intenta el complemento web
      sobre el `answer` ya revisado/corregido por el grafo -- mismo
      _supplement_with_web() que usa /query, best-effort y no puede
      alterar lo que el grafo ya decidió. Los adjuntos ad-hoc, si los
      hay, viajan en RAGState.attachments y generate_node los inyecta
      en el prompt final (ver src/graph/nodes.py).
    """
    logger.info(
        f"[api] POST /query/agent | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r} "
        f"| web_search={request.web_search}"
    )

    _validate_web_search(request)
    _validate_generation_options(request)

    collections = _resolve_collections(request.collections)
    ephemeral = _resolve_ephemeral(request.conversation_id, ephemeral_store)
    if ephemeral is not None:
        collections = [*collections, ephemeral]

    attachments = _fetch_attachments(request.conversation_id, attachment_store)

    chat_memory = _build_chat_memory(request.chat_history)
    used_web_search = False
    web_sources: list[WebSource] = []
    quota_exceeded = False

    if not collections and not request.web_search:
        # Caso raw -- ver _answer_raw().
        answer = _answer_raw(request, chat_memory, attachments)
        confidence = 0.0
        reformulated = False

    elif not collections:
        # Caso A -- ver docstring de _answer_web_only.
        answer, web_sources = _answer_web_only(request, chat_memory, attachments)
        confidence = 0.0
        reformulated = False
        used_web_search = True

    else:
        # Caso normal: el grafo corre exactamente igual que antes de
        # esta feature -- ni graph/state.py (salvo el nuevo campo
        # attachments) ni graph/nodes.py cambiaron su lógica de
        # retrieval/review. top_k_initial/top_k_final/max_turns ya no
        # viven en RAGState (ver su docstring en src/graph/state.py):
        # retrieve_node y generate_node los resuelven internamente
        # contra settings, sin override por-request.
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
            "max_tokens": gen_kwargs["max_tokens"],
            "think_mode": gen_kwargs["think_mode"],
            "extra": gen_kwargs["extra"],
            "attachments": attachments,
        }

        # CompiledStateGraph.invoke() está tipado en la librería como
        # `dict[str, Any] | Any` (no como el StateT genérico), así que un
        # cast explícito es más honesto acá que ignorar el error a ciegas:
        # documenta justo el punto donde termina la precisión de LangGraph
        # y empieza la nuestra.
        #
        # ContextLimitExceeded puede escapar desde generate_node (ver su
        # docstring en src/graph/nodes.py) -- se traduce a 413 acá igual
        # que en /query.
        try:
            final_state = cast(RAGState, rag_graph.invoke(initial_state))
        except ContextLimitExceeded as e:
            raise HTTPException(status_code=413, detail=e.as_detail()) from e

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

    # Adjuntos ad-hoc de un solo uso -- ver docstring de AttachmentStore.
    _consume_attachments(request.conversation_id, attachment_store)

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
