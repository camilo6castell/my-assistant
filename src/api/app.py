"""
API REST del sistema RAG.

Expone el RAG como servicio HTTP para que herramientas externas
(n8n, scripts, otros servicios) puedan consumirlo sin depender
de la interfaz CLI.

Endpoints:
  GET  /health          health check (para n8n y load balancers)
  GET  /collections     lista colecciones disponibles en disco
  POST /query           pipeline lineal: retrieve → generate
  POST /query/agent     pipeline LangGraph: retrieve → evaluate → [reformulate] → generate

Diseño stateless:
  La API no mantiene estado de sesión entre requests. El cliente
  (n8n, frontend, etc.) es responsable de:
    - Enviar las colecciones en cada request.
    - Mantener y reenviar el chat_history si quiere memoria conversacional.

  Esto facilita el escalado horizontal y simplifica la lógica del servidor.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CollectionsResponse,
    QueryRequest,
    QueryResponse,
)
from src.chat.types import TurnMemory
from src.context.manager import ContextManager, LoadedCollection
from src.graph import RAGState, build_rag_graph
from src.llm.generate import ask_llm
from src.prompts.builder import build_prompt
from src.retrieval.search import search
from src.utils.logger import logger


# ======================================================
# LIFESPAN — recursos compartidos
# ======================================================

# El grafo LangGraph se compila una vez al arrancar y se reutiliza
# en todos los requests — compilarlo en cada request añadiría ~100ms.
_rag_graph: object = None
_context_manager: ContextManager = ContextManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _rag_graph
    logger.info("[api] Compilando grafo RAG...")
    _rag_graph = build_rag_graph()
    logger.info("[api] API lista.")
    yield
    logger.info("[api] Cerrando API.")


# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="MyAssistant RAG API",
    description="API REST para el sistema RAG local con LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS abierto para desarrollo local — n8n corre en un puerto distinto.
# En producción restringir a los orígenes permitidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# HELPERS
# ======================================================


def _resolve_collections(tokens: list[str]) -> list[LoadedCollection]:
    """
    Carga las colecciones pedidas en el request y las devuelve.

    Cada request carga sus propias colecciones desde disco. No hay
    estado compartido entre requests — ContextManager es local a
    esta función, no el singleton global.

    El singleton _context_manager solo se usa para listar colecciones
    disponibles (GET /collections), no para cargar contexto.
    """
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


# ======================================================
# ENDPOINTS
# ======================================================


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check. n8n lo usa para verificar que el servicio está vivo."""
    return {"status": "ok"}


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections() -> CollectionsResponse:
    """
    Lista todas las colecciones disponibles en disco.

    n8n puede llamar a este endpoint para poblar dinámicamente
    un selector de colecciones en un workflow.
    """
    collections = _context_manager.list_all()
    return CollectionsResponse(collections=collections)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Pipeline lineal: retrieve → generate.

    Equivale a _handle_question() del CLI pero sin estado de sesión.
    Útil cuando no se necesita adaptive retrieval y se prioriza velocidad.
    """
    logger.info(
        f"[api] POST /query | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r}"
    )

    collections = _resolve_collections(request.collections)
    chat_memory = _build_chat_memory(request.chat_history)

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
        question=request.question,
        mode=request.mode,
    )

    answer = ask_llm(prompt=prompt, chat_memory=chat_memory)

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        collections_used=[c["collection_name"] for c in collections],
    )


@app.post("/query/agent", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """
    Pipeline LangGraph: retrieve → evaluate → [reformulate →] generate.

    Equivale a _handle_agent_question() del CLI. Si la confidence
    inicial es baja, el grafo reformula la query con el LLM y reintenta
    el retrieval antes de generar la respuesta.
    """
    logger.info(
        f"[api] POST /query/agent | collections={request.collections} "
        f"| mode={request.mode} | question={request.question!r}"
    )

    collections = _resolve_collections(request.collections)
    chat_memory = _build_chat_memory(request.chat_history)

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
    }

    final_state: RAGState = _rag_graph.invoke(initial_state)  # type: ignore[union-attr]

    answer = final_state["answer"]

    if not answer:
        raise HTTPException(
            status_code=422,
            detail="No relevant context found for the given question and collections.",
        )

    return QueryResponse(
        answer=answer,
        confidence=final_state["confidence"],
        collections_used=[c["collection_name"] for c in collections],
        reformulated=final_state["reformulated"],
    )
