"""
API REST del sistema RAG.

Expone el RAG como servicio HTTP para que herramientas externas
(n8n, el frontend React, scripts, otros servicios) puedan consumirlo
sin depender de la interfaz CLI. Todos los clientes hablan con los
mismos endpoints -- no hay archivos de API por-cliente, la separación
es por dominio funcional (ver src/api/routers/).

Endpoints (todos bajo /api/v1, ver routers/ para el detalle):
  chat.py    /collections, /query, /query/agent  -- funciones regulares
  files.py   /files                               -- upload + colecciones efímeras
  config.py  /config/providers                    -- capacidades por provider (solo lectura)

GET /health queda fuera de /api/v1 a propósito: es un liveness probe
para load balancers, no un recurso versionado de la API.

Diseño stateless (con una excepción acotada):
  La API no mantiene estado de sesión entre requests. El cliente es
  responsable de enviar las colecciones y el chat_history en cada
  request. La única excepción es EphemeralStore (src/context/ephemeral.py):
  los archivos subidos "solo para esta conversación" sí viven en memoria
  del servidor, porque el chunking/embeddings no se puede hacer en el
  navegador. Se limpian solos por TTL (ver _cleanup_loop) o explícitamente
  vía DELETE /api/v1/files/{conversation_id}.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import deps
from src.api.routers import chat, config, files
from src.graph import build_rag_graph
from src.utils.logger import logger

# Cada cuánto se revisan conversaciones efímeras inactivas -- más frecuente
# que el TTL mismo para que la limpieza sea razonablemente puntual sin
# ser costosa (sweep_expired es O(conversaciones activas), trivial).
_CLEANUP_INTERVAL_SECONDS = 60 * 30


async def _cleanup_loop() -> None:
    """Tarea de fondo: barre conversaciones efímeras vencidas por TTL."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        deps.get_ephemeral_store().sweep_expired(deps.EPHEMERAL_TTL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("[api] Compilando grafo RAG...")
    deps.set_rag_graph(build_rag_graph())

    cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("[api] API lista.")

    yield

    cleanup_task.cancel()
    logger.info("[api] Cerrando API.")


# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="MyAssistant RAG API",
    description="API REST para el sistema RAG local con LangGraph.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS abierto para desarrollo local -- n8n y el frontend React corren en
# puertos distintos. En producción restringir a los orígenes permitidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(files.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Red de seguridad para cualquier excepción no capturada dentro de un
    endpoint (bugs de prompts, del grafo, lo que sea).

    Sin esto, Starlette resuelve una excepción no manejada con su propio
    ServerErrorMiddleware, que envuelve por FUERA al CORSMiddleware que
    agregamos arriba -- la respuesta 500 resultante nunca pasa por
    CORSMiddleware y llega al navegador sin headers CORS. El navegador
    entonces bloquea la lectura de esa respuesta y la reporta como error
    de red, no como el 500 que en realidad es -- exactamente el síntoma
    de "no se pudo conectar con el backend" aunque el backend sí
    respondió (y el log del servidor sí tiene el traceback real).
    Registrar un handler acá hace que FastAPI lo resuelva vía
    ExceptionMiddleware, que sí queda DENTRO de CORSMiddleware.
    """
    logger.exception(f"[api] Excepción no manejada en {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Revisá los logs del backend."},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check. n8n y los load balancers lo usan para verificar que el servicio está vivo."""
    return {"status": "ok"}
