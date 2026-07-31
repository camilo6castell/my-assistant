"""
REST API for the RAG system.

Exposes the RAG as an HTTP service so that external tools
(n8n, the React frontend, scripts, other services) can consume it
without depending on the CLI interface. All clients talk to the
same endpoints -- there are no per-client API files; separation
is by functional domain (see src/api/routers/).

Endpoints (all under /api/v1, see routers/ for details):
  chat.py        /collections, /query               -- regular functions
  demo.py        /demo/query                         -- streaming demo
  files.py       /files                              -- upload + ephemeral collections
  config.py      /config/providers                   -- per-provider capabilities (read-only)
  attachments.py /attachments                        -- ad-hoc single-use
                                                          attachments (see
                                                          src/context/attachments.py)

GET /health is intentionally outside /api/v1: it is a liveness probe
for load balancers, not a versioned API resource.

Stateless design (with one narrow exception):
  The API does not maintain session state between requests. The client is
  responsible for sending collections and chat_history with every
  request. The only exception is EphemeralStore (src/context/ephemeral.py):
  files uploaded "just for this conversation" do live in server
  memory, because chunking/embeddings cannot be done in the
  browser. They are cleaned up automatically by TTL (see _cleanup_loop) or
  explicitly via DELETE /api/v1/files/{conversation_id}.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import deps
from src.api.routers import attachments, chat, config, demo, files
from src.mcp_server.server import create_app as create_mcp_app
from src.mcp_server.server import get_mcp_lifespan
from src.utils.logger import logger

# How often inactive ephemeral conversations are checked -- more frequent
# than the TTL itself so cleanup is reasonably punctual without
# being costly (sweep_expired is O(active conversations), trivial).
_CLEANUP_INTERVAL_SECONDS = 60 * 30


async def _cleanup_loop() -> None:
    """Background task: sweeps expired ephemeral collections and unconsumed attachments."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        deps.get_ephemeral_store().sweep_expired(deps.EPHEMERAL_TTL)
        deps.get_attachment_store().sweep_expired(deps.EPHEMERAL_TTL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("[api] API ready.")

    # Mounted sub-app lifespans are not called by Starlette, so we
    # initialise the MCP session manager ourselves (see server.py:get_mcp_lifespan).
    async with get_mcp_lifespan():
        yield

    cleanup_task.cancel()
    logger.info("[api] Shutting down API.")


# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="Ragsody API",
    description="REST API for the local RAG system.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS open for local development -- n8n and the React frontend run on
# different ports. In production, restrict to allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(files.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(attachments.router, prefix="/api/v1")
app.include_router(demo.router, prefix="/api/v1")

# Mount the MCP server under /mcp -- available at http://host:port/mcp/
# (uses streamable_http_path="/" so the final path after mounting matches)
mcp_app = create_mcp_app(streamable_http_path="/")
app.mount("/mcp", mcp_app)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Safety net for any unhandled exception inside an
    endpoint (prompt bugs, graph bugs, anything).

    Without this, Starlette handles an unhandled exception with its own
    ServerErrorMiddleware, which wraps OUTSIDE the CORSMiddleware we
    added above -- the resulting 500 response never passes through
    CORSMiddleware and reaches the browser without CORS headers. The
    browser then blocks reading the response and reports it as a network
    error, not the actual 500 -- exactly the symptom of "could not
    connect to the backend" even though the backend did respond (and the
    server log does have the real traceback).
    Registering a handler here makes FastAPI resolve it via
    ExceptionMiddleware, which sits INSIDE CORSMiddleware.
    """
    logger.exception(f"[api] Unhandled exception at {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the backend logs."},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check. n8n and load balancers use it to verify the service is alive."""
    return {"status": "ok"}
