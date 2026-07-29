from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config.settings import settings
from src.context.manager import ContextManager
from src.retrieval.search import search
from src.utils.logger import logger

# ======================================================
# AUTH MIDDLEWARE
# ======================================================


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Validates Bearer token on every request to the MCP server."""

    async def dispatch(self, request: Request, call: Any) -> Response:
        token = settings.mcp_bearer_token

        if token:
            auth = request.headers.get("Authorization", "")
            parts = auth.split()

            if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Bearer token"},
                )

        return await call(request)  # type: ignore[no-any-return]


# ======================================================
# MCP SERVER
# ======================================================

server = MCPServer(
    name="MyAssistant RAG",
    title="MyAssistant RAG MCP Server",
    description="Private semantic memory engine — retrieval and FAISS search over collections.",
    version="1.0.0",
)


@server.tool(
    name="list_collections",
    description="Lists all available collections on disk in 'namespace/name' format.",
)
async def list_collections() -> list[str]:
    manager = ContextManager()
    return manager.list_all()


@server.tool(
    name="retrieve_chunks",
    description=(
        "Searches for relevant text chunks in the given collections. "
        "Returns results with text, source metadata, and a confidence score. "
        "Mode 'SOFT' generates 3 query variants (literal + 2 semantic); "
        "'HARD' uses the query verbatim."
    ),
)
async def retrieve_chunks(
    query: str,
    collections: list[str],
    mode: str = "HARD",
) -> dict[str, Any]:
    if mode not in ("SOFT", "HARD"):
        return {"error": f"Invalid mode {mode!r}. Use 'SOFT' or 'HARD'."}

    manager = ContextManager()

    for col_name in collections:
        loaded = manager.activate(col_name)
        if not loaded:
            logger.warning(f"[mcp] Collection not found or empty: {col_name}")

    loaded_collections = manager.get_loaded_collections()

    if not loaded_collections:
        return {"results": [], "confidence": 0.0, "collections_used": []}

    results, confidence = search(query, mode, loaded_collections)
    collections_used = list({r.collection for r in results})

    return {
        "results": [
            {
                "text": r.text,
                "source": r.source,
                "collection": r.collection,
                "page": r.page,
                "score": r.score,
            }
            for r in results
        ],
        "confidence": confidence,
        "collections_used": collections_used,
    }


# ======================================================
# APP FACTORY
# ======================================================
# SAFETY: The MCP server exposes your entire FAISS index to anyone who can
# reach it. In production:
#   - Keep MCP_HOST=127.0.0.1 so it is NOT reachable from the internet.
#   - Only the API server (API_HOST) should be exposed via Tailscale Funnel.
#   - Set MCP_BEARER_TOKEN to a non-empty value as a second line of defense.


def create_app() -> Starlette:
    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.mcp_allowed_hosts,
        ),
    )

    # CORS first so preflight OPTIONS requests (no auth header) pass
    mcp_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Auth middleware — runs after CORS for actual requests
    mcp_app.add_middleware(BearerTokenMiddleware)

    return mcp_app
