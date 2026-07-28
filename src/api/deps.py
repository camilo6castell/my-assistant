"""
Shared dependencies between routers (FastAPI Depends() pattern).

Heavy resources -- the compiled LangGraph graph and the ephemeral
collection store -- are instantiated once and live here so that any
router can request them via Depends() without importing app.py directly.
This avoids the import cycle router -> app -> router that would appear
if each router read these objects from app.py.

app.py calls set_rag_graph() once inside lifespan(), when the process
starts.
"""

from __future__ import annotations

from datetime import timedelta

from langgraph.graph.state import CompiledStateGraph

from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager
from src.graph.state import RAGState

# Inactivity TTL for ephemeral collections (see EphemeralStore.sweep_expired)
# and for unsent file attachments (see AttachmentStore.sweep_expired) --
# same value, same semantics of "single-conversation context with no
# activity".
EPHEMERAL_TTL = timedelta(hours=6)

_rag_graph: CompiledStateGraph[RAGState] | None = None
_context_manager: ContextManager = ContextManager()
_ephemeral_store: EphemeralStore = EphemeralStore()
_attachment_store: AttachmentStore = AttachmentStore()


def set_rag_graph(graph: CompiledStateGraph[RAGState]) -> None:
    global _rag_graph
    _rag_graph = graph


def get_rag_graph() -> CompiledStateGraph[RAGState]:
    # Populated in lifespan() when the app starts; in a real request it
    # is never None. The assert makes it explicit for mypy and acts as a
    # runtime safety net if something invokes this before startup.
    assert _rag_graph is not None, "_rag_graph not initialized: missing lifespan()"
    return _rag_graph


def get_context_manager() -> ContextManager:
    return _context_manager


def get_ephemeral_store() -> EphemeralStore:
    return _ephemeral_store


def get_attachment_store() -> AttachmentStore:
    return _attachment_store
