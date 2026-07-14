"""
Dependencias compartidas entre routers (patrón FastAPI Depends()).

Los recursos pesados -- el grafo LangGraph compilado y el almacén de
colecciones efímeras -- se instancian una sola vez y viven acá para que
cualquier router los pida vía Depends() sin importar app.py directamente.
Eso evita el ciclo de imports router -> app -> router que aparecería si
cada router leyera estos objetos desde app.py.

app.py llama a set_rag_graph() una sola vez dentro de lifespan(), al
arrancar el proceso.
"""

from __future__ import annotations

from datetime import timedelta

from langgraph.graph.state import CompiledStateGraph

from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager
from src.graph.state import RAGState

# TTL de inactividad para colecciones efímeras (ver EphemeralStore.sweep_expired)
# y para archivos adjuntos sin enviar (ver AttachmentStore.sweep_expired) --
# mismo valor, misma semántica de "contexto de una sola conversación sin
# actividad".
EPHEMERAL_TTL = timedelta(hours=6)

_rag_graph: CompiledStateGraph[RAGState] | None = None
_context_manager: ContextManager = ContextManager()
_ephemeral_store: EphemeralStore = EphemeralStore()
_attachment_store: AttachmentStore = AttachmentStore()


def set_rag_graph(graph: CompiledStateGraph[RAGState]) -> None:
    global _rag_graph
    _rag_graph = graph


def get_rag_graph() -> CompiledStateGraph[RAGState]:
    # Se puebla en lifespan() al arrancar la app; en un request real
    # nunca es None. El assert lo hace explícito para mypy y actúa como
    # red de seguridad en runtime si algo invoca esto antes del startup.
    assert _rag_graph is not None, "_rag_graph no inicializado: falta lifespan()"
    return _rag_graph


def get_context_manager() -> ContextManager:
    return _context_manager


def get_ephemeral_store() -> EphemeralStore:
    return _ephemeral_store


def get_attachment_store() -> AttachmentStore:
    return _attachment_store
