"""
Almacén de archivos "adjuntos" ad-hoc: archivos que el usuario sube para
que se inyecten crudos (sin chunking ni embeddings) en el prompt de su
PRÓXIMA query -- ver inject_attachments() en src/prompts/builder.py y
QueryRequest en src/api/schemas/chat.py.

A diferencia de EphemeralStore (src/context/ephemeral.py), que indexa
archivos en FAISS para retrieval semántico y persiste mientras dure la
conversación, acá no hay indexado ni persistencia más allá de un solo
envío: una vez que una query se procesa con archivos adjuntos
presentes, el router los consume (AttachmentStore.remove_conversation)
y la lista vuelve a estar vacía -- ver "Archivos adjuntos" en
RightSidebar.tsx, que se refresca después de cada envío para reflejar
esto.

Aplican en cualquier modo de respuesta: con colecciones activas, sin
ninguna (ver _answer_raw en src/api/routers/chat.py), o con
web_search. No son "contexto RAG" en sí mismos -- son contexto puntual
del usuario para esta pregunta.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from src.utils.logger import logger

# Extensiones de texto plano que tiene sentido inyectar crudas en un
# prompt. A diferencia de _SUPPORTED_SUFFIXES en api/routers/files.py
# (.pdf/.html/.txt, pensado para indexar en una colección efímera), acá
# el caso de uso típico es "pegame este módulo puntual" -- todo lo que
# sea texto plano de código/config/datos aplica.
SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".yaml", ".yml", ".toml", ".csv", ".sql", ".sh",
    ".env", ".cfg", ".ini", ".xml", ".css", ".html",
}

# Límite de tamaño por archivo -- un archivo de texto gigante (log,
# dataset) reventaría el context window igual que rechazarlo silencioso
# más adelante en el guard de contexto (ver src/llm/context_guard.py),
# pero rechazarlo acá da un error inmediato y específico en vez de un
# 413 genérico recién al enviar la query.
MAX_FILE_BYTES = 512_000  # 500 KB


class AttachmentInfo(BaseModel):
    """Metadata de un archivo adjunto (para respuestas de API)."""

    file_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime


@dataclass
class _Attachment:
    filename: str
    content: str
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class _ConversationAttachments:
    """Estado en memoria de los adjuntos pendientes de una conversación."""

    files: dict[str, _Attachment] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


class AttachmentStore:
    """Archivos adjuntos en memoria, uno por conversation_id, pendientes de envío."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationAttachments] = {}

    # -------------------------------------------------
    # ESCRITURA
    # -------------------------------------------------

    def add_file(self, conversation_id: str, filename: str, content: str) -> AttachmentInfo:
        file_id = uuid.uuid4().hex[:12]
        store = self._conversations.setdefault(conversation_id, _ConversationAttachments())

        attachment = _Attachment(filename=filename, content=content)
        store.files[file_id] = attachment
        store.last_used = datetime.now(UTC)

        info = AttachmentInfo(
            file_id=file_id,
            filename=filename,
            size_bytes=len(content.encode()),
            uploaded_at=attachment.uploaded_at,
        )

        logger.info(
            f"[attachments] archivo agregado | conversation={conversation_id} "
            f"| file={filename} | file_id={file_id} | bytes={info.size_bytes}"
        )
        return info

    # -------------------------------------------------
    # BORRADO
    # -------------------------------------------------

    def remove_file(self, conversation_id: str, file_id: str) -> bool:
        """Borra un adjunto puntual (botón de borrar manual en la UI, antes de enviar)."""
        store = self._conversations.get(conversation_id)
        if store is None or file_id not in store.files:
            return False

        del store.files[file_id]
        store.last_used = datetime.now(UTC)

        if not store.files:
            del self._conversations[conversation_id]
        return True

    def remove_conversation(self, conversation_id: str) -> bool:
        """
        Borra todos los adjuntos pendientes de una conversación.

        Se llama en dos casos: (1) el usuario los borra todos manualmente
        desde la UI, o (2) -- el caso normal -- el router de chat los
        consume automáticamente después de procesar una query que los
        incluyó (ver src/api/routers/chat.py), para que la lista quede
        vacía de cara al próximo mensaje.
        """
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        if existed:
            logger.info(f"[attachments] conversación consumida/limpiada | conversation={conversation_id}")
        return existed

    # -------------------------------------------------
    # LECTURA
    # -------------------------------------------------

    def list_files(self, conversation_id: str) -> list[AttachmentInfo]:
        store = self._conversations.get(conversation_id)
        if store is None:
            return []
        return [
            AttachmentInfo(
                file_id=fid,
                filename=f.filename,
                size_bytes=len(f.content.encode()),
                uploaded_at=f.uploaded_at,
            )
            for fid, f in store.files.items()
        ]

    def list_contents(self, conversation_id: str | None) -> list[tuple[str, str]]:
        """(filename, content) de todos los adjuntos -- para inject_attachments()."""
        if conversation_id is None:
            return []
        store = self._conversations.get(conversation_id)
        if store is None:
            return []
        store.last_used = datetime.now(UTC)
        return [(f.filename, f.content) for f in store.files.values()]

    # -------------------------------------------------
    # MANTENIMIENTO
    # -------------------------------------------------

    def sweep_expired(self, ttl: timedelta) -> int:
        """
        Borra conversaciones con adjuntos sin consumir hace más de `ttl`
        -- red de seguridad para adjuntos que el usuario subió pero nunca
        llegó a enviar. Llamado periódicamente junto con
        EphemeralStore.sweep_expired desde el mismo _cleanup_loop en
        src/api/app.py.
        """
        cutoff = datetime.now(UTC) - ttl
        expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

        for cid in expired:
            del self._conversations[cid]

        if expired:
            logger.info(f"[attachments] limpieza TTL | conversaciones eliminadas={len(expired)}")

        return len(expired)
