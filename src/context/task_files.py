"""
Almacén de archivos crudos para el modo Task: a diferencia de
EphemeralStore (src/context/ephemeral.py), acá NO hay chunking ni
embeddings -- el modo Task no hace retrieval, así que no tiene sentido
pagar ese costo. El archivo completo se guarda como texto y se inyecta
entero en el prompt (ver build_task_prompt() en src/prompts/builder.py).

Mismo motivo de existencia en memoria (no disco), mismo indexado por
conversation_id, y mismo mecanismo de TTL que EphemeralStore -- ver ese
docstring para el razonamiento completo de concurrencia/single-worker.
Reutiliza el mismo EPHEMERAL_TTL (src/api/deps.py) en vez de definir
uno propio: son dos formas de "contexto de una sola conversación" con
la misma semántica de vida útil.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from src.utils.logger import logger

# Extensiones de texto plano que tiene sentido inyectar crudas en un
# prompt de código. A diferencia de _SUPPORTED_SUFFIXES en
# api/routers/files.py (.pdf/.html/.txt, pensado para RAG), acá el caso
# de uso es "pegame este módulo para refactorizarlo" -- todo lo que sea
# texto plano de código/config aplica.
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


class TaskFileInfo(BaseModel):
    """Metadata de un archivo de Task (para respuestas de API)."""

    file_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime


@dataclass
class _TaskFile:
    filename: str
    content: str
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class _ConversationTaskFiles:
    """Estado en memoria de los archivos de Task de una conversación."""

    files: dict[str, _TaskFile] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


class TaskFileStore:
    """Archivos crudos en memoria, uno por conversation_id, para el modo Task."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationTaskFiles] = {}

    # -------------------------------------------------
    # ESCRITURA
    # -------------------------------------------------

    def add_file(self, conversation_id: str, filename: str, content: str) -> TaskFileInfo:
        file_id = uuid.uuid4().hex[:12]
        store = self._conversations.setdefault(conversation_id, _ConversationTaskFiles())

        task_file = _TaskFile(filename=filename, content=content)
        store.files[file_id] = task_file
        store.last_used = datetime.now(UTC)

        info = TaskFileInfo(
            file_id=file_id,
            filename=filename,
            size_bytes=len(content.encode()),
            uploaded_at=task_file.uploaded_at,
        )

        logger.info(
            f"[task_files] archivo agregado | conversation={conversation_id} "
            f"| file={filename} | file_id={file_id} | bytes={info.size_bytes}"
        )
        return info

    # -------------------------------------------------
    # BORRADO
    # -------------------------------------------------

    def remove_file(self, conversation_id: str, file_id: str) -> bool:
        """
        Borra un archivo puntual. Si era el último de la conversación,
        borra la conversación entera (mismo criterio que
        EphemeralStore.remove_file). Devuelve False si conversation_id
        o file_id no existen (idempotente: el router lo traduce a 404).
        """
        store = self._conversations.get(conversation_id)
        if store is None or file_id not in store.files:
            return False

        del store.files[file_id]
        store.last_used = datetime.now(UTC)

        if not store.files:
            del self._conversations[conversation_id]
            logger.info(
                f"[task_files] se borró el último archivo -> conversación "
                f"limpiada | conversation={conversation_id}"
            )
        else:
            logger.info(
                f"[task_files] archivo borrado | conversation={conversation_id} "
                f"| file_id={file_id} | archivos_restantes={len(store.files)}"
            )
        return True

    def remove_conversation(self, conversation_id: str) -> bool:
        """Borra todos los archivos de Task de una conversación (ej: al cerrarla en la UI)."""
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        if existed:
            logger.info(f"[task_files] conversación eliminada | conversation={conversation_id}")
        return existed

    # -------------------------------------------------
    # LECTURA
    # -------------------------------------------------

    def list_files(self, conversation_id: str) -> list[TaskFileInfo]:
        store = self._conversations.get(conversation_id)
        if store is None:
            return []
        return [
            TaskFileInfo(
                file_id=fid,
                filename=f.filename,
                size_bytes=len(f.content.encode()),
                uploaded_at=f.uploaded_at,
            )
            for fid, f in store.files.items()
        ]

    def list_contents(self, conversation_id: str) -> list[tuple[str, str]]:
        """(filename, content) de todos los archivos -- para build_task_prompt()."""
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
        Borra conversaciones sin actividad hace más de `ttl`. Llamado
        periódicamente junto con EphemeralStore.sweep_expired desde el
        mismo _cleanup_loop en src/api/app.py.
        """
        cutoff = datetime.now(UTC) - ttl
        expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

        for cid in expired:
            del self._conversations[cid]

        if expired:
            logger.info(f"[task_files] limpieza TTL | conversaciones eliminadas={len(expired)}")

        return len(expired)
