"""
Store for ad-hoc "attachment" files: files the user uploads to be
injected raw (without chunking or embeddings) into the prompt of their
NEXT query -- see inject_attachments() in src/prompts/builder.py and
QueryRequest in src/api/schemas/chat.py.

Unlike EphemeralStore (src/context/ephemeral.py), which indexes files
in FAISS for semantic retrieval and persists for the duration of the
conversation, here there is no indexing or persistence beyond a single
send: once a query is processed with attachments present, the router
consumes them (AttachmentStore.remove_conversation) and the list becomes
empty again -- see "Attachments" in RightSidebar.tsx, which refreshes
after each send to reflect this.

Applies in any response mode: with active collections, with none (see
_answer_raw in src/api/routers/chat.py), or with web_search. They are
not "RAG context" per se -- they are ad-hoc user context for this
specific question.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from src.utils.logger import logger

# Plain text extensions that make sense to inject raw into a prompt.
# Unlike _SUPPORTED_SUFFIXES in api/routers/files.py
# (.pdf/.html/.txt, meant for indexing in an ephemeral collection), here
# the typical use case is "paste me this specific module" -- any plain
# text code/config/data applies.
SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".sql",
    ".sh",
    ".env",
    ".cfg",
    ".ini",
    ".xml",
    ".css",
    ".html",
}

# Per-file size limit -- a giant text file (log, dataset) would break
# the context window just like silently rejecting it later in the
# context guard (see src/llm/context_guard.py), but rejecting it here
# gives an immediate specific error instead of a generic 413 only when
# sending the query.
MAX_FILE_BYTES = 512_000  # 500 KB


class AttachmentInfo(BaseModel):
    """Metadata for an attachment file (for API responses)."""

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
    """In-memory state for pending attachments in a conversation."""

    files: dict[str, _Attachment] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))


class AttachmentStore:
    """In-memory attachments, one per conversation_id, pending send."""

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationAttachments] = {}

    # -------------------------------------------------
    # WRITING
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
            f"[attachments] file added | conversation={conversation_id} "
            f"| file={filename} | file_id={file_id} | bytes={info.size_bytes}"
        )
        return info

    # -------------------------------------------------
    # DELETION
    # -------------------------------------------------

    def remove_file(self, conversation_id: str, file_id: str) -> bool:
        """Deletes a specific attachment (manual delete button in the UI, before sending)."""
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
        Deletes all pending attachments for a conversation.

        Called in two cases: (1) the user manually deletes all of them
        from the UI, or (2) -- the normal case -- the chat router
        automatically consumes them after processing a query that
        included them (see src/api/routers/chat.py), so the list is
        empty for the next message.
        """
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        if existed:
            logger.info(
                f"[attachments] conversation consumed/cleared | conversation={conversation_id}"
            )
        return existed

    # -------------------------------------------------
    # READING
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
        """(filename, content) of all attachments -- for inject_attachments()."""
        if conversation_id is None:
            return []
        store = self._conversations.get(conversation_id)
        if store is None:
            return []
        store.last_used = datetime.now(UTC)
        return [(f.filename, f.content) for f in store.files.values()]

    # -------------------------------------------------
    # MAINTENANCE
    # -------------------------------------------------

    def sweep_expired(self, ttl: timedelta) -> int:
        """
        Deletes conversations with unconsumed attachments older than `ttl`
        -- safety net for attachments the user uploaded but never sent.
        Called periodically alongside EphemeralStore.sweep_expired from
        the same _cleanup_loop in src/api/app.py.
        """
        cutoff = datetime.now(UTC) - ttl
        expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

        for cid in expired:
            del self._conversations[cid]

        if expired:
            logger.info(f"[attachments] TTL sweep | conversations deleted={len(expired)}")

        return len(expired)
