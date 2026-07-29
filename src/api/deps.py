"""
Shared dependencies between routers (FastAPI Depends() pattern).

Ephemeral collection store, attachment store, and context manager are
instantiated once per process and live here so that any router can
request them via Depends() without importing app.py directly.
This avoids the import cycle router -> app -> router that would appear
if each router read these objects from app.py.
"""

from __future__ import annotations

from datetime import timedelta

from src.context.attachments import AttachmentStore
from src.context.ephemeral import EphemeralStore
from src.context.manager import ContextManager

# Inactivity TTL for ephemeral collections (see EphemeralStore.sweep_expired)
# and for unsent file attachments (see AttachmentStore.sweep_expired) --
# same value, same semantics of "single-conversation context with no
# activity".
EPHEMERAL_TTL = timedelta(hours=6)

_context_manager: ContextManager = ContextManager()
_ephemeral_store: EphemeralStore = EphemeralStore()
_attachment_store: AttachmentStore = AttachmentStore()


def get_context_manager() -> ContextManager:
    return _context_manager


def get_ephemeral_store() -> EphemeralStore:
    return _ephemeral_store


def get_attachment_store() -> AttachmentStore:
    return _attachment_store
