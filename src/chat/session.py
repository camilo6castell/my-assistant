from __future__ import annotations

from src.chat.modes import SOFT, HARD
from src.chat.types import TurnMemory
from src.context.manager import ContextManager, LoadedCollection


class ChatSession:

    def __init__(self) -> None:
        self.context_manager: ContextManager = ContextManager()
        self.soft_mode: bool = True
        self.chat_memory: list[TurnMemory] = []

    # =====================================================
    # CONTEXTS
    # =====================================================

    def load_context(self, pattern: str) -> list[str]:
        return self.context_manager.activate(pattern)

    def unload_context(self, pattern: str) -> list[str]:
        return self.context_manager.deactivate(pattern)

    def clear_contexts(self) -> None:
        self.context_manager.clear()

    def get_active_contexts(self) -> list[str]:
        return self.context_manager.get_active()

    def get_loaded_collections(self) -> list[LoadedCollection]:
        return self.context_manager.get_loaded_collections()

    # =====================================================
    # MODE
    # =====================================================

    @property
    def mode(self) -> str:
        return SOFT if self.soft_mode else HARD

    def toggle_mode(self) -> str:
        self.soft_mode = not self.soft_mode
        return self.mode

    # =====================================================
    # MEMORY
    # =====================================================

    def reset_memory(self) -> None:
        self.chat_memory = []

    def add_to_memory(self, user: str, assistant: str) -> None:
        self.chat_memory.append(TurnMemory(user=user, assistant=assistant))

    # =====================================================
    # UI
    # =====================================================

    def get_prompt_header(self) -> str:
        """
        Genera el prompt del input acortando los nombres de colección
        al basename (parte después de '/') para evitar desbordamiento.

        Ejemplos:
          [debord, freud | SOFT] >
          [No-context | HARD] >
        """
        active: list[str] = self.get_active_contexts()
        mode_label: str = "SOFT" if self.soft_mode else "HARD"

        if not active:
            ctx_label = "No-context"
        else:
            names = [ctx.split("/")[-1] for ctx in sorted(active)]
            ctx_label = ", ".join(names)

        return f"[{ctx_label} | {mode_label}] > "
