from __future__ import annotations

from src.context.manager import ContextManager, LoadedCollection
from src.domain.models import ChatMode, TurnMemory


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
        return ChatMode.SOFT if self.soft_mode else ChatMode.HARD

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
        Generates the prompt for the input by shortening the collection names
        to their basename (the part after '/') to avoid overflow.

        Examples:
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

        return f"[ {ctx_label} | {mode_label} ]\n> "
