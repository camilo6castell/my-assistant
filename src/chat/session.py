from src.context.manager import ContextManager

from src.chat.modes import (
    RIGOROUS,
    INTERPRETATIVE,
)


class ChatSession:

    def __init__(self):

        self.context_manager = ContextManager()

        self.interpretative_mode = True

        self.chat_memory = []

    # =====================================================
    # CONTEXTS
    # =====================================================

    def load_context(self, pattern: str):
        return self.context_manager.activate(pattern)

    def unload_context(self, pattern: str):
        return self.context_manager.deactivate(pattern)

    def clear_contexts(self):
        self.context_manager.clear()

    def get_active_contexts(self):
        return self.context_manager.get_active()

    # =====================================================
    # MODE
    # =====================================================

    @property
    def mode(self):

        if self.interpretative_mode:
            return INTERPRETATIVE

        return RIGOROUS

    def toggle_mode(self):

        self.interpretative_mode = not self.interpretative_mode

        return self.mode

    # =====================================================
    # MEMORY
    # =====================================================

    def reset_memory(self):
        self.chat_memory = []

    def add_to_memory(
        self,
        user: str,
        assistant: str,
    ):

        self.chat_memory.append(
            {
                "user": user,
                "assistant": assistant,
            }
        )

    # =====================================================
    # UI
    # =====================================================

    def get_prompt_header(self):

        active = self.get_active_contexts()

        if not active:
            ctx = "SIN-CONTEXTO"
        else:
            ctx = ", ".join(sorted(active))

        return f"[{ctx} | {self.mode}] > "
