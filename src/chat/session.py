"""
src/chat/session.py

Estado de la sesión de chat activa: contextos cargados, modo y memoria.
"""

from src.context.manager import ContextManager

from src.chat.modes import (
    RIGOROUS,
    INTERPRETATIVE,
)


class ChatSession:

    def __init__(self):
        self.context_manager = ContextManager()
        self.interpretative_mode = False
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

    def get_prompt_header(self) -> str:
        """
        Genera el prompt del input acortando los nombres de colección
        al basename (la parte después de /) para que el header no se
        desborde en pantallas estrechas.

        Ejemplos:
          [debord, freud | INTERP] >
          [SIN-CONTEXTO | RIG] >
        """
        active = self.get_active_contexts()

        mode_label = "INTERP" if self.interpretative_mode else "RIG"

        if not active:
            ctx_label = "SIN-CONTEXTO"
        else:
            # Usa solo el basename de cada colección (parte tras "/")
            names = [ctx.split("/")[-1] for ctx in sorted(active)]
            ctx_label = ", ".join(names)

        return f"[{ctx_label} | {mode_label}] > "
