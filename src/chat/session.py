"""
src/chat/session.py

Estado de la sesión de chat activa: contextos cargados, modo y memoria.
"""

from typing import Any

from src.context.manager import ContextManager

from src.chat.modes import (
    RIGOROUS,
    INTERPRETATIVE,
)


class ChatSession:

    def __init__(self) -> None:
        self.context_manager: ContextManager = ContextManager()
        self.interpretative_mode: bool = False
        self.chat_memory: list[dict[str, str]] = []

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

    # =====================================================
    # MODE
    # =====================================================

    @property
    def mode(self) -> str:
        if self.interpretative_mode:
            return INTERPRETATIVE
        return RIGOROUS

    def toggle_mode(self) -> str:
        self.interpretative_mode = not self.interpretative_mode
        return self.mode

    # =====================================================
    # MEMORY
    # =====================================================

    def reset_memory(self) -> None:
        self.chat_memory = []

    def add_to_memory(
        self,
        user: str,
        assistant: str,
    ) -> None:
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
        active: list[str] = self.get_active_contexts()

        mode_label: str = "INTERP" if self.interpretative_mode else "RIG"

        if not active:
            ctx_label: str = "SIN-CONTEXTO"
        else:
            # Usa solo el basename de cada colección (parte tras "/")
            names: list[str] = [ctx.split("/")[-1] for ctx in sorted(active)]
            ctx_label = ", ".join(names)

        return f"[{ctx_label} | {mode_label}] > "
