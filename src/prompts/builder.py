from typing import List

from src.chat.modes import (
    ChatMode,
)

from src.config.settings import (
    MAX_TURNS,
)


def build_history_block(
    chat_memory: List[dict],
) -> str:

    if not chat_memory:
        return "No hay historial previo."

    history_lines = []

    recent_turns = chat_memory[-MAX_TURNS:]

    for turn in recent_turns:

        history_lines.append(f"Usuario: {turn['user']}")

        history_lines.append(f"Asistente: {turn['assistant']}")

        history_lines.append("")

    return "\n".join(history_lines)


def build_context_block(
    context_chunks: List[str],
) -> str:

    return "\n\n---\n\n".join(context_chunks)


def build_rules_block(
    mode: str,
) -> str:

    if mode == ChatMode.INTERPRETATIVE:

        return """
REGLAS (MODO INTERPRETATIVO):

- Puedes conectar ideas entre múltiples fuentes.
- Puedes sintetizar conceptos.
- Puedes abstraer principios generales.
- Puedes explicar implicaciones teóricas.
- Mantente fiel al contexto.
- Nunca inventes información externa.
- Indica fuentes cuando sea posible.
"""

    return """
REGLAS (MODO RIGUROSO):

- Usa únicamente el contenido presente en el contexto.
- No inventes información.
- No uses conocimiento externo.
- Si algo no está en el contexto, dilo explícitamente.
- Prioriza precisión textual.
"""


def build_prompt(
    context_chunks: List[str],
    question: str,
    mode: str,
    chat_memory: List[dict],
) -> str:

    context_block = build_context_block(context_chunks)

    history_block = build_history_block(chat_memory)

    rules_block = build_rules_block(mode)

    return f"""
Eres un asistente RAG.

Tu tarea es responder preguntas usando
EXCLUSIVAMENTE el contexto proporcionado.

{rules_block}

========================================
HISTORIAL
========================================

{history_block}

========================================
CONTEXTO
========================================

{context_block}

========================================
PREGUNTA
========================================

{question}

========================================
RESPUESTA
========================================
"""
