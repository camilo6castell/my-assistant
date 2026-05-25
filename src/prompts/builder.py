from src.chat.modes import ChatMode
from src.chat.types import TurnMemory
from src.config.settings import MAX_TURNS


def build_history_block(chat_memory: list[TurnMemory]) -> str:
    if not chat_memory:
        return "No hay historial previo."

    lines: list[str] = []

    for turn in chat_memory[-MAX_TURNS:]:
        lines.append(f"Usuario: {turn['user']}")
        lines.append(f"Asistente: {turn['assistant']}")
        lines.append("")

    return "\n".join(lines)


def build_context_block(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)


def build_rules_block(mode: str) -> str:
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
    context_chunks: list[str],
    question: str,
    mode: str,
    chat_memory: list[TurnMemory],
) -> str:
    return f"""
Eres un asistente RAG.

Tu tarea es responder preguntas usando
EXCLUSIVAMENTE el contexto proporcionado.

{build_rules_block(mode)}

========================================
HISTORIAL
========================================

{build_history_block(chat_memory)}

========================================
CONTEXTO
========================================

{build_context_block(context_chunks)}

========================================
PREGUNTA
========================================

{question}

========================================
RESPUESTA
========================================
"""
