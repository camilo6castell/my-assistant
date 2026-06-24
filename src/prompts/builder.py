"""
Constructor del prompt que se envía al LLM.

El historial de conversación NO se incluye aquí — viaja como mensajes
estructurados user/assistant a través de la API (build_messages en
generate.py). Incluirlo en el prompt también sería redundante.
"""

from src.chat.modes import ChatMode


def build_context_block(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)


def build_rules_block(mode: str) -> str:
    if mode == ChatMode.SOFT:
        return """
REGLAS (MODO SOFT):

- Puedes conectar ideas entre múltiples fuentes.
- Puedes sintetizar conceptos.
- Puedes abstraer principios generales.
- Puedes explicar implicaciones teóricas.
- Mantente fiel al contexto.
- Nunca inventes información externa.
- Indica fuentes cuando sea posible.
"""

    return """
REGLAS (MODO HARD):

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
) -> str:
    """
    Construye el prompt con el contexto recuperado y la pregunta.

    El parámetro chat_memory fue eliminado: el historial viaja como
    mensajes de API en build_messages(), no como texto en el prompt.
    """
    return f"""
Eres un asistente RAG.

Tu tarea es responder preguntas usando
EXCLUSIVAMENTE el contexto proporcionado.

{build_rules_block(mode)}

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
