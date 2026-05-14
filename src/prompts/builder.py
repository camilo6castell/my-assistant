from typing import List

from src.config.settings import MAX_TURNS


def build_prompt(
    context_chunks: List[str],
    question: str,
    interpretative_mode: bool,
    chat_memory: List[dict],
) -> str:

    context_text = "\n\n---\n\n".join(context_chunks)

    history_text = ""

    for turn in chat_memory[-MAX_TURNS:]:

        history_text += (
            f"Usuario: {turn['user']}\n" f"Asistente: {turn['assistant']}\n\n"
        )

    if interpretative_mode:

        rules_block = """
REGLAS (MODO INTERPRETATIVO AVANZADO):

- Puedes conectar ideas entre fragmentos distintos.
- Puedes abstraer principios generales.
- Puedes formular síntesis conceptuales.
- Indica siempre fuentes.
- Nunca uses conocimiento externo.
"""

    else:

        rules_block = """
REGLAS (MODO RIGUROSO):

- Usa solo el contexto.
- No inventes.
- No uses conocimiento externo.
"""

    return f"""
Eres un asistente que responde
EXCLUSIVAMENTE usando el contexto proporcionado.

Historial reciente:
{history_text}

{rules_block}

CONTEXTO:
{context_text}

PREGUNTA:
{question}

RESPUESTA:
"""
