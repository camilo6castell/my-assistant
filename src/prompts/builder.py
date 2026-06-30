"""
Constructor del prompt que se envía al LLM.

El historial de conversación NO se incluye aquí — viaja como mensajes
estructurados user/assistant a través de la API (build_messages en
generate.py). Incluirlo en el prompt también sería redundante.

Funciones exportadas:
  build_prompt             → prompt principal para generate_node
  build_review_prompt      → prompt para review_node (Gemini evalúa la respuesta)
  build_correction_prompt  → prompt para generate_node cuando el reviewer rechazó
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


def build_review_prompt(
    context_chunks: list[str],
    question: str,
    answer: str,
) -> str:
    """
    Prompt para review_node.

    Le entrega a Gemini los mismos chunks que usó generate_node,
    la pregunta original, y la respuesta producida. Le pide que
    evalúe dos dimensiones críticas de RAG:
      1. Anclaje: ¿cada afirmación tiene respaldo en el contexto?
      2. Citas: ¿la respuesta menciona las fuentes cuando las usa?

    Formato de respuesta esperado (JSON estricto) para parseo simple:
      { "passed": true }
      { "passed": false, "feedback": "motivo concreto del rechazo" }

    Se usa JSON en vez de texto libre para evitar parseos frágiles.
    Gemini recibe instrucción explícita de no añadir nada fuera del JSON.
    """
    return f"""
Eres un evaluador de calidad para un sistema RAG.
Tu tarea es revisar si una respuesta cumple dos criterios:

CRITERIO 1 — ANCLAJE:
Cada afirmación de la respuesta debe estar respaldada por el contexto.
Si la respuesta contiene información que NO aparece en el contexto (alucinación),
debe ser rechazada.

CRITERIO 2 — CITAS:
La respuesta debe mencionar de qué fuente proviene la información
cuando hace afirmaciones concretas (ej: "Según [fuente]...").
Si no cita ninguna fuente siendo que el contexto tiene metadatos de fuente,
debe ser rechazada.

========================================
CONTEXTO RECUPERADO
========================================

{build_context_block(context_chunks)}

========================================
PREGUNTA ORIGINAL
========================================

{question}

========================================
RESPUESTA A EVALUAR
========================================

{answer}

========================================
INSTRUCCIÓN
========================================

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional,
sin bloques de código, sin explicaciones fuera del JSON.

Si la respuesta cumple ambos criterios:
{{"passed": true}}

Si falla algún criterio:
{{"passed": false, "feedback": "descripción concreta del problema"}}
"""


def build_correction_prompt(
    context_chunks: list[str],
    question: str,
    previous_answer: str,
    feedback: str,
    mode: str,
) -> str:
    """
    Prompt para generate_node cuando el reviewer rechazó la respuesta anterior.

    Incluye la respuesta rechazada y el feedback del reviewer para que
    el modelo local corrija exactamente lo que falló, sin regenerar desde cero.
    """
    return f"""
Eres un asistente RAG.

Tu tarea es CORREGIR una respuesta previa que fue rechazada por un evaluador.

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
RESPUESTA ANTERIOR (RECHAZADA)
========================================

{previous_answer}

========================================
MOTIVO DEL RECHAZO
========================================

{feedback}

========================================
INSTRUCCIÓN
========================================

Genera una nueva respuesta que corrija el problema indicado.
Mantén lo que estaba bien. Corrige solo lo señalado.

========================================
RESPUESTA CORREGIDA
========================================
"""
