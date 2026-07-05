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


def build_system_prompt() -> str:
    return f"""
You are an assistant specialized in Retrieval-Augmented Generation (RAG).

Your knowledge for each response is limited to the context provided by the user. Always prioritize the retrieved context over any prior knowledge.

Guidelines:

- Respond in the same language as the user's question.
- Use only information that is supported by the provided context.
- Do not fabricate or infer facts that are not grounded in the context.
- If the context is insufficient to answer the question, explicitly say so.
- Integrate information from multiple fragments when they complement each other.
- If the retrieved fragments contain contradictions, point them out instead of resolving them yourself.
- Provide clear, precise, and well-structured answers.
- When the answer is supported by one or more fragments, cite their sources whenever possible.

The user will specify one of the following response modes:

HARD
{build_mode_rules(ChatMode.HARD)}

SOFT
{build_mode_rules(ChatMode.SOFT)}
"""


def build_reformulation_system_prompt() -> str:
    return """
You are an expert in semantic information retrieval for Retrieval-Augmented Generation (RAG).

Your task is to rewrite the user's question to maximize retrieval quality in a vector database.

Requirements:

- Preserve the original intent exactly.
- Do not answer the question.
- Do not introduce new facts or assumptions.
- Resolve ambiguity only when it can be inferred from the original wording.
- Prefer clear, specific, and self-contained questions.
- Replace vague references with explicit terms whenever possible.
- Keep the rewritten question concise and natural.
- Output only the rewritten question, with no explanations or additional text.
"""


def build_context_block(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)


def build_mode_rules(mode: str) -> str:
    if mode == ChatMode.HARD:
        return """
- Restrict the answer to information explicitly stated in the context.
- Avoid paraphrasing beyond what is necessary for readability.
- Do not generalize or draw implicit conclusions.
"""

    return """
- You may synthesize and connect information from multiple fragments.
- You may explain relationships and high-level implications that are directly supported by the context.
- Never introduce external knowledge or unsupported assumptions.
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
Mode: {"SOFT" if mode == ChatMode.SOFT else "HARD"}

Context:
{build_context_block(context_chunks)}

Question:
{question}
"""


def build_review_prompt(
    context_chunks: list[str],
    question: str,
    answer: str,
) -> str:
    """
    Build the prompt used by the review node.

    The reviewer validates that the generated answer satisfies the
    quality requirements of the RAG system before it is returned to
    the user.

    Evaluation criteria:
      1. Grounding: every factual statement must be supported by the
         retrieved context.
      2. Source attribution: whenever factual information is used,
         the answer should cite the corresponding source when available.

    The reviewer must return only a valid JSON object so the response
    can be parsed deterministically.
    """

    return f"""
You are a quality reviewer for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate whether the generated answer satisfies the required quality standards.

Evaluation criteria:

1. Grounding
- Every factual statement must be supported by the retrieved context.
- Reject the answer if it contains unsupported information, hallucinations, or external knowledge.

2. Source attribution
- When the retrieved context includes source metadata, the answer should cite the relevant source(s) for factual statements.
- Reject the answer if source citations are missing when they should reasonably be provided.

3. If the answer fails to meet the criteria, the value of "reason" must be one of:
- "grounding": the answer contains unsupported information, hallucinations, or external knowledge.
- "missing_sources": the answer is grounded in the context but fails to cite the relevant source(s) when source metadata is available.

Retrieved context:

{build_context_block(context_chunks)}

Original question:

{question}

Generated answer:

{answer}

Return only one valid JSON object.

If the answer satisfies every criterion:

{{"passed": true}}

Otherwise:

{{
  "passed": false,
  "reason": "<grounding|missing_sources>",
  "feedback": "<concise explanation of the problem>"
}}

Do not return markdown, code fences, explanations, or any text outside the JSON object.
"""


def build_correction_prompt(
    context_chunks: list[str],
    question: str,
    previous_answer: str,
    feedback: str,
    mode: str,
) -> str:
    """
    Build the prompt used to repair a response rejected by the review node.

    The model receives the original context, the rejected answer, and
    the reviewer's feedback. Its goal is to minimally modify the answer
    so that it satisfies every review criterion while preserving all
    correct information.
    """

    return f"""
The previous answer was rejected during the RAG review process.

Your task is to correct the answer, not to generate a completely new one.

Response mode: {mode}

Apply the rules associated with this response mode.

Retrieved context:

{build_context_block(context_chunks)}

Original question:

{question}

Rejected answer:

{previous_answer}

Reviewer feedback:

{feedback}

Instructions:

- Correct every issue identified by the reviewer.
- Preserve all information that is already correct.
- Modify only the parts necessary to address the review feedback.
- Keep the answer fully grounded in the retrieved context.
- Do not introduce external knowledge.
- Cite sources whenever appropriate.
- Respond in the same language as the original question.

Return only the corrected answer.
"""
