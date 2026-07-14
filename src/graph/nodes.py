"""
Nodos del grafo RAG.

Cada función recibe el RAGState completo y devuelve un dict con solo
los campos que modifica — LangGraph hace el merge. Ningún nodo sabe
de qué nodo viene ni a cuál va: esa lógica vive en graph.py.

Nodos:
  retrieve_node    ejecuta search() con la question actual
  evaluate_node    compara confidence con settings.confidence_limit
  reformulate_node llama al LLM para reescribir la query
  generate_node    construye el prompt y llama al LLM

--- Métrica: confidence promedio con umbral calibrado ---

La confidence es el promedio de scores coseno del top-K devuelto por
search(). bge-small-en-v1.5 produce scores en un rango empíricamente
observado con estos documentos:

  on-topic:  >= 0.80   (query semánticamente alineada con el doc)
  off-topic: ~0.75     (query sin match real, e.g. gatos vs Freud)

Por eso confidence_limit=0.80 en settings. Si tu modelo o corpus
produce rangos distintos, ajusta CONFIDENCE_LIMIT en .env.
"""

from __future__ import annotations

import json

from src.config.settings import settings
from src.context.manager import LoadedCollection
from src.graph.state import RAGState, RAGStateUpdate
from src.llm.context_guard import ContextLimitExceeded, check_context_fit
from src.llm.generate import ask_llm, ask_llm_internal
from src.llm.roles import LLMRole
from src.prompts.builder import (
    build_correction_prompt,
    build_prompt,
    build_reformulation_system_prompt,
    build_review_prompt,
    build_review_system_prompt,
    build_system_prompt,
    inject_attachments,
)
from src.retrieval.search import search
from src.utils.logger import logger

# Máximo de ciclos generate → review → generate antes de dar la respuesta
# tal como está. Valor de 1 = un solo reintento (2 llamadas a generate total).
# Subir esto consume más tokens/tiempo: en un local runner es un tradeoff real.
MAX_REVIEW_ATTEMPTS = 1

# ======================================================
# RETRIEVE
# ======================================================


def retrieve_node(state: RAGState) -> RAGStateUpdate:
    """Recupera chunks relevantes desde las colecciones FAISS activas."""
    logger.info(f"[graph] retrieve_node | question={state['question']!r} | mode={state['mode']}")

    results, confidence = search(
        question=state["question"],
        mode=state["mode"],
        collections=state["collections"],
        top_k_initial=state.get("top_k_initial"),
        top_k_final=state.get("top_k_final"),
    )

    logger.info(f"[graph] retrieve_node | chunks={len(results)} | confidence={confidence:.4f}")

    return {"results": results, "confidence": confidence}


# ======================================================
# EVALUATE
# ======================================================


def evaluate_node(state: RAGState) -> RAGStateUpdate:
    """
    Nodo de evaluación. No modifica el estado: solo registra la decisión
    que tomará route_after_evaluate.
    """
    confidence = state["confidence"]
    reformulated = state["reformulated"]
    limit = settings.confidence_limit

    if reformulated:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} | ya reformulado → generando"
        )
    elif confidence < limit:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} < limit={limit} → reformulando"
        )
    else:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} >= limit={limit} → generando"
        )

    return {}


# ======================================================
# REFORMULATE
# ======================================================


def _format_collection_names(collections: list[LoadedCollection]) -> str:
    """
    Convierte paths internos en nombres legibles para el LLM.

    'sociologia/George-Orwell_1984' → 'George Orwell - 1984 (sociologia)'
    """
    result = []
    for c in collections:
        raw: str = c["collection_name"]  # ej: sociologia/George-Orwell_1984
        parts = raw.split("/", 1)
        if len(parts) == 2:
            namespace, name = parts
            readable = name.replace("-", " ").replace("_", " - ", 1)
            result.append(f"{readable} ({namespace})")
        else:
            result.append(raw.replace("-", " ").replace("_", " - ", 1))
    return ", ".join(result)


def reformulate_node(state: RAGState) -> RAGStateUpdate:
    """
    Reescribe la query usando el LLM para mejorar el recall.

    Usa ask_llm_internal() en lugar de ask_llm():
      - System prompt orientado a reformulación, no a respuesta RAG.
      - Sin chat_memory: operación interna del grafo, no turno del usuario.
      - Fallback a la pregunta original si el LLM falla.
    """
    original = state["question"]
    collection_names = _format_collection_names(state["collections"])

    reformulation_prompt = (
        f"Una búsqueda semántica sobre [{collection_names}] devolvió resultados "
        f"con baja relevancia para esta pregunta:\n\n"
        f'"{original}"\n\n'
        f"Reescribe la pregunta para maximizar la similitud semántica "
        f"con el vocabulario y los conceptos que probablemente usan esos documentos.\n\n"
        f"Estrategias:\n"
        f"- Sustituye términos abstractos o coloquiales por conceptos teóricos del dominio.\n"
        f"- Descompón la pregunta en sus conceptos nucleares y exprésalos explícitamente.\n"
        f"- Si la pregunta es general, hazla más específica al contenido probable del documento.\n"
        f"- Usa el vocabulario que usaría el autor, no el del usuario."
    )

    reformulated_question = ask_llm_internal(
        system_prompt=build_reformulation_system_prompt(),
        prompt=reformulation_prompt,
        provider=settings.provider_for(LLMRole.REFORMULATE),
    )

    logger.info(
        f"[graph] reformulate_node | original={original!r} | reformulada={reformulated_question!r}"
    )

    return {
        "question": reformulated_question if reformulated_question else original,
        "reformulated": True,
    }


# ======================================================
# GENERATE
# ======================================================


def generate_node(state: RAGState) -> RAGStateUpdate:
    """Construye el prompt con los chunks recuperados y llama al LLM."""
    results = state["results"]

    if not results:
        logger.warning("[graph] generate_node | sin resultados — respuesta vacía")
        return {"answer": "No se encontró contexto relevante para tu pregunta."}

    context_chunks = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in results
    ]

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=inject_attachments(state["question"], state["attachments"]),
        mode=state["mode"],
    )

    logger.info(
        f"[graph] generate_node | chunks={len(results)} | confidence={state['confidence']:.4f}"
    )

    # Guard preventivo de contexto (src/llm/context_guard.py). Vive acá
    # y no en el router de /query/agent porque el prompt con los chunks
    # recuperados recién existe en este punto -- antes de invocar el
    # grafo, el router todavía no sabe qué se va a recuperar.
    # ContextLimitExceeded se propaga tal cual hasta el router
    # (src/api/routers/chat.py), que la traduce a un 413.
    check_context_fit(
        system_prompt=build_system_prompt(),
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=settings.provider_for(LLMRole.GENERATE),
        max_tokens=state.get("max_tokens"),
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=settings.provider_for(LLMRole.GENERATE),
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
        max_turns=state.get("max_turns"),
    )

    return {"answer": answer}


# ======================================================
# REVIEW
# ======================================================


def review_node(state: RAGState) -> RAGStateUpdate:
    """
    Evalúa la respuesta de generate_node usando Gemini como reviewer.

    Qué verifica:
      - Anclaje: ¿las afirmaciones están en el contexto o son alucinaciones?
      - Citas: ¿la respuesta menciona las fuentes cuando hace afirmaciones concretas?

    Protocolo de respuesta esperado de Gemini: JSON puro.
      {"passed": true}
      {"passed": false, "feedback": "..."}

    Si Gemini devuelve algo que no es JSON válido (timeout, respuesta libre,
    error de red), se asume passed=True para no bloquear al usuario.
    Falla silenciosa explícita: mejor entregar la respuesta sin revisar que
    no entregar nada.

    Si review_attempts ya alcanzó MAX_REVIEW_ATTEMPTS, también se aprueba
    sin importar el feedback — evita loops infinitos.
    """
    results = state["results"]
    attempts = state.get("review_attempts", 0)

    if attempts >= MAX_REVIEW_ATTEMPTS:
        logger.warning(
            f"[graph] review_node | MAX_REVIEW_ATTEMPTS={MAX_REVIEW_ATTEMPTS} alcanzado "
            "→ aprobando respuesta sin revisar"
        )
        return {"review_passed": True, "review_feedback": ""}

    context_chunks = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in results
    ]

    review_prompt = build_review_prompt(
        context_chunks=context_chunks,
        question=state["question"],
        answer=state["answer"],
    )

    raw = ask_llm_internal(
        system_prompt=build_review_system_prompt(),
        prompt=review_prompt,
        provider=settings.provider_for(LLMRole.REVIEW),
    )

    temp: str = raw if raw is not None else '{"passed": true, "feedback": ""}'

    logger.info(f"[graph] review_node | respuesta raw del reviewer: {raw!r}")

    try:
        # Gemini a veces envuelve el JSON en ```json ... ``` aunque se le pide
        # que no lo haga. Limpieza defensiva antes de parsear.
        clean = temp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        passed: bool = bool(result.get("passed", True))
        feedback: str = str(result.get("feedback", ""))
    except (json.JSONDecodeError, AttributeError):
        logger.warning("[graph] review_node | no se pudo parsear JSON → aprobando por defecto")
        passed = True
        feedback = ""

    if passed:
        logger.info("[graph] review_node | ✓ respuesta aprobada")
    else:
        logger.info(f"[graph] review_node | ✗ respuesta rechazada | feedback={feedback!r}")

    return {
        "review_passed": passed,
        "review_feedback": feedback,
        "review_attempts": attempts + 1,
    }


def correct_node(state: RAGState) -> RAGStateUpdate:
    """
    Regenera la respuesta incorporando el feedback del reviewer.

    Es intencionalmente un nodo separado de generate_node (en vez de
    reutilizarlo con un flag) para que el grafo sea legible: generate
    produce, correct corrige. Cada nodo tiene una sola responsabilidad.
    """
    results = state["results"]

    context_chunks = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in results
    ]

    correction_prompt = build_correction_prompt(
        context_chunks=context_chunks,
        question=state["question"],
        previous_answer=state["answer"],
        feedback=state["review_feedback"],
        mode=state["mode"],
    )

    logger.info(
        f"[graph] correct_node | reintento={state['review_attempts']} "
        f"| feedback={state['review_feedback']!r}"
    )

    corrected = ask_llm(
        prompt=correction_prompt,
        chat_memory=state["chat_memory"],
        provider=settings.provider_for(LLMRole.GENERATE),
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
        max_turns=state.get("max_turns"),
    )

    return {"answer": corrected, "review_passed": False}


def route_after_review(state: RAGState) -> str:
    """
    - review_passed=True  → END
    - review_passed=False → correct (regenerar con feedback)
    """
    if state.get("review_passed", True):
        logger.info("[graph] route_after_review → END")
        return "end"
    logger.info("[graph] route_after_review → correct")
    return "correct"
