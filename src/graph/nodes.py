"""
Nodos del grafo RAG.

Cada función recibe el RAGState completo y devuelve un dict con solo
los campos que modifica — LangGraph hace el merge. Ningún nodo sabe
de qué nodo viene ni a cuál va: esa lógica vive en graph.py.

Nodos:
  retrieve_node    ejecuta search() con la question actual
  evaluate_node    decide si los resultados son suficientes
  reformulate_node reescribe la question para mejorar el recall
  generate_node    construye el prompt y llama al LLM

La decisión de routing (suficiente / insuficiente / ya reformulado)
la toma la función route_after_evaluate en graph.py, no aquí.
"""

from __future__ import annotations

from src.graph.state import RAGState
from src.llm.generate import ask_llm
from src.prompts.builder import build_prompt
from src.retrieval.search import search
from src.utils.logger import logger
from src.config.settings import settings

# Umbral mínimo de confianza para aceptar los resultados sin reformular.
# El score es similitud coseno (IndexFlatIP sobre vectores normalizados),
# por lo que el rango útil es [0.0, 1.0].


# ======================================================
# RETRIEVE
# ======================================================


def retrieve_node(state: RAGState) -> dict[str, object]:
    """
    Recupera chunks relevantes desde las colecciones FAISS activas.

    Delega completamente en search() del módulo retrieval — sin lógica
    duplicada. Actualiza results y confidence en el estado.
    """
    logger.info(
        f"[graph] retrieve_node | question={state['question']!r} "
        f"| mode={state['mode']}"
    )

    results, confidence = search(
        question=state["question"],
        mode=state["mode"],
        collections=state["collections"],
    )

    logger.info(
        f"[graph] retrieve_node | chunks={len(results)} "
        f"| confidence={confidence:.4f}"
    )

    return {"results": results, "confidence": confidence}


# ======================================================
# EVALUATE  (solo logging — el routing lo hace graph.py)
# ======================================================


def evaluate_node(state: RAGState) -> dict[str, object]:
    """
    Nodo de evaluación. No modifica el estado: solo registra la decisión
    que tomará route_after_evaluate para que quede visible en logs.

    Separar evaluación de routing es el patrón LangGraph recomendado:
    los nodos transforman datos, las funciones de routing deciden caminos.
    """
    confidence = state["confidence"]
    reformulated = state["reformulated"]

    if reformulated:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            "| ya reformulado → generando de todos modos"
        )
    elif confidence < settings.confidence_threshold:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            f"< umbral={settings.confidence_threshold} → reformulando"
        )
    else:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            "| suficiente → generando"
        )

    return {}


# ======================================================
# REFORMULATE
# ======================================================


def reformulate_node(state: RAGState) -> dict[str, object]:
    """
    Reescribe la pregunta para mejorar el recall en el siguiente retrieve.

    La reformulación es deliberadamente simple y local (sin llamada al LLM)
    para no añadir latencia. Agrega prefijos semánticos que amplían el
    espacio de búsqueda, igual que build_queries() en SOFT, pero enfocados
    en el caso de baja confianza.

    reformulated=True es el flag de guarda que impide un segundo ciclo.
    """
    original = state["question"]

    reformulated_question = f"Explica detalladamente y con contexto: {original}"

    logger.info(
        f"[graph] reformulate_node | original={original!r} "
        f"| reformulada={reformulated_question!r}"
    )

    return {
        "question": reformulated_question,
        "reformulated": True,
    }


# ======================================================
# GENERATE
# ======================================================


def generate_node(state: RAGState) -> dict[str, object]:
    """
    Construye el prompt con los chunks recuperados y llama al LLM.

    Usa build_prompt() y ask_llm() del pipeline existente — sin código
    duplicado. El historial de conversación viaja en chat_memory y llega
    a build_messages() dentro de ask_llm(), igual que en el pipeline lineal.
    """
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
        question=state["question"],
        mode=state["mode"],
    )

    logger.info(
        f"[graph] generate_node | chunks={len(results)} "
        f"| confidence={state['confidence']:.4f}"
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
    )

    return {"answer": answer}
