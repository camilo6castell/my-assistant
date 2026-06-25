"""
Nodos del grafo RAG.

Cada función recibe el RAGState completo y devuelve un dict con solo
los campos que modifica — LangGraph hace el merge. Ningún nodo sabe
de qué nodo viene ni a cuál va: esa lógica vive en graph.py.

Nodos:
  retrieve_node    ejecuta search() con la question actual
  evaluate_node    calcula relevance_gap y decide la ruta
  reformulate_node reescribe la question para mejorar el recall
  generate_node    construye el prompt y llama al LLM

--- Por qué se usa relevance_gap en lugar de un umbral absoluto ---

bge-small-en-v1.5 con IndexFlatIP sobre vectores L2-normalizados produce
similitud coseno. En la práctica, el piso del modelo para cualquier par
de textos en español es ~0.62-0.65, incluso para queries completamente
off-topic. Un umbral absoluto (ej. 0.55) nunca se activa porque todos
los scores caen por encima de ese piso.

La solución es medir dispersión relativa:

    relevance_gap = score_max - score_min

Cuando el retriever encuentra algo realmente relevante, el top-1 tiene
un score notablemente más alto que el top-K → gap grande → relevante.
Cuando devuelve resultados genéricos sin match real, todos los scores
son similares entre sí → gap pequeño → baja relevancia real.

Ejemplos observados con los logs del usuario:
  on-topic real (fascismo vs Freud):   0.7700 avg, gap estimado ~0.06+
  off-topic (gato vs Freud):           0.7172 avg, gap estimado ~0.02-0.03

GAP_THRESHOLD = 0.05 detecta la diferencia entre "el modelo encontró
algo específico" y "el modelo devolvió lo menos malo disponible".
"""

from __future__ import annotations

from src.graph.state import RAGState
from src.llm.generate import ask_llm
from src.prompts.builder import build_prompt
from src.retrieval.search import search
from src.utils.logger import logger
from src.config.settings import settings


def _relevance_gap(state: RAGState) -> float:
    """
    Diferencia entre el mejor y el peor score de los resultados actuales.

    Un gap grande indica que el top-1 destaca sobre el resto → match real.
    Un gap pequeño indica que todos los resultados son igualmente genéricos.
    Retorna 0.0 si no hay resultados.
    """
    results = state["results"]
    if not results:
        return 0.0
    scores = [r.score for r in results]
    return max(scores) - min(scores)


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
    gap = _relevance_gap(state)
    reformulated = state["reformulated"]

    if reformulated:
        logger.info(
            f"[graph] evaluate_node | gap={gap:.4f} "
            "| ya reformulado → generando de todos modos"
        )
    elif gap < settings.gap_threshold:
        logger.info(
            f"[graph] evaluate_node | gap={gap:.4f} "
            f"< umbral={settings.gap_threshold} → reformulando "
            "(resultados genéricos, sin match específico)"
        )
    else:
        logger.info(
            f"[graph] evaluate_node | gap={gap:.4f} "
            "| match específico detectado → generando"
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
    en el caso de baja relevancia detectada por el gap.

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

    gap = _relevance_gap(state)
    logger.info(
        f"[graph] generate_node | chunks={len(results)} "
        f"| confidence={state['confidence']:.4f} | gap={gap:.4f}"
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
    )

    return {"answer": answer}
