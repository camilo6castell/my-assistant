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

from src.config.settings import settings
from src.context.models import SearchResult
from src.graph.state import RAGState
from src.llm.generate import ask_llm, ask_llm_internal
from src.prompts.builder import build_prompt
from src.retrieval.search import search
from src.utils.logger import logger
from src.context.manager import LoadedCollection

# ======================================================
# RETRIEVE
# ======================================================


def retrieve_node(state: RAGState) -> dict[str, object]:
    """Recupera chunks relevantes desde las colecciones FAISS activas."""
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
# EVALUATE
# ======================================================


def evaluate_node(state: RAGState) -> dict[str, object]:
    """
    Nodo de evaluación. No modifica el estado: solo registra la decisión
    que tomará route_after_evaluate.
    """
    confidence = state["confidence"]
    reformulated = state["reformulated"]
    limit = settings.confidence_limit

    if reformulated:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            "| ya reformulado → generando"
        )
    elif confidence < limit:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            f"< limit={limit} → reformulando"
        )
    else:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} "
            f">= limit={limit} → generando"
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


def reformulate_node(state: RAGState) -> dict[str, object]:
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

    reformulated_question = ask_llm_internal(prompt=reformulation_prompt)

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
