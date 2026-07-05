"""
Grafo RAG con adaptive retrieval usando LangGraph.

Flujo:
                         ┌─────────────┐
                         │   retrieve  │◄──────────────┐
                         └──────┬──────┘               │
                                │                      │
                         ┌──────▼──────┐               │
                         │  evaluate   │               │
                         └──────┬──────┘               │
                                │                      │
               ┌────────────────┴─────────────┐        │
          confidence ok                  confidence    │
          o ya reformulado                 baja        │
               │                              │        │
               ▼                              ▼        │
         ┌──────────┐                  ┌────────────┐  │
         │ generate │                  │ reformulate│──┘
         └────┬─────┘                  └────────────┘
              │
              ▼
         ┌──────────┐
         │  review  │  ← Gemini evalúa anclaje + citas
         └────┬─────┘
              │
       ┌──────┴──────┐
    passed        rechazado
       │              │
       ▼              ▼
      END         ┌─────────┐
                  │ correct │  ← local regenera con feedback
                  └────┬────┘
                       │
                       ▼
                    review  (loop acotado por MAX_REVIEW_ATTEMPTS)

Routing evaluate: confidence >= settings.confidence_limit → generate
                  confidence <  settings.confidence_limit → reformulate (una vez)
Routing review:   passed=True  → END
                  passed=False → correct → review (máx MAX_REVIEW_ATTEMPTS veces)
"""

from __future__ import annotations

from typing import Any, Hashable, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config.settings import settings
from src.graph.nodes import (
    correct_node,
    evaluate_node,
    generate_node,
    reformulate_node,
    retrieve_node,
    review_node,
    route_after_review,
)
from src.graph.state import RAGState
from src.utils.logger import logger

_RETRIEVE = "retrieve"
_EVALUATE = "evaluate"
_REFORMULATE = "reformulate"
_GENERATE = "generate"
_REVIEW = "review"
_CORRECT = "correct"


def route_after_evaluate(state: RAGState) -> Hashable:
    """
    - Si ya reformuló                            → generate (evita loop)
    - Si confidence >= settings.confidence_limit → generate
    - Si confidence <  settings.confidence_limit → reformulate
    """
    confidence = state["confidence"]
    limit = settings.confidence_limit

    if state["reformulated"] or confidence >= limit:
        logger.info(
            f"[graph] route → {_GENERATE} "
            f"(confidence={confidence:.4f}, reformulated={state['reformulated']})"
        )
        return _GENERATE

    logger.info(
        f"[graph] route → {_REFORMULATE} "
        f"(confidence={confidence:.4f} < limit={limit})"
    )
    return _REFORMULATE


def build_rag_graph() -> CompiledStateGraph[RAGState]:
    """Construye y compila el grafo RAG."""
    graph: StateGraph[RAGState] = StateGraph(RAGState)

    graph.add_node(_RETRIEVE, retrieve_node)
    graph.add_node(_EVALUATE, evaluate_node)
    graph.add_node(_REFORMULATE, reformulate_node)
    graph.add_node(_GENERATE, generate_node)
    graph.add_node(_REVIEW, review_node)
    graph.add_node(_CORRECT, correct_node)

    graph.set_entry_point(_RETRIEVE)
    graph.add_edge(_RETRIEVE, _EVALUATE)
    graph.add_edge(_REFORMULATE, _RETRIEVE)
    graph.add_edge(_GENERATE, _REVIEW)
    graph.add_edge(_CORRECT, _REVIEW)

    graph.add_conditional_edges(
        _EVALUATE,
        route_after_evaluate,
        {_REFORMULATE: _REFORMULATE, _GENERATE: _GENERATE},
    )

    graph.add_conditional_edges(
        _REVIEW,
        route_after_review,
        {"end": END, "correct": _CORRECT},
    )

    # StateGraph.compile() no resuelve su TypeVar genérico igual en todas
    # las versiones de langgraph (en algunas queda un StateT libre en el
    # tipo inferido para Input/Output en vez de bindearlo a RAGState).
    # Un cast directo a CompiledStateGraph[RAGState] sería "redundante"
    # en una versión y "incompatible" en otra -- cast(Any, ...) nunca es
    # redundante (Any nunca coincide con lo que mypy infiera) y la
    # anotación de la variable impone el tipo real hacia afuera, así que
    # esto se mantiene correcto sin importar la versión instalada.
    compiled: CompiledStateGraph[RAGState] = cast(Any, graph.compile())
    logger.info("[graph] Grafo RAG compilado correctamente")
    return compiled
