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
               ┌────────────────┼──────────────────┐   │
               │ confidence ok  │ ya reformulado   │ confidence baja
               │ o reformulado  │                  │   │
               ▼                ▼                  ▼   │
         ┌──────────┐     ┌──────────┐    ┌────────────┤
         │ generate │     │ generate │    │ reformulate│
         └────┬─────┘     └────┬─────┘    └────────────┘
              │                │
              └──────┬─────────┘
                     ▼
                   [END]

Routing: confidence >= settings.confidence_limit → generate
         confidence <  settings.confidence_limit → reformulate (una vez)
Configurable en .env: CONFIDENCE_LIMIT=0.80
"""

from __future__ import annotations

from typing import Hashable

from langgraph.graph import END, StateGraph

from src.config.settings import settings
from src.graph.nodes import (
    evaluate_node,
    generate_node,
    reformulate_node,
    retrieve_node,
)
from src.graph.state import RAGState
from src.utils.logger import logger

_RETRIEVE = "retrieve"
_EVALUATE = "evaluate"
_REFORMULATE = "reformulate"
_GENERATE = "generate"


def route_after_evaluate(state: RAGState) -> Hashable:
    """
    - Si ya reformuló                          → generate (evita loop)
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


def build_rag_graph() -> object:
    """Construye y compila el grafo RAG."""
    graph: StateGraph[RAGState] = StateGraph(RAGState)

    graph.add_node(_RETRIEVE, retrieve_node)
    graph.add_node(_EVALUATE, evaluate_node)
    graph.add_node(_REFORMULATE, reformulate_node)
    graph.add_node(_GENERATE, generate_node)

    graph.set_entry_point(_RETRIEVE)
    graph.add_edge(_RETRIEVE, _EVALUATE)
    graph.add_edge(_REFORMULATE, _RETRIEVE)
    graph.add_edge(_GENERATE, END)

    graph.add_conditional_edges(
        _EVALUATE,
        route_after_evaluate,
        {_REFORMULATE: _REFORMULATE, _GENERATE: _GENERATE},
    )

    compiled = graph.compile()
    logger.info("[graph] Grafo RAG compilado correctamente")
    return compiled
