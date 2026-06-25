"""
Grafo RAG con adaptive retrieval usando LangGraph.

Flujo:
                         ┌─────────────┐
                         │   retrieve  │◄───────────────┐
                         └──────┬──────┘                │
                                │                       │
                         ┌──────▼──────┐                │
                         │  evaluate   │                │
                         └──────┬──────┘                │
                                │                       │
               ┌────────────────┼───────────────────┐   │
               │ gap ok o       │ ya reformulado    │ gap bajo
               │ reformulado    │                   │   │
               ▼                ▼                   ▼   │
         ┌──────────┐      ┌──────────┐    ┌────────────┤
         │ generate │      │ generate │    │ reformulate│
         └────┬─────┘      └────┬─────┘    └────────────┘
              │                 │
              └───────┬─────────┘
                      ▼
                    [END]

El nodo evaluate no modifica el estado — solo registra la decisión.
El routing lo toma route_after_evaluate basándose en relevance_gap
(diferencia max-min de scores) y el flag reformulated.

Ver nodes.py para la explicación de por qué se usa gap en vez de
un umbral absoluto sobre confidence.
"""

from __future__ import annotations

from typing import Hashable

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    _relevance_gap,
    evaluate_node,
    generate_node,
    reformulate_node,
    retrieve_node,
)
from src.graph.state import RAGState
from src.utils.logger import logger
from src.config.settings import settings

# Nombres de nodos — constantes para evitar typos en add_edge / routing
_RETRIEVE = "retrieve"
_EVALUATE = "evaluate"
_REFORMULATE = "reformulate"
_GENERATE = "generate"


# ======================================================
# ROUTING
# ======================================================


def route_after_evaluate(state: RAGState) -> Hashable:
    """
    Decide el siguiente nodo tras evaluate:

    - Si ya se reformuló antes          → generate (evita loop infinito)
    - Si relevance_gap >= GAP_THRESHOLD → generate (match específico)
    - Si relevance_gap <  GAP_THRESHOLD → reformulate (resultados genéricos)
    """
    gap = _relevance_gap(state)

    if state["reformulated"] or gap >= settings.gap_threshold:
        logger.info(
            f"[graph] route → {_GENERATE} "
            f"(gap={gap:.4f}, reformulated={state['reformulated']})"
        )
        return _GENERATE

    logger.info(
        f"[graph] route → {_REFORMULATE} "
        f"(gap={gap:.4f} < umbral={settings.gap_threshold} → resultados genéricos)"
    )
    return _REFORMULATE


# ======================================================
# BUILDER
# ======================================================


def build_rag_graph() -> object:
    """
    Construye y compila el grafo RAG.

    Retorna un CompiledGraph con método .invoke(state) → state.
    El tipo de retorno es `object` para evitar importar el tipo privado
    de LangGraph en los módulos que usan el grafo.
    """
    graph: StateGraph[RAGState] = StateGraph(RAGState)

    # Nodos
    graph.add_node(_RETRIEVE, retrieve_node)
    graph.add_node(_EVALUATE, evaluate_node)
    graph.add_node(_REFORMULATE, reformulate_node)
    graph.add_node(_GENERATE, generate_node)

    # Edges deterministas
    graph.set_entry_point(_RETRIEVE)
    graph.add_edge(_RETRIEVE, _EVALUATE)
    graph.add_edge(_REFORMULATE, _RETRIEVE)
    graph.add_edge(_GENERATE, END)

    # Edge condicional: evaluate → reformulate | generate
    graph.add_conditional_edges(
        _EVALUATE,
        route_after_evaluate,
        {
            _REFORMULATE: _REFORMULATE,
            _GENERATE: _GENERATE,
        },
    )

    compiled = graph.compile()
    logger.info("[graph] Grafo RAG compilado correctamente")
    return compiled
