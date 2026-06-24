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
               ┌────────────────┼────────────────┐     │
               │ confianza ok   │ ya reformulado  │ baja confianza
               │ o reformulado  │                 │     │
               ▼                ▼                 ▼     │
         ┌──────────┐     ┌──────────┐    ┌─────────────┤
         │ generate │     │ generate │    │ reformulate │
         └────┬─────┘     └────┬─────┘    └─────────────┘
              │                │
              └──────┬─────────┘
                     ▼
                   [END]

El nodo evaluate no modifica el estado — solo registra la decisión.
El routing lo toma route_after_evaluate basándose en confidence y
el flag reformulated, que evita loops infinitos.

build_rag_graph() devuelve el grafo compilado listo para .invoke().
Se llama una vez y el resultado puede reutilizarse en múltiples turnos.
"""

from __future__ import annotations

from typing import Hashable

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
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

    - Si confidence >= umbral           → generate (resultados suficientes)
    - Si ya se reformuló antes          → generate (evita loop infinito)
    - Si confidence < umbral y es nuevo → reformulate (segundo intento)
    """
    if state["confidence"] >= settings.confidence_threshold or state["reformulated"]:
        logger.info(
            f"[graph] route → {_GENERATE} "
            f"(confidence={state['confidence']:.4f}, "
            f"reformulated={state['reformulated']})"
        )
        return _GENERATE

    logger.info(
        f"[graph] route → {_REFORMULATE} "
        f"(confidence={state['confidence']:.4f} < {settings.confidence_threshold})"
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
