"""
RAG graph with adaptive retrieval using LangGraph.

Flow:
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
          or already reformulated           low        │
               │                              │        │
               ▼                              ▼        │
         ┌──────────┐                  ┌────────────┐  │
         │ generate │                  │ reformulate│──┘
         └────┬─────┘                  └────────────┘
              │
              ▼
         ┌──────────┐
         │  review  │  ← Gemini evaluates grounding + citations
         └────┬─────┘
              │
       ┌──────┴──────┐
    passed        rejected
       │              │
       ▼              ▼
      END         ┌─────────┐
                  │ correct │  ← local regenerates with feedback
                  └────┬────┘
                       │
                       ▼
                    review  (bounded loop by MAX_REVIEW_ATTEMPTS)

Routing evaluate: confidence >= settings.confidence_limit → generate
                  confidence <  settings.confidence_limit → reformulate (once)
Routing review:   passed=True  → END
                  passed=False → correct → review (max MAX_REVIEW_ATTEMPTS times)
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

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
    - If already reformulated                        → generate (avoids loop)
    - If confidence >= settings.confidence_limit → generate
    - If confidence <  settings.confidence_limit → reformulate
    """
    confidence = state["confidence"]
    limit = settings.confidence_limit

    if state["reformulated"] or confidence >= limit:
        logger.info(
            f"[graph] route → {_GENERATE} "
            f"(confidence={confidence:.4f}, reformulated={state['reformulated']})"
        )
        return _GENERATE

    logger.info(f"[graph] route → {_REFORMULATE} (confidence={confidence:.4f} < limit={limit})")
    return _REFORMULATE


def build_rag_graph() -> CompiledStateGraph[RAGState]:
    """Builds and compiles the RAG graph."""
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

    # StateGraph.compile() doesn't resolve its generic TypeVar the same way
    # in all langgraph versions (in some a free StateT remains in the
    # inferred type for Input/Output instead of being bound to RAGState).
    # A direct cast to CompiledStateGraph[RAGState] would be "redundant"
    # in one version and "incompatible" in another -- cast(Any, ...) is
    # never redundant (Any never matches whatever mypy infers) and the
    # variable annotation imposes the real type outward, so this remains
    # correct regardless of the installed version.
    compiled: CompiledStateGraph[RAGState] = cast(Any, graph.compile())
    logger.info("[graph] RAG graph compiled successfully")
    return compiled
