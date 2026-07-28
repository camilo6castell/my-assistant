"""
Shared state for the LangGraph graph.

RAGState is the only object that travels between nodes. LangGraph passes
it as an argument to each node and applies the returned dict as a partial
merge -- only the returned fields are updated, the rest is preserved.

Fields:
  question       original user question
  mode           ChatMode.SOFT | ChatMode.HARD
  collections    FAISS collections loaded in memory
  chat_memory    turn history (sliding window in generate.py)
  results        chunks retrieved by retrieve_node
  confidence     average score of the results
  reformulated   True if the query has been reformulated in this execution
                 (prevents infinite loops in the graph)
  answer         final LLM answer
  review_passed    True if review_node approved the answer (or was not reviewed)
  review_feedback  reason for rejection, used to regenerate with correction
  review_attempts  how many times the answer was regenerated after a reviewer
                   rejection (prevents infinite loops: see MAX_REVIEW_ATTEMPTS)
  max_tokens/think_mode/extra
                   per-request generation overrides (see
                   GenerationOptions in src/api/schemas/chat.py); None
                   in all = current behavior without changes. Temperature
                   does NOT live here -- it is a fixed property of each
                   model (src/config/models/<backend>.py), never a
                   per-request override.

                   There are no top_k_initial/top_k_final or max_turns here:
                   they stopped being per-request overrides (see
                   GenerationOptions in src/api/schemas/chat.py) --
                   retrieve_node/generate_node resolve them directly
                   against settings, without going through the graph
                   state. If they ever need to be re-exposed, they
                   existed here before and were deliberately removed --
                   see the history of this file before reinventing
                   the wheel.
  attachments      ad-hoc file attachments (see
                   src/context/attachments.py), as a list of
                   (filename, content). Only generate_node injects them
                   into the final prompt (see inject_attachments() in
                   src/prompts/builder.py) -- retrieve_node and
                   reformulate_node use `question` without attachments,
                   to avoid polluting the search embedding or
                   reformulation with file content.

NOTE: this module does NOT use `from __future__ import annotations`.
LangGraph calls get_type_hints(RAGState) at runtime to inspect the
state fields. With postponed annotations, all types become lazy strings
and get_type_hints() fails to resolve LoadedCollection if it is under
TYPE_CHECKING (NameError at runtime).

The solution is to import LoadedCollection directly --without a guard--
so it exists in the module namespace when LangGraph evaluates it.
"""

from typing import Any, TypedDict

from src.context.manager import LoadedCollection
from src.context.models import SearchResult
from src.domain.models import TurnMemory


class RAGState(TypedDict):
    question: str
    mode: str
    collections: list[LoadedCollection]
    chat_memory: list[TurnMemory]
    results: list[SearchResult]
    confidence: float
    reformulated: bool
    answer: str
    review_passed: bool
    review_feedback: str
    review_attempts: int
    # Per-request generation overrides (see GenerationOptions in
    # src/api/schemas/chat.py). None = use the settings/.env default.
    # Only generate_node/correct_node read them -- reformulate_node and
    # review_node always use the default temperature, they are single-pass
    # internal tasks, not the final response to the user.
    max_tokens: int | None
    think_mode: bool | None
    # Generic passthrough without validation (see GenerationOptions.extra) --
    # dict[str, Any] is the only deliberate exception to the strict typing
    # in the rest of the project: by definition it can contain any
    # provider-specific parameter that the backend does not model.
    extra: dict[str, Any] | None
    attachments: list[tuple[str, str]]


class RAGStateUpdate(TypedDict, total=False):
    """
    Partial update to RAGState.

    Each graph node (src/graph/nodes.py) returns only the subset of
    fields it modifies -- LangGraph applies the rest as a partial merge
    on the existing state (see module docstring). `total=False` models
    exactly that: all fields are optional in the returned dict, but each
    one that is present is typed the same as in RAGState, instead of
    losing precision with `dict[str, object]`.
    """

    question: str
    mode: str
    collections: list[LoadedCollection]
    chat_memory: list[TurnMemory]
    results: list[SearchResult]
    confidence: float
    reformulated: bool
    answer: str
    review_passed: bool
    review_feedback: str
    review_attempts: int
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None
    attachments: list[tuple[str, str]]
