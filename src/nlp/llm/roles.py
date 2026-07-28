"""
LLM roles: every pipeline point that calls a model is identified by one
of these roles. The backend+model serving each role is configured
independently (see Settings.role_spec in src/config/settings.py) --
previously reformulate, review, and web supplement shared the same
"reformulate_provider" for no reason other than they were never
separated; now each role has its own env var, so you can for example
run the final answer on a powerful local model and short support tasks
on a fast cloud model, without coupling them together.

Separate module, importing nothing from the rest of the project (not
even settings), specifically so it can be imported from both
src/config/settings.py and the call sites (src/llm/generate.py,
src/graph/nodes.py, src/api/routers/chat.py, src/chat/interface.py)
without circular import risk -- settings.py is one of the earliest
imports in the project.
"""

from enum import StrEnum


class LLMRole(StrEnum):
    """A pipeline point that needs an LLM."""

    # Generates the final RAG answer -- generate_node/correct_node in
    # the graph, the linear /query pipeline, and direct CLI chat.
    GENERATE = "generate"
    # Rewrites the user's question to improve retrieval when initial
    # confidence is low -- reformulate_node.
    REFORMULATE = "reformulate"
    # Evaluates whether the generated answer passes quality checks
    # (grounding, source attribution) -- review_node.
    REVIEW = "review"
    # Decides whether a web search adds genuinely new information to
    # an answer already generated from local context -- see
    # _supplement_with_web in src/api/routers/chat.py.
    WEB_SUPPLEMENT = "web_supplement"
