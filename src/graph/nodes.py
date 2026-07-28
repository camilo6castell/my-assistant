"""
RAG graph nodes.

Each function receives the full RAGState and returns a dict with only
the fields it modifies -- LangGraph handles the merge. No node knows
which node it comes from or which it goes to: that logic lives in graph.py.

Nodes:
  retrieve_node    executes search() with the current question
  evaluate_node    compares confidence with settings.confidence_limit
  reformulate_node calls the LLM to rewrite the query
  generate_node    builds the prompt and calls the LLM

--- Metric: average confidence with calibrated threshold ---

Confidence is the average of cosine scores from the top-K returned by
search(). bge-small-en-v1.5 produces scores in an empirically observed
range with these documents:

  on-topic:  >= 0.80   (query semantically aligned with the doc)
  off-topic: ~0.75     (query with no real match, e.g. cats vs Freud)

That's why confidence_limit=0.80 in settings. If your model or corpus
produces different ranges, adjust CONFIDENCE_LIMIT in .env.
"""

from __future__ import annotations

import json

from src.config.settings import settings
from src.context.manager import LoadedCollection
from src.graph.state import RAGState, RAGStateUpdate
from src.nlp.llm.context_guard import check_context_fit
from src.nlp.llm.generate import ask_llm, ask_llm_internal
from src.nlp.llm.roles import LLMRole
from src.prompts.builder import (
    build_correction_prompt,
    build_prompt,
    build_reformulation_system_prompt,
    build_review_prompt,
    build_review_system_prompt,
    build_system_prompt,
    inject_attachments,
)
from src.retrieval.search import format_context_chunks, search
from src.utils.logger import logger

# Maximum cycles generate → review → generate before returning the answer
# as-is. A value of 1 = one retry (2 generate calls total).
# Increasing this consumes more tokens/time: on a local runner it's a real tradeoff.
MAX_REVIEW_ATTEMPTS = settings.max_review_attempts

# ======================================================
# RETRIEVE
# ======================================================


def retrieve_node(state: RAGState) -> RAGStateUpdate:
    """Retrieves relevant chunks from the active FAISS collections."""
    logger.info(f"[graph] retrieve_node | question={state['question']!r} | mode={state['mode']}")

    # top_k_initial/top_k_final are no longer per-request overrides (see
    # GenerationOptions in src/api/schemas/chat.py) -- search() resolves
    # both internally against settings.soft_top_k_*/hard_top_k_* depending
    # on state["mode"].
    results, confidence = search(
        question=state["question"],
        mode=state["mode"],
        collections=state["collections"],
    )

    logger.info(f"[graph] retrieve_node | chunks={len(results)} | confidence={confidence:.4f}")

    return {"results": results, "confidence": confidence}


# ======================================================
# EVALUATE
# ======================================================


def evaluate_node(state: RAGState) -> RAGStateUpdate:
    """
    Evaluation node. Does not modify state: only logs the decision
    that route_after_evaluate will take.
    """
    confidence = state["confidence"]
    reformulated = state["reformulated"]
    limit = settings.confidence_limit

    if reformulated:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f}"
            f" | already reformulated -> generating"
        )
    elif confidence < limit:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} < limit={limit} → reformulating"
        )
    else:
        logger.info(
            f"[graph] evaluate_node | confidence={confidence:.4f} >= limit={limit} → generating"
        )

    return {}


# ======================================================
# REFORMULATE
# ======================================================


def _format_collection_names(collections: list[LoadedCollection]) -> str:
    """
    Converts internal paths into human-readable names for the LLM.

    'sociologia/George-Orwell_1984' → 'George Orwell - 1984 (sociologia)'
    """
    result = []
    for c in collections:
        raw: str = c["collection_name"]  # e.g.: sociologia/George-Orwell_1984
        parts = raw.split("/", 1)
        if len(parts) == 2:
            namespace, name = parts
            readable = name.replace("-", " ").replace("_", " - ", 1)
            result.append(f"{readable} ({namespace})")
        else:
            result.append(raw.replace("-", " ").replace("_", " - ", 1))
    return ", ".join(result)


def reformulate_node(state: RAGState) -> RAGStateUpdate:
    """
    Rewrites the query using the LLM to improve recall.

    Uses ask_llm_internal() instead of ask_llm():
      - System prompt oriented toward reformulation, not RAG response.
      - No chat_memory: internal graph operation, not a user turn.
      - Falls back to the original question if the LLM fails.
    """
    original = state["question"]
    collection_names = _format_collection_names(state["collections"])

    reformulation_prompt = (
        f"A semantic search over [{collection_names}] returned results "
        f"with low relevance for this question:\n\n"
        f'"{original}"\n\n'
        f"Rewrite the question to maximize semantic similarity "
        f"with the vocabulary and concepts those documents likely use.\n\n"
        f"Strategies:\n"
        f"- Replace abstract or colloquial terms with domain-specific theoretical concepts.\n"
        f"- Break down the question into its core concepts and express them explicitly.\n"
        f"- If the question is general, make it more specific to the probable document content.\n"
        f"- Use the vocabulary the author would use, not the user's."
    )

    reformulated_question = ask_llm_internal(
        system_prompt=build_reformulation_system_prompt(),
        prompt=reformulation_prompt,
        provider=LLMRole.REFORMULATE.value,
    )

    logger.info(
        f"[graph] reformulate_node | original={original!r} | reformulated={reformulated_question!r}"
    )

    return {
        "question": reformulated_question if reformulated_question else original,
        "reformulated": True,
    }


# ======================================================
# GENERATE
# ======================================================


def generate_node(state: RAGState) -> RAGStateUpdate:
    """Builds the prompt with the retrieved chunks and calls the LLM."""
    results = state["results"]

    if not results:
        logger.warning("[graph] generate_node | no results — empty answer")
        return {"answer": "No relevant context was found for your question."}

    context_chunks = format_context_chunks(results)

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=inject_attachments(state["question"], state["attachments"]),
        mode=state["mode"],
    )

    logger.info(
        f"[graph] generate_node | chunks={len(results)} | confidence={state['confidence']:.4f}"
    )

    # Preventive context guard (src/llm/context_guard.py). Lives here
    # and not in the /query/agent router because the prompt with the
    # retrieved chunks only exists at this point -- before invoking the
    # graph, the router doesn't know what will be retrieved.
    # ContextLimitExceeded propagates as-is to the router
    # (src/api/routers/chat.py), which translates it to a 413.
    check_context_fit(
        system_prompt=build_system_prompt(),
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
    )

    return {"answer": answer}


# ======================================================
# REVIEW
# ======================================================


def review_node(state: RAGState) -> RAGStateUpdate:
    """
    Evaluates the generate_node answer using Gemini as a reviewer.

    What it checks:
      - Grounding: are the claims in the context or are they hallucinations?
      - Citations: does the answer cite sources when making specific claims?

    Expected Gemini response protocol: pure JSON.
      {"passed": true}
      {"passed": false, "feedback": "..."}

    If Gemini returns something that isn't valid JSON (timeout, free-form
    response, network error), passed=True is assumed to avoid blocking
    the user. Explicit silent failure: better to deliver the answer
    unreviewed than to deliver nothing.

    If review_attempts already reached MAX_REVIEW_ATTEMPTS, the answer is
    also approved regardless of feedback -- prevents infinite loops.
    """
    results = state["results"]
    attempts = state.get("review_attempts", 0)

    if attempts >= MAX_REVIEW_ATTEMPTS:
        logger.warning(
            f"[graph] review_node | MAX_REVIEW_ATTEMPTS={MAX_REVIEW_ATTEMPTS} reached "
            "→ approving answer without reviewing"
        )
        return {"review_passed": True, "review_feedback": ""}

    context_chunks = format_context_chunks(results)

    review_prompt = build_review_prompt(
        context_chunks=context_chunks,
        question=state["question"],
        answer=state["answer"],
    )

    raw = ask_llm_internal(
        system_prompt=build_review_system_prompt(),
        prompt=review_prompt,
        provider=LLMRole.REVIEW.value,
    )

    temp: str = raw if raw is not None else '{"passed": true, "feedback": ""}'

    logger.info(f"[graph] review_node | raw reviewer response: {raw!r}")

    try:
        # Gemini sometimes wraps JSON in ```json ... ``` even when asked
        # not to. Defensive cleanup before parsing.
        clean = temp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        passed: bool = bool(result.get("passed", True))
        feedback: str = str(result.get("feedback", ""))
    except (json.JSONDecodeError, AttributeError):
        logger.warning("[graph] review_node | could not parse JSON → approving by default")
        passed = True
        feedback = ""

    if passed:
        logger.info("[graph] review_node | ✓ answer approved")
    else:
        logger.info(f"[graph] review_node | ✗ answer rejected | feedback={feedback!r}")

    return {
        "review_passed": passed,
        "review_feedback": feedback,
        "review_attempts": attempts + 1,
    }


def correct_node(state: RAGState) -> RAGStateUpdate:
    """
    Regenerates the answer incorporating the reviewer's feedback.

    This is intentionally a separate node from generate_node (instead of
    reusing it with a flag) so the graph is readable: generate produces,
    correct fixes. Each node has a single responsibility.
    """
    results = state["results"]

    context_chunks = format_context_chunks(results)

    correction_prompt = build_correction_prompt(
        context_chunks=context_chunks,
        question=state["question"],
        previous_answer=state["answer"],
        feedback=state["review_feedback"],
        mode=state["mode"],
    )

    logger.info(
        f"[graph] correct_node | attempt={state['review_attempts']} "
        f"| feedback={state['review_feedback']!r}"
    )

    corrected = ask_llm(
        prompt=correction_prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
    )

    return {"answer": corrected, "review_passed": False}


def route_after_review(state: RAGState) -> str:
    """
    - review_passed=True  → END
    - review_passed=False → correct (regenerate with feedback)
    """
    if state.get("review_passed", True):
        logger.info("[graph] route_after_review → END")
        return "end"
    logger.info("[graph] route_after_review → correct")
    return "correct"
