"""
Chat interface. Manages the command loop and questions.

Available commands:
  /context <tokens...>   load one or more contexts
  /remove  <tokens...>   unload one or more contexts
  /list                  show all available contexts
  /active                show active contexts
  /clear                 unload all contexts
  /reset                 clear conversation memory
  /mode                  toggle between STRICT and INTERPRETIVE
  /help                  show this help
  /exit                  return to main menu

Token syntax:
  sociologia             → all collections under sociologia/
  sociologia/debord      → exact collection
  sociologia react       → sociologia/* + react/*
  sociologia/debord react/hooks psicologia
                         → mix of exact and namespaces
"""

import warnings
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from src.cli.session import ChatSession
from src.context.manager import LoadedCollection
from src.context.models import SearchResult
from src.context.selector import match_contexts
from src.graph import RAGState, build_rag_graph
from src.nlp.llm.generate import ask_llm
from src.nlp.llm.roles import LLMRole
from src.prompts.builder import build_prompt
from src.retrieval.search import format_context_chunks, search

# Compiled graph, cached once per process (same pattern as api/app.py lifespan).
_compiled_graph = None


def _get_graph() -> CompiledStateGraph[RAGState]:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph


warnings.filterwarnings(
    "ignore",
    message="You're using a BertTokenizerFast tokenizer.*",
)

HELP: str = """

═════════════════════════════════════════════════════════════════════════
  Commands:

  /context <tokens>   load context(s)
  /remove  <tokens>   unload context(s)
  /list               view available contexts
  /active             view active contexts
  /clear              unload all contexts
  /reset              clear conversation memory
  /mode               toggle mode (STRICT / INTERPRETIVE)
  /agent              enable / disable agent mode (LangGraph)
  /help               show this help
  /exit               return to main menu

  Token syntax:
    sociologia                    → entire namespace
    sociologia/debord             → exact collection
    sociologia react              → two namespaces
    sociologia/debord react/hooks → mix of exact and namespaces

  Agent mode:
    When active, each question passes through a LangGraph graph that
    evaluates the confidence of the results. If low, it automatically
    reformulates the query and retries before generating the response.
═════════════════════════════════════════════════════════════════════════

"""


# ======================================================
# PRESENTATION HELPERS
# ======================================================


def _print_contexts_tree(contexts: list[str], label: str) -> None:
    if not contexts:
        return

    print(f"\n  {label}:\n")

    current_ns: str | None = None

    for ctx in sorted(contexts):
        ns: str
        name: str
        ns, name = ctx.split("/", 1)

        if ns != current_ns:
            print(f"  [{ns}]")
            current_ns = ns

        print(f"    - {name}")

    print()


def _print_available(session: ChatSession) -> None:
    contexts: list[str] = session.context_manager.list_all()
    print()
    if not contexts:
        print("  No contexts available.\n")
        return
    _print_contexts_tree(contexts, "Available")


def _print_active(session: ChatSession) -> None:
    active: list[str] = session.get_active_contexts()
    print()
    if not active:
        print("  No active contexts.\n")
        return
    _print_contexts_tree(active, "Active")


# ======================================================
# /context AND /remove HANDLER
# ======================================================


def _handle_context(session: ChatSession, raw_tokens: str) -> None:
    if not raw_tokens:
        print("\n  Usage: /context <tokens>  (e.g., /context sociologia react)\n")
        return

    available: list[str] = session.context_manager.list_all()
    targets: list[str] = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  No matches for: {raw_tokens!r}\n")
        return

    loaded: list[str] = []
    for ctx in targets:
        result: list[str] = session.load_context(ctx)
        loaded.extend(result)

    print()
    if not loaded:
        print(f"  Already active: {', '.join(targets)}")
    else:
        for ctx in loaded:
            print(f"  + {ctx}")
    print()


def _handle_remove(session: ChatSession, raw_tokens: str) -> None:
    if not raw_tokens:
        print("\n  Usage: /remove <tokens>  (e.g., /remove sociologia)\n")
        return

    available: list[str] = session.context_manager.list_all()
    targets: list[str] = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  No matches for: {raw_tokens!r}\n")
        return

    removed: list[str] = []
    for ctx in targets:
        result: list[str] = session.unload_context(ctx)
        removed.extend(result)

    print()
    if not removed:
        print("  None of those contexts were active.")
    else:
        for ctx in removed:
            print(f"  - {ctx}")
    print()


# ======================================================
# QUESTION HANDLER
# ======================================================


def _handle_question(session: ChatSession, question: str) -> None:
    collections: list[LoadedCollection] = session.context_manager.get_loaded_collections()

    if not collections:
        print("\n  Load a context first.  E.g., /context sociologia\n")
        return

    print("\n  Searching...\n")

    results: list[SearchResult]
    confidence: float
    results, confidence = search(
        question=question,
        mode=session.mode,
        collections=collections,
    )

    if not results:
        print("  No relevant context found.\n")
        return

    context_chunks: list[str] = format_context_chunks(results)

    # chat_memory is no longer passed to build_prompt -- history travels
    # as API messages in ask_llm → build_messages
    prompt: str = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
    )

    answer: str = ask_llm(
        prompt=prompt,
        chat_memory=session.chat_memory,
        provider=LLMRole.GENERATE.value,
    )

    print("  Answer:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]\n")

    session.add_to_memory(user=question, assistant=answer)


# ======================================================
# /agent HANDLER (LangGraph)
# ======================================================


def _handle_agent_question(session: ChatSession, question: str) -> None:
    """
    Question handler variant with adaptive retrieval via LangGraph.

    Differences from the linear pipeline (_handle_question):
      - If initial results confidence is low, the graph reformulates
        the query and retries once before generating.
      - The flow is a state graph (retrieve → evaluate → generate /
        reformulate → retrieve → generate) instead of a linear chain.
      - Shows whether reformulation was triggered so the user knows.

    The graph is compiled on first use and reused across subsequent
    calls within the same session (build_rag_graph is cached in the
    _graph attribute of ChatSession extended by start_chat).
    """
    collections: list[LoadedCollection] = session.context_manager.get_loaded_collections()

    if not collections:
        print("\n  Load a context first.  E.g., /context liberty\n")
        return

    print("\n  [agent] Executing RAG graph...\n")

    # The compiled graph is cached once per process (see _get_graph()).
    graph = _get_graph()

    initial_state: RAGState = {
        "question": question,
        "mode": session.mode,
        "collections": collections,
        "chat_memory": session.chat_memory,
        "results": [],
        "confidence": 0.0,
        "reformulated": False,
        "answer": "",
        "review_passed": False,
        "review_feedback": "",
        "review_attempts": 0,
        "max_tokens": None,
        "think_mode": None,
        "extra": None,
        # The CLI has no ad-hoc file attachments (that is a web API
        # feature -- see src/context/attachments.py and
        # src/api/routers/chat.py); empty list = generate_node injects
        # nothing extra into the prompt (see inject_attachments() in
        # src/prompts/builder.py, which returns `question` unmodified
        # if `attachments` is empty).
        "attachments": [],
    }

    # CompiledStateGraph.invoke() is typed in the library as
    # `dict[str, Any] | Any` (not as the generic StateT), so an explicit
    # cast is more honest here than blindly ignoring the error: it
    # documents exactly the point where LangGraph's precision ends and
    # ours begins (same pattern as src/api/app.py).
    final_state = cast(RAGState, graph.invoke(initial_state))

    answer = final_state["answer"]
    confidence = final_state["confidence"]
    reformulated = final_state["reformulated"]

    if not answer or answer == "No relevant context found for your question.":
        print("  No relevant context found.\n")
        return

    if reformulated:
        print("  [agent] Low confidence in initial search → query reformulated automatically.\n")

    print("  Answer:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]")
    if reformulated:
        print("  [reformulated: yes]")
    print()

    session.add_to_memory(user=question, assistant=answer)


def start_chat(session: ChatSession) -> None:
    print("\n═════════════════════════════════════════════════════════════════════════")
    print("  Welcome to the RAG-chat!\n")
    print("  Type '/help' to see the available commands.")
    print("═════════════════════════════════════════════════════════════════════════\n")

    while True:
        try:
            command: str = input(session.get_prompt_header()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue

        if command == "/exit":
            print()
            break

        if command in ("/help", "/?"):
            print(HELP)
            continue

        if command.startswith("/context"):
            _handle_context(session, command[len("/context") :].strip())
            continue

        if command.startswith("/remove"):
            _handle_remove(session, command[len("/remove") :].strip())
            continue

        if command == "/list":
            _print_available(session)
            continue

        if command == "/active":
            _print_active(session)
            continue

        if command == "/clear":
            session.clear_contexts()
            print("\n  Contexts cleared.\n")
            continue

        if command == "/reset":
            session.reset_memory()
            print("\n  Conversation memory cleared.\n")
            continue

        if command == "/mode":
            mode: str = session.toggle_mode()
            print(f"\n  Mode: {mode}\n")
            continue

        if command == "/agent":
            session.toggle_agent()
            status = "ON" if session.agent_active else "OFF"
            print(f"\n  Agent: {status}\n")
            continue

        if command.startswith("/"):
            print(f"\n  Unknown command: {command!r}  (type '/help')\n")
            continue

        if session.agent_active:
            _handle_agent_question(session, command)
        else:
            _handle_question(session, command)
