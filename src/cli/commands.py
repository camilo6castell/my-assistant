"""
Informational handlers for the main menu.
The menu is deliberately simple: Chat is where everything happens.
Options 2-4 are read-only, with no state modification.
"""

from src.cli.session import ChatSession
from src.config.settings import settings
from src.nlp.llm.roles import LLMRole

# ======================================================
# MAIN MENU
# ======================================================


def show_main_menu() -> None:
    print("""
╔══════════════════════════╗
║    My assistant (RAG)    ║
╠══════════════════════════╣
║  1. Start chat           ║
║  2. Show contexts        ║
║  3. Show modes           ║
║  4. About                ║
║  5. Exit                 ║
╚══════════════════════════╝
""")


# ======================================================
# 2. SHOW CONTEXTS
# ======================================================


def show_contexts(session: ChatSession) -> None:
    """List all available collections in vector_stores."""

    contexts: list[str] = session.context_manager.list_all()

    print()

    if not contexts:
        print("  No contexts available.")
        print(f"  Directory: {settings.vector_store_path}\n")
        return

    print("  Contexts available:\n")

    current_ns: str | None = None

    for ctx in contexts:
        ns: str
        name: str
        ns, name = ctx.split("/", 1)

        if ns != current_ns:
            print(f"  [{ns}]")
            current_ns = ns

        print(f"    - {name}")

    print()


# ======================================================
# 3. SHOW MODES
# ======================================================


def show_modes() -> None:
    """Describe the available response modes."""

    print("""
  ─────────────────────────────────────────────────────────────────
  SOFT (default)
    - Can synthesize and connect concepts from different sources.
    - Generates more semantic search variations.
    - Ideal for analysis, comparisons, and conceptual synthesis.

  HARD
    - Responds using only the content of the loaded context.
    - Does not infer or connect ideas external to the text.
    - Ideal for precise and verifiable queries.

  **Change the mode within the chat with: '/mode'""
  ─────────────────────────────────────────────────────────────────
""")


# ======================================================
# 4. ABOUT
# ======================================================


def show_about() -> None:
    """Display application information."""

    generate_backend, generate_model = settings.role_spec(LLMRole.GENERATE)

    print(f"""
  ─────────────────────────────────────────────────────────────────
  My-Asisstant is a RAG SYSTEM made by camilo6castell
  ─────────────────────────────────────────────────────────────────
  Embedding model : {settings.embedding_backend}/{settings.embedding_model}
  LLM model       : {generate_backend}/{generate_model}
  Chunk size      : {settings.chunk_size} chars  (overlap {settings.chunk_overlap})
  Top-K riguroso  : {settings.hard_top_k_final} final results
  Top-K interpret : {settings.soft_top_k_final} final results
  Vector stores   : {settings.vector_store_path}
  Data path       : {settings.data_path}
  ─────────────────────────────────────────────────────────────────

  File ingest:
    python -m src.ingest.ingest <namespace> <collection>

  Single URL ingest:
    python -m src.ingest.web_ingest <namespace> <collection> <url>

  Crawler ingest:
    python -m src.ingest.web_crawler <collection> <base_url>

  ─────────────────────────────────────────────────────────────────
""")
