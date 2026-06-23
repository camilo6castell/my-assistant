"""
Handlers informativos del menú principal.
El menú es deliberadamente simple: Chat es donde ocurre todo.
Las opciones 2-4 son solo consulta, sin modificar estado.
"""

from src.chat.session import ChatSession
from src.config.settings import BASE_VECTOR_PATH

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
    """Lista todas las colecciones disponibles en vector_stores."""

    contexts: list[str] = session.context_manager.list_all()

    print()

    if not contexts:
        print("  No contexts available.")
        print(f"  Directory: {BASE_VECTOR_PATH}\n")
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
    """Describe los modos de respuesta disponibles."""

    print(f"""
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
    """Información de la aplicación."""

    from src.config.settings import (
        BASE_TOP_K_FINAL,
        BASE_VECTOR_PATH,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATA_PATH,
        EMBED_MODEL,
        SOFT_TOP_K_FINAL,
        LLM_MODEL,
    )

    print(f"""
  ─────────────────────────────────────────────────────────────────
  My-Asisstant is a RAG SYSTEM made by Camilo6Castell
  ─────────────────────────────────────────────────────────────────
  Embedding model : {EMBED_MODEL}
  LLM model       : {LLM_MODEL}
  Chunk size      : {CHUNK_SIZE} chars  (overlap {CHUNK_OVERLAP})
  Top-K riguroso  : {BASE_TOP_K_FINAL} resultados finales
  Top-K interpret : {SOFT_TOP_K_FINAL} resultados finales
  Vector stores   : {BASE_VECTOR_PATH}
  Data path       : {DATA_PATH}
  ─────────────────────────────────────────────────────────────────

  Ingest de archivos:
    python -m src.ingest.ingest <namespace> <coleccion>

  Ingest de URL única:
    python -m src.ingest.web_ingest <namespace> <coleccion> <url>

  Ingest crawler:
    python -m src.ingest.web_crawler <coleccion> <url_base>

  ─────────────────────────────────────────────────────────────────
""")
