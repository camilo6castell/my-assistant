"""
src/cli/commands.py

Handlers informativos del menú principal.
El menú es deliberadamente simple: Chat es donde ocurre todo.
Las opciones 2-4 son solo consulta, sin modificar estado.
"""

from src.config.settings import BASE_VECTOR_PATH

# ======================================================
# 2. SHOW CONTEXTS
# ======================================================


def show_contexts(session):
    """Lista todas las colecciones disponibles en vector_stores."""

    contexts = session.context_manager.list_all()

    print()

    if not contexts:
        print("  No hay contextos disponibles.")
        print(f"  Directorio: {BASE_VECTOR_PATH}\n")
        return

    print("  Contextos disponibles:\n")

    current_ns = None

    for ctx in contexts:
        ns, name = ctx.split("/", 1)

        if ns != current_ns:
            print(f"  [{ns}]")
            current_ns = ns

        print(f"    - {name}")

    print()


# ======================================================
# 3. SHOW MODES
# ======================================================


def show_modes(session):
    """Describe los modos de respuesta disponibles."""

    current = session.mode

    print(f"""
  Modo actual: {current}

  RIGUROSO (por defecto)
    Responde usando únicamente el contenido del contexto cargado.
    No infiere ni conecta ideas externas al texto.
    Ideal para consultas precisas y verificables.

  INTERPRETATIVO
    Puede sintetizar y conectar conceptos entre fuentes.
    Genera más variantes de búsqueda semántica.
    Ideal para análisis, comparaciones y síntesis conceptual.

  Cambia el modo dentro del chat con: /mode
""")


# ======================================================
# 4. ABOUT
# ======================================================


def show_about(_session):
    """Información del sistema."""

    from src.config.settings import (
        EMBED_MODEL,
        LLM_MODEL,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        BASE_TOP_K_FINAL,
        INTERPRETATIVE_TOP_K_FINAL,
        BASE_VECTOR_PATH,
        DATA_PATH,
    )

    print(f"""
  RAG SYSTEM v2
  ─────────────────────────────────────
  Embedding model : {EMBED_MODEL}
  LLM model       : {LLM_MODEL}
  Chunk size      : {CHUNK_SIZE} chars  (overlap {CHUNK_OVERLAP})
  Top-K riguroso  : {BASE_TOP_K_FINAL} resultados finales
  Top-K interpret : {INTERPRETATIVE_TOP_K_FINAL} resultados finales
  Vector stores   : {BASE_VECTOR_PATH}
  Data path       : {DATA_PATH}

  Ingest de archivos:
    python -m src.ingest.ingest <namespace> <coleccion>

  Ingest de URL única:
    python -m src.ingest.web_ingest <namespace> <coleccion> <url>

  Ingest crawler:
    python -m src.ingest.web_crawler <coleccion> <url_base>
""")
