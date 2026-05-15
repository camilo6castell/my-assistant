"""
src/chat/interface.py

Interfaz de chat. Gestiona el loop de comandos y preguntas.

Comandos disponibles:
  /context <tokens...>   carga uno o más contextos
  /remove  <tokens...>   descarga uno o más contextos
  /list                  muestra todos los contextos disponibles
  /active                muestra los contextos activos
  /clear                 descarga todos los contextos
  /reset                 limpia la memoria de conversación
  /mode                  alterna entre RIGUROSO e INTERPRETATIVO
  /help                  muestra esta ayuda
  /exit                  vuelve al menú principal

Sintaxis de <tokens>:
  sociologia             → todas las colecciones bajo sociologia/
  sociologia/debord      → colección exacta
  sociologia react       → sociologia/* + react/*
  sociologia/debord react/hooks psicologia
                         → mezcla de exactos y namespaces
"""

import warnings

from src.chat.session import ChatSession
from src.context.selector import match_contexts
from src.llm.generate import ask_llm
from src.retrieval.search import search
from src.prompts.builder import build_prompt

warnings.filterwarnings(
    "ignore",
    message="You're using a BertTokenizerFast tokenizer.*",
)

HELP = """
  Comandos:

  /context <tokens>   cargar contexto(s)
  /remove  <tokens>   descargar contexto(s)
  /list               ver contextos disponibles
  /active             ver contextos activos
  /clear              descargar todos los contextos
  /reset              limpiar memoria de conversación
  /mode               alternar modo (RIGUROSO / INTERPRETATIVO)
  /help               mostrar esta ayuda
  /exit               volver al menú principal

  Sintaxis de tokens:
    sociologia                    → todo el namespace
    sociologia/debord             → colección exacta
    sociologia react              → dos namespaces
    sociologia/debord react/hooks → mezcla exactos y namespaces
"""


# ======================================================
# HELPERS DE PRESENTACIÓN
# ======================================================


def _print_contexts_tree(contexts: list[str], label: str):
    """Imprime una lista de colecciones agrupada por namespace."""

    if not contexts:
        return

    print(f"\n  {label}:\n")

    current_ns = None

    for ctx in sorted(contexts):
        ns, name = ctx.split("/", 1)

        if ns != current_ns:
            print(f"  [{ns}]")
            current_ns = ns

        print(f"    - {name}")

    print()


def _print_available(session):
    contexts = session.context_manager.list_all()

    print()

    if not contexts:
        print("  No hay contextos disponibles.\n")
        return

    _print_contexts_tree(contexts, "Disponibles")


def _print_active(session):
    active = session.get_active_contexts()

    print()

    if not active:
        print("  No hay contextos activos.\n")
        return

    _print_contexts_tree(active, "Activos")


# ======================================================
# HANDLER DE /context Y /remove
# ======================================================


def _handle_context(session, raw_tokens: str):
    """Carga uno o más contextos a partir de tokens separados por espacio."""

    if not raw_tokens:
        print("\n  Uso: /context <tokens>  (ej: /context sociologia react)\n")
        return

    available = session.context_manager.list_all()
    targets = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  Sin coincidencias para: {raw_tokens!r}\n")
        return

    loaded = []

    for ctx in targets:
        result = session.load_context(ctx)
        loaded.extend(result)

    print()

    if not loaded:
        already = ", ".join(targets)
        print(f"  Ya activos: {already}")
    else:
        for ctx in loaded:
            print(f"  + {ctx}")

    print()


def _handle_remove(session, raw_tokens: str):
    """Descarga uno o más contextos a partir de tokens separados por espacio."""

    if not raw_tokens:
        print("\n  Uso: /remove <tokens>  (ej: /remove sociologia)\n")
        return

    available = session.context_manager.list_all()
    targets = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  Sin coincidencias para: {raw_tokens!r}\n")
        return

    removed = []

    for ctx in targets:
        result = session.unload_context(ctx)
        removed.extend(result)

    print()

    if not removed:
        print(f"  Ninguno de esos contextos estaba activo.")
    else:
        for ctx in removed:
            print(f"  - {ctx}")

    print()


# ======================================================
# HANDLER DE PREGUNTAS
# ======================================================


def _handle_question(session, question: str):
    collections = session.context_manager.get_loaded_collections()

    if not collections:
        print("\n  Carga un contexto primero.  Ej: /context sociologia\n")
        return

    print("\n  Buscando...\n")

    results, confidence = search(
        question=question,
        mode=session.mode,
        chat_memory=session.chat_memory,
        collections=collections,
    )

    if not results:
        print("  No se encontró contexto relevante.\n")
        return

    context_chunks = []

    for r in results:
        context_chunks.append(
            f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        )

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
        chat_memory=session.chat_memory,
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=session.chat_memory,
    )

    print(f"  Respuesta:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]\n")

    session.add_to_memory(user=question, assistant=answer)


# ======================================================
# LOOP PRINCIPAL
# ======================================================


def start_chat(session: ChatSession):
    print("\n  === CHAT ===")
    print("  Escribe /help para ver los comandos disponibles.\n")

    while True:
        try:
            command = input(session.get_prompt_header()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue

        # ── salida ────────────────────────────────────────
        if command == "/exit":
            print()
            break

        # ── ayuda ─────────────────────────────────────────
        if command in ("/help", "/?"):
            print(HELP)
            continue

        # ── contextos ─────────────────────────────────────
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
            print("\n  Contextos descargados.\n")
            continue

        # ── memoria ───────────────────────────────────────
        if command == "/reset":
            session.reset_memory()
            print("\n  Memoria de conversación limpiada.\n")
            continue

        # ── modo ──────────────────────────────────────────
        if command == "/mode":
            mode = session.toggle_mode()
            print(f"\n  Modo: {mode}\n")
            continue

        # ── comando desconocido ───────────────────────────
        if command.startswith("/"):
            print(f"\n  Comando desconocido: {command!r}  (escribe /help)\n")
            continue

        # ── pregunta ──────────────────────────────────────
        _handle_question(session, command)
