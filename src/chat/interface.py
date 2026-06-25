"""
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
from src.context.manager import LoadedCollection
from src.context.models import SearchResult
from src.context.selector import match_contexts
from src.graph import RAGState, build_rag_graph
from src.llm.generate import ask_llm
from src.prompts.builder import build_prompt
from src.retrieval.search import search

warnings.filterwarnings(
    "ignore",
    message="You're using a BertTokenizerFast tokenizer.*",
)

HELP: str = """
  Comandos:

  /context <tokens>   cargar contexto(s)
  /remove  <tokens>   descargar contexto(s)
  /list               ver contextos disponibles
  /active             ver contextos activos
  /clear              descargar todos los contextos
  /reset              limpiar memoria de conversación
  /mode               alternar modo (RIGUROSO / INTERPRETATIVO)
  /agent              activar / desactivar modo agente (LangGraph)
  /help               mostrar esta ayuda
  /exit               volver al menú principal

  Sintaxis de tokens:
    sociologia                    → todo el namespace
    sociologia/debord             → colección exacta
    sociologia react              → dos namespaces
    sociologia/debord react/hooks → mezcla exactos y namespaces

  Modo agente:
    Cuando está activo, cada pregunta pasa por un grafo LangGraph que
    evalúa la confianza de los resultados. Si es baja, reformula la
    query automáticamente y reintenta antes de generar la respuesta.
"""


# ======================================================
# HELPERS DE PRESENTACIÓN
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
        print("  No hay contextos disponibles.\n")
        return
    _print_contexts_tree(contexts, "Disponibles")


def _print_active(session: ChatSession) -> None:
    active: list[str] = session.get_active_contexts()
    print()
    if not active:
        print("  No hay contextos activos.\n")
        return
    _print_contexts_tree(active, "Activos")


# ======================================================
# HANDLER DE /context Y /remove
# ======================================================


def _handle_context(session: ChatSession, raw_tokens: str) -> None:
    if not raw_tokens:
        print("\n  Uso: /context <tokens>  (ej: /context sociologia react)\n")
        return

    available: list[str] = session.context_manager.list_all()
    targets: list[str] = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  Sin coincidencias para: {raw_tokens!r}\n")
        return

    loaded: list[str] = []
    for ctx in targets:
        result: list[str] = session.load_context(ctx)
        loaded.extend(result)

    print()
    if not loaded:
        print(f"  Ya activos: {', '.join(targets)}")
    else:
        for ctx in loaded:
            print(f"  + {ctx}")
    print()


def _handle_remove(session: ChatSession, raw_tokens: str) -> None:
    if not raw_tokens:
        print("\n  Uso: /remove <tokens>  (ej: /remove sociologia)\n")
        return

    available: list[str] = session.context_manager.list_all()
    targets: list[str] = match_contexts(raw_tokens, available)

    if not targets:
        print(f"\n  Sin coincidencias para: {raw_tokens!r}\n")
        return

    removed: list[str] = []
    for ctx in targets:
        result: list[str] = session.unload_context(ctx)
        removed.extend(result)

    print()
    if not removed:
        print("  Ninguno de esos contextos estaba activo.")
    else:
        for ctx in removed:
            print(f"  - {ctx}")
    print()


# ======================================================
# HANDLER DE PREGUNTAS
# ======================================================


def _handle_question(session: ChatSession, question: str) -> None:
    collections: list[LoadedCollection] = (
        session.context_manager.get_loaded_collections()
    )

    if not collections:
        print("\n  Carga un contexto primero.  Ej: /context sociologia\n")
        return

    print("\n  Buscando...\n")

    results: list[SearchResult]
    confidence: float
    results, confidence = search(
        question=question,
        mode=session.mode,
        collections=collections,
    )

    if not results:
        print("  No se encontró contexto relevante.\n")
        return

    context_chunks: list[str] = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in results
    ]

    # chat_memory ya no se pasa a build_prompt — el historial viaja
    # como mensajes de API en ask_llm → build_messages
    prompt: str = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
    )

    answer: str = ask_llm(
        prompt=prompt,
        chat_memory=session.chat_memory,
    )

    print("  Respuesta:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]\n")

    session.add_to_memory(user=question, assistant=answer)


# ======================================================
# HANDLER DE /agent (LangGraph)
# ======================================================


def _handle_agent_question(session: ChatSession, question: str) -> None:
    """
    Versión del handler de preguntas con adaptive retrieval via LangGraph.

    Diferencias respecto al pipeline lineal (_handle_question):
      - Si la confianza de los resultados iniciales es baja, el grafo
        reformula la query y reintenta una vez antes de generar.
      - El flujo es un grafo de estados (retrieve → evaluate → generate /
        reformulate → retrieve → generate) en lugar de una cadena lineal.
      - Muestra si se activó la reformulación para que el usuario lo sepa.

    El grafo se compila en el primer uso y se reutiliza en llamadas
    posteriores dentro de la misma sesión (build_rag_graph está cacheado
    en el atributo _graph del ChatSession extendido por start_chat).
    """
    collections: list[LoadedCollection] = (
        session.context_manager.get_loaded_collections()
    )

    if not collections:
        print("\n  Carga un contexto primero.  Ej: /context sociologia\n")
        return

    print("\n  [agente] Ejecutando grafo RAG...\n")

    # El grafo compilado se guarda en el frame de start_chat para no
    # recompilarlo en cada pregunta. Se accede via el dict de la función.
    graph = build_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "mode": session.mode,
        "collections": collections,
        "chat_memory": session.chat_memory,
        "results": [],
        "confidence": 0.0,
        "reformulated": False,
        "answer": "",
    }

    final_state: RAGState = graph.invoke(initial_state)  # type: ignore[attr-defined]

    answer = final_state["answer"]
    confidence = final_state["confidence"]
    reformulated = final_state["reformulated"]

    if not answer or answer == "No se encontró contexto relevante para tu pregunta.":
        print("  No se encontró contexto relevante.\n")
        return

    if reformulated:
        print(
            "  [agente] Confianza baja en búsqueda inicial → "
            "query reformulada automáticamente.\n"
        )

    print("  Respuesta:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]")
    if reformulated:
        print("  [reformulado: sí]")
    print()

    session.add_to_memory(user=question, assistant=answer)


def start_chat(session: ChatSession) -> None:
    print("\n  === CHAT ===")
    print("  Type '/help' to see the available commands.\n")

    agent_mode: bool = False

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
            agent_mode = not agent_mode
            status = "activado" if agent_mode else "desactivado"
            print(f"\n  Modo agente: {status}\n")
            continue

        if command.startswith("/"):
            print(f"\n  Unknown command: {command!r}  (type '/help')\n")
            continue

        if agent_mode:
            _handle_agent_question(session, command)
        else:
            _handle_question(session, command)
