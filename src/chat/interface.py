import warnings

from src.chat.session import ChatSession

from src.llm.generate import ask_llm

from src.retrieval.search import search

from src.prompts.builder import build_prompt

warnings.filterwarnings(
    "ignore",
    message="You're using a BertTokenizerFast tokenizer.*",
)


COMMANDS = """
Comandos:

/context <pattern>
/remove <pattern>
/list
/active
/clear
/reset
/mode
/exit
"""


def print_available_contexts(session):

    contexts = session.context_manager.list_all()

    print()

    if not contexts:

        print("No hay contextos disponibles")

        return

    print("Contextos disponibles:\n")

    for ctx in contexts:
        print(f" - {ctx}")

    print()


def print_active_contexts(session):

    active = session.get_active_contexts()

    print()

    if not active:

        print("No hay contextos activos")

        return

    print("Contextos activos:\n")

    for ctx in active:
        print(f" - {ctx}")

    print()


def handle_question(
    session,
    question,
):

    collections = session.context_manager.get_loaded_collections()

    if not collections:

        print("\nDebes cargar un contexto\n")

        return

    print("\nBuscando contexto...\n")

    results, confidence = search(
        question=question,
        mode=session.mode,
        chat_memory=session.chat_memory,
        collections=collections,
    )

    if not results:

        print("No se encontró contexto relevante")

        return

    context_chunks = []

    for result in results:

        context_chunks.append(f"""
FUENTE: {result.source}
COLECCION: {result.collection}
PAGINA: {result.page}

{result.text}
""")

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

    print("\nRespuesta:\n")

    print(answer)

    print(f"\nConfidence: {confidence:.4f}\n")

    session.add_to_memory(
        user=question,
        assistant=answer,
    )


def start_chat(session: ChatSession):

    print("\n=== CHAT RAG ===\n")

    print(COMMANDS)

    while True:

        command = input(session.get_prompt_header()).strip()

        if not command:
            continue

        if command == "/exit":
            break

        if command == "/list":

            print_available_contexts(session)

            continue

        if command == "/active":

            print_active_contexts(session)

            continue

        if command.startswith("/context"):

            pattern = command.replace(
                "/context",
                "",
                1,
            ).strip()

            loaded = session.load_context(pattern)

            print()

            if not loaded:
                print("No se cargaron contextos")
            else:
                for item in loaded:
                    print(f"+ {item}")

            print()

            continue

        if command.startswith("/remove"):

            pattern = command.replace(
                "/remove",
                "",
                1,
            ).strip()

            removed = session.unload_context(pattern)

            print()

            if not removed:
                print("No se removieron contextos")
            else:
                for item in removed:
                    print(f"- {item}")

            print()

            continue

        if command == "/clear":

            session.clear_contexts()

            print("\nContextos limpiados\n")

            continue

        if command == "/reset":

            session.reset_memory()

            print("\nMemoria limpiada\n")

            continue

        if command == "/mode":

            mode = session.toggle_mode()

            print(f"\nModo actual: {mode}\n")

            continue

        handle_question(
            session,
            command,
        )
