import warnings

from typing import Optional

from src.llm.generate import ask_llm

from src.retrieval.search import search

from src.prompts.builder import build_prompt

from src.context.manager import (
    list_contexts,
    load_context,
)

from src.utils.logger import logger

from src.utils.env import (
    INTERPRETATIVE_MODE,
    BASE_VECTOR_PATH,
)

warnings.filterwarnings(
    "ignore",
    message="You're using a BertTokenizerFast tokenizer.*",
)

# ===============================
# GLOBAL STATE
# ===============================

CHAT_MEMORY: list[dict] = []

CURRENT_MODE = INTERPRETATIVE_MODE

CURRENT_CONTEXT: Optional[dict] = None


# ===============================
# MAIN LOOP
# ===============================


def main():

    global CHAT_MEMORY
    global CURRENT_MODE
    global CURRENT_CONTEXT

    logger.info("Iniciando RAG")

    print("\n=== RAG MULTI-CONTEXTO ===")

    print("\nComandos:")
    print("  /context <nombre>")
    print("  /list")
    print("  /mode")
    print("  /reset")
    print("  /exit o /bye\n")

    contexts = list_contexts()

    if not contexts:

        logger.warning("No se encontraron contextos.")

        print(f"No hay contextos disponibles " f"en {BASE_VECTOR_PATH}")

        return

    print("Contextos disponibles:")

    for c in contexts:
        print(f"  - {c}")

    print()

    while True:

        mode_label = "INTERPRETATIVO" if CURRENT_MODE else "RIGUROSO"

        context_label = CURRENT_CONTEXT["name"] if CURRENT_CONTEXT else "SIN-CONTEXTO"

        question = input(f"[{context_label} | " f"{mode_label}] > ").strip()

        if not question:
            continue

        # ===============================
        # EXIT
        # ===============================

        if question.lower() in ["/exit", "/bye"]:

            logger.info("Sesión finalizada.")

            print("\nSesión finalizada.\n")

            break

        # ===============================
        # RESET
        # ===============================

        if question.lower() == "/reset":

            CHAT_MEMORY = []

            logger.info("Memoria reiniciada.")

            print("Memoria reiniciada.\n")

            continue

        # ===============================
        # MODE
        # ===============================

        if question.lower() == "/mode":

            CURRENT_MODE = not CURRENT_MODE

            logger.info(f"Modo cambiado: " f"{CURRENT_MODE}")

            print(
                f"Modo cambiado a: "
                f"{'INTERPRETATIVO' if CURRENT_MODE else 'RIGUROSO'}\n"
            )

            continue

        # ===============================
        # LIST
        # ===============================

        if question.lower() == "/list":

            print("\nContextos disponibles:")

            for c in list_contexts():
                print(f"  - {c}")

            print()

            continue

        # ===============================
        # LOAD CONTEXT
        # ===============================

        if question.lower().startswith("/context"):

            parts = question.split()

            if len(parts) != 2:

                print("Uso: /context nombre\n")

                continue

            loaded = load_context(parts[1])

            if loaded is None:

                print("Contexto no encontrado.\n")

                continue

            CURRENT_CONTEXT = loaded

            print(f"\nContexto cargado: " f"{parts[1]}\n")

            continue

        # ===============================
        # NO CONTEXT
        # ===============================

        if CURRENT_CONTEXT is None:

            print("Debes cargar un contexto.\n")

            continue

        print("Buscando respuesta...\n")

        indices, confidence = search(
            question=question,
            interpretative_mode=CURRENT_MODE,
            chat_memory=CHAT_MEMORY,
            index=CURRENT_CONTEXT["index"],
            metadata=CURRENT_CONTEXT["metadata"],
            chunk_vectors=CURRENT_CONTEXT["chunk_vectors"],
        )

        if not indices:

            logger.warning("No se encontraron chunks relevantes.")

            print("No se encontraron fragmentos relevantes.\n")

            continue

        context_chunks = []

        for idx in indices:

            chunk = CURRENT_CONTEXT["metadata"][idx]

            context_chunks.append(
                f"Fuente: {chunk['source']} "
                f"(página {chunk['page']})\n"
                f"{chunk['text']}"
            )

        prompt = build_prompt(
            context_chunks=context_chunks,
            question=question,
            interpretative_mode=CURRENT_MODE,
            chat_memory=CHAT_MEMORY,
        )

        answer = ask_llm(prompt)

        print("\nRespuesta:\n")

        print(answer)

        print(f"\nNivel de confianza: " f"{confidence:.4f}\n")

        CHAT_MEMORY.append(
            {
                "user": question,
                "assistant": answer,
            }
        )


if __name__ == "__main__":
    main()
