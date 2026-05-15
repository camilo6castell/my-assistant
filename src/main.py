"""
src/main.py

Punto de entrada del sistema RAG.
Muestra el menú principal y delega en los handlers de src/cli/commands.py.
"""

from src.chat.session import ChatSession

from src.cli.menu import show_main_menu

from src.cli.commands import (
    run_chat,
    show_contexts,
    show_active_contexts,
    activate_context,
    deactivate_context,
    reset_contexts,
    toggle_mode,
)

HANDLERS = {
    "1": run_chat,
    "2": show_contexts,
    "3": show_active_contexts,
    "4": activate_context,
    "5": deactivate_context,
    "6": reset_contexts,
    "7": toggle_mode,
}


def main():
    session = ChatSession()

    while True:
        show_main_menu()

        choice = input("> ").strip()

        if choice == "8":
            print("\nHasta luego.\n")
            break

        handler = HANDLERS.get(choice)

        if handler:
            handler(session)
        else:
            print("\nOpción inválida.\n")


if __name__ == "__main__":
    main()
