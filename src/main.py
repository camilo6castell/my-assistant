"""
src/main.py

Punto de entrada del sistema RAG.
"""

from src.chat.session import ChatSession
from src.chat.interface import start_chat
from src.cli.menu import show_main_menu
from src.cli.commands import show_contexts, show_modes, show_about


def main():
    session = ChatSession()

    while True:
        show_main_menu()

        choice = input("> ").strip()

        if choice == "1":
            start_chat(session)

        elif choice == "2":
            show_contexts(session)

        elif choice == "3":
            show_modes(session)

        elif choice == "4":
            show_about(session)

        elif choice == "5":
            print("\n  Hasta luego.\n")
            break

        else:
            print("\n  Opción inválida.\n")


if __name__ == "__main__":
    main()
