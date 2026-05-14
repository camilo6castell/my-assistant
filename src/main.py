from src.chat.session import ChatSession

from src.chat.interface import start_chat

from src.cli.menu import show_main_menu

from src.cli.commands import (
    show_contexts,
)


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

            print("\nHasta luego.\n")

            break

        else:

            print("\nOpción inválida.\n")


if __name__ == "__main__":
    main()
