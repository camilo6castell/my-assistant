from src.chat.interface import start_chat


def show_contexts(session):
    contexts = session.context_manager.list_all()

    print()

    if not contexts:
        print("No hay contextos disponibles.\n")
        return

    print("Contextos disponibles:")

    for context in contexts:
        print(f" - {context}")

    print()


def show_active_contexts(session):
    print()

    if not session.active_contexts:
        print("No hay contextos activos.\n")
        return

    print("Contextos activos:")

    for context in session.active_contexts:
        print(f" - {context}")

    print()


def activate_context(session):
    print()

    context_name = input("Contexto a activar: ").strip()

    if not context_name:
        print("Nombre inválido.\n")
        return

    success = session.load_context(context_name)

    if success:
        print(f"\nContexto activado: {context_name}\n")
    else:
        print(f"\nNo se pudo cargar: {context_name}\n")


def deactivate_context(session):
    print()

    if not session.active_contexts:
        print("No hay contextos activos.\n")
        return

    context_name = input("Contexto a desactivar: ").strip()

    if context_name not in session.active_contexts:
        print("\nEse contexto no está activo.\n")
        return

    session.unload_context(context_name)

    print(f"\nContexto desactivado: {context_name}\n")


def reset_contexts(session):
    session.reset_contexts()

    print("\nTodos los contextos fueron desactivados.\n")


def toggle_mode(session):
    session.toggle_mode()

    mode = "INTERPRETATIVO" if session.interpretative_mode else "RIGUROSO"

    print(f"\nModo actual: {mode}\n")


def run_chat(session):
    start_chat(session)
