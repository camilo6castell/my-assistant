"""
src/cli/commands.py

Handlers para cada opción del menú principal.
Cada función recibe la sesión activa y realiza la acción correspondiente,
delegando en la lógica de negocio de session/context_manager.
"""

from src.chat.interface import start_chat

# ======================================================
# 1. CHAT
# ======================================================


def run_chat(session):
    start_chat(session)


# ======================================================
# 2. VER CONTEXTOS (todos los disponibles en disco)
# ======================================================


def show_contexts(session):
    contexts = session.context_manager.list_all()

    print()

    if not contexts:
        print("No hay contextos disponibles.\n")
        return

    print("Contextos disponibles:\n")

    for ctx in contexts:
        print(f"  - {ctx}")

    print()


# ======================================================
# 3. CONTEXTOS ACTIVOS (cargados en memoria)
# ======================================================


def show_active_contexts(session):
    active = session.get_active_contexts()

    print()

    if not active:
        print("No hay contextos activos.\n")
        return

    print("Contextos activos:\n")

    for ctx in active:
        print(f"  - {ctx}")

    print()


# ======================================================
# 4. ACTIVAR CONTEXTO
# ======================================================


def activate_context(session):
    print()

    pattern = input("Patrón a activar (ej: sociologia/* o sociologia/libro): ").strip()

    if not pattern:
        print("Patrón inválido.\n")
        return

    loaded = session.load_context(pattern)

    print()

    if not loaded:
        print("No se encontraron contextos para ese patrón.\n")
    else:
        for ctx in loaded:
            print(f"  + {ctx}")
        print()


# ======================================================
# 5. DESACTIVAR CONTEXTO
# ======================================================


def deactivate_context(session):
    active = session.get_active_contexts()

    print()

    if not active:
        print("No hay contextos activos.\n")
        return

    print("Contextos activos:\n")
    for ctx in active:
        print(f"  - {ctx}")
    print()

    pattern = input(
        "Patrón a desactivar (ej: sociologia/* o sociologia/libro): "
    ).strip()

    if not pattern:
        print("Patrón inválido.\n")
        return

    removed = session.unload_context(pattern)

    print()

    if not removed:
        print("No se encontraron contextos activos para ese patrón.\n")
    else:
        for ctx in removed:
            print(f"  - {ctx}")
        print()


# ======================================================
# 6. RESETEAR CONTEXTOS (desactiva todos)
# ======================================================


def reset_contexts(session):
    session.clear_contexts()
    print("\nTodos los contextos fueron desactivados.\n")


# ======================================================
# 7. CAMBIAR MODO
# ======================================================


def toggle_mode(session):
    mode = session.toggle_mode()
    print(f"\nModo actual: {mode}\n")
