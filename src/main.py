"""
Punto de entrada único del sistema RAG.

Uso:
  python -m src.main                                   menú interactivo
  python -m src.main chat                              entra directo al chat
  python -m src.main api                               levanta la API REST
  python -m src.main ingest <cat> <col>                ingest de archivos locales
  python -m src.main ingest-url <cat> <col> <url>      ingest de una URL puntual
  python -m src.main crawl <cat> <col> <start_url>     rastrea un sitio completo

Ejemplos:
  python -m src.main ingest sociologia Guy-Debord_La-sociedad-del-espectaculo
  python -m src.main ingest-url sociologia debord https://sitio.com/articulo
  python -m src.main crawl react hooks https://react.dev/learn
  python -m src.main api
"""

from __future__ import annotations

import sys
from typing import Callable


def _cmd_chat() -> None:
    from src.chat.session import ChatSession
    from src.chat.interface import start_chat

    session = ChatSession()
    start_chat(session)


def _cmd_menu() -> None:
    from src.chat.session import ChatSession
    from src.chat.interface import start_chat
    from src.cli.commands import show_main_menu, show_contexts, show_modes, show_about

    session = ChatSession()

    while True:
        show_main_menu()
        choice = input("> ").strip()

        if choice == "1":
            start_chat(session)
        elif choice == "2":
            show_contexts(session)
        elif choice == "3":
            show_modes()
        elif choice == "4":
            show_about()
        elif choice in ("5", "exit"):
            print("\n  Bye!\n")
            break
        else:
            print("\n  Wrong option. Please choose one of the options above.\n")


def _cmd_api() -> None:
    import uvicorn
    from src.config.settings import settings

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",
    )


def _cmd_ingest(args: list[str]) -> None:
    if len(args) != 2:
        print("Uso: python -m src.main ingest <categoria> <coleccion>")
        print("Ej:  python -m src.main ingest sociologia Guy-Debord_La-sociedad")
        sys.exit(1)
    sys.argv = ["ingest", args[0], args[1]]
    from src.ingest.ingest import main as ingest_main

    ingest_main()


def _cmd_ingest_url(args: list[str]) -> None:
    """
    Ingesta una URL puntual (una sola pagina).
    Equivale a: python -m src.ingest.web_ingest <cat> <col> <url>
    """
    if len(args) != 3:
        print("Uso: python -m src.main ingest-url <categoria> <coleccion> <url>")
        print(
            "Ej:  python -m src.main ingest-url sociologia debord https://sitio.com/articulo"
        )
        sys.exit(1)
    sys.argv = ["ingest-url", args[0], args[1], args[2]]
    from src.ingest.web_ingest import main as web_ingest_main

    web_ingest_main()


def _cmd_crawl(args: list[str]) -> None:
    """
    Rastrea un sitio completo siguiendo enlaces (BFS) dentro del mismo dominio.

    Como funciona realmente (no es escaneo de rutas del arbol del sitio,
    no infiere /hijo/nieto por estructura de URL):
      1. Visita start_url, extrae y chunkea su contenido.
      2. Busca todos los <a href> de esa pagina que apunten al mismo
         dominio (comparacion estricta de netloc -- 'www.x.com' != 'x.com').
      3. Encola esos enlaces y repite hasta agotar la cola o alcanzar
         settings.max_pages.
    Solo descubre paginas efectivamente enlazadas desde alguna pagina ya
    visitada -- no adivina URLs que existen pero no estan enlazadas.

    Internamente web_crawler.py recibe una sola 'collection' (sin separar
    categoria), asi que aqui componemos f"{categoria}/{coleccion}" antes
    de invocarlo -- mismo patron de argumentos que ingest e ingest-url.
    """
    if len(args) != 3:
        print("Uso: python -m src.main crawl <categoria> <coleccion> <start_url>")
        print("Ej:  python -m src.main crawl react hooks https://react.dev/learn")
        sys.exit(1)
    category, collection_name, start_url = args
    sys.argv = ["crawl", category, collection_name, start_url]
    from src.ingest.web_crawler import main as crawler_main

    crawler_main()


def _print_help() -> None:
    print(__doc__)


CommandFn = Callable[..., None]

COMMANDS: dict[str, tuple[CommandFn, int]] = {
    "chat": (_cmd_chat, 0),
    "api": (_cmd_api, 0),
    "ingest": (_cmd_ingest, 2),
    "ingest-url": (_cmd_ingest_url, 3),
    "crawl": (_cmd_crawl, 3),
    "--help": (_print_help, 0),
    "-h": (_print_help, 0),
}


def main() -> None:
    args = sys.argv[1:]

    if not args:
        _cmd_menu()
        return

    cmd = args[0]

    if cmd not in COMMANDS:
        print(f"\n  Comando desconocido: '{cmd}'")
        print("  Usa: python -m src.main --help\n")
        sys.exit(1)

    fn, n_extra_args = COMMANDS[cmd]
    extra = args[1:]

    if n_extra_args == 0:
        fn()
    else:
        fn(extra)


if __name__ == "__main__":
    main()
