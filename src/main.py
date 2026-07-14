"""
Punto de entrada único del sistema RAG.

Uso:
  python -m src.main                                   menú interactivo
  python -m src.main chat                              entra directo al chat
  python -m src.main api                               levanta la API REST
  python -m src.main web                               levanta API + frontend (ui/) y abre browser
  python -m src.main ingest <cat> <col>                ingest de archivos locales
  python -m src.main ingest-url <cat> <col> <url>      ingest de una URL puntual
  python -m src.main crawl <cat> <col> <start_url>     rastrea un sitio completo

Ejemplos:
  python -m src.main ingest sociologia Guy-Debord_La-sociedad-del-espectaculo
  python -m src.main ingest-url sociologia debord https://sitio.com/articulo
  python -m src.main crawl react hooks https://react.dev/learn
  python -m src.main api
  python -m src.main web
"""

from __future__ import annotations

import sys
from collections.abc import Callable


def _cmd_chat() -> None:
    from src.cli.interface import start_chat
    from src.cli.session import ChatSession

    session = ChatSession()
    start_chat(session)


def _cmd_menu() -> None:
    from src.cli.commands import show_about, show_contexts, show_main_menu, show_modes
    from src.cli.interface import start_chat
    from src.cli.session import ChatSession

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


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Sondea host:port hasta que acepte conexiones TCP o venza el timeout."""
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _cmd_web() -> None:
    """
    Levanta la API (in-process, como `api`) y el dev server de Vite de
    ui/ (subproceso) en paralelo, y abre el navegador en la URL del
    frontend apenas ambos responden.

    Por qué la API corre en un thread y no en un subproceso separado:
    reutiliza exactamente la misma configuración/lifespan que `_cmd_api`
    sin duplicar el comando `uvicorn.run(...)` en dos lugares. El
    frontend sí necesita ser un subproceso real porque es un proceso
    Node/Vite independiente (pnpm/npm), no algo importable en Python.

    Ctrl+C detiene ambos: primero el subproceso de Vite, después el
    servidor uvicorn (server.should_exit = True), y se espera a que el
    thread de la API termine antes de salir.
    """
    import shutil
    import subprocess
    import sys
    import threading
    import webbrowser
    from pathlib import Path

    import uvicorn

    from src.config.settings import settings
    from src.utils.logger import logger

    project_root = Path(__file__).resolve().parent.parent
    ui_dir = project_root / "ui"

    if not ui_dir.exists():
        print(f"\n  No se encontró el directorio del frontend en {ui_dir}\n")
        sys.exit(1)

    pkg_manager = shutil.which("pnpm") or shutil.which("npm")
    if pkg_manager is None:
        print(
            "\n  No se encontró 'pnpm' ni 'npm' en el PATH.\n"
            "  Instalá Node.js (y opcionalmente pnpm) para poder levantar ui/.\n"
        )
        sys.exit(1)

    # --- API: mismo server que `_cmd_api`, corriendo en un thread propio ---
    server_config = uvicorn.Config(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    api_thread = threading.Thread(target=server.run, name="uvicorn-api", daemon=True)

    # --- Frontend: dev server de Vite como subproceso ---
    ui_process: subprocess.Popen[bytes] | None = None

    # Vite por defecto sirve en 5173 (o el siguiente puerto libre si está
    # ocupado, pero no hay forma de saberlo de antemano sin parsear su
    # stdout). 5173 cubre el caso normal de desarrollo local.
    #
    # "localhost" y NO "127.0.0.1": Node (y por lo tanto Vite) resuelve
    # el host "localhost" con la política de DNS del propio Node, que en
    # versiones recientes puede preferir ::1 (IPv6) sobre 127.0.0.1
    # según el sistema -- si Vite terminó bindeado a ::1, un probe TCP
    # explícito a 127.0.0.1 nunca conecta y el timeout salta aunque Vite
    # esté listo (esto pasaba antes: el log de Vite mostraba "ready" pero
    # _wait_for_port igual reportaba timeout). socket.create_connection
    # con un hostname (en vez de una IP literal) prueba todas las
    # direcciones que devuelva getaddrinfo, en el mismo orden que
    # preferiría el navegador -- coincide con lo que Vite haya bindeado
    # realmente, sea IPv4 o IPv6.
    frontend_host = "localhost"
    frontend_port = 5173
    frontend_url = f"http://{frontend_host}:{frontend_port}"

    print(f"\n  Levantando API en http://{settings.api_host}:{settings.api_port} ...")
    api_thread.start()

    print(f"  Levantando frontend (ui/) con '{Path(pkg_manager).name} run dev' ...")
    ui_process = subprocess.Popen(
        [pkg_manager, "run", "dev"],
        cwd=str(ui_dir),
    )

    try:
        api_ready = _wait_for_port(settings.api_host, settings.api_port, timeout=20.0)
        if not api_ready:
            logger.warning("[web] La API no respondió en el tiempo esperado.")

        ui_ready = _wait_for_port(frontend_host, frontend_port, timeout=30.0)
        if ui_ready:
            print(f"  Abriendo {frontend_url} en el navegador...\n")
            webbrowser.open(frontend_url)
        else:
            print(
                f"\n  El frontend no respondió en {frontend_url} dentro del "
                "tiempo esperado. Revisá la salida de Vite arriba (puede "
                "estar usando otro puerto) y abrí la URL manualmente.\n"
            )

        # Bloquea acá con la vida del subproceso de Vite -- Ctrl+C lo
        # interrumpe y cae al finally, que apaga todo en orden.
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n  Cerrando frontend y API...")
    finally:
        if ui_process is not None and ui_process.poll() is None:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()

        server.should_exit = True
        api_thread.join(timeout=10)
        print("  Listo.\n")


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
        print("Ej:  python -m src.main ingest-url sociologia debord https://sitio.com/articulo")
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
    "web": (_cmd_web, 0),
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
