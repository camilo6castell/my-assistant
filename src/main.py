"""
Single entry point for the RAG system.

Usage:
  python -m src.main                                   interactive menu
  python -m src.main chat                              go directly to chat
  python -m src.main api                               start the REST API
  python -m src.main web                               start API + frontend (ui/) and open browser
  python -m src.main ingest <cat> <col>                ingest local files
  python -m src.main ingest-url <cat> <col> <url>      ingest a single URL
  python -m src.main crawl <cat> <col> <start_url>     crawl an entire site

Examples:
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
    """Polls host:port until it accepts TCP connections or the timeout expires."""
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
    Starts the API (in-process, like `api`) and the Vite dev server for
    ui/ (subprocess) in parallel, and opens the browser to the frontend
    URL as soon as both are ready.

    Why the API runs in a thread and not a separate subprocess:
    it reuses exactly the same configuration/lifespan as `_cmd_api`
    without duplicating the `uvicorn.run(...)` command in two places.
    The frontend does need to be a real subprocess because it is an
    independent Node/Vite process (pnpm/npm), not something importable
    in Python.

    Ctrl+C stops both: first the Vite subprocess, then the uvicorn
    server (server.should_exit = True), and waits for the API thread
    to finish before exiting.
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
        print(f"\n  Frontend directory not found at {ui_dir}\n")
        sys.exit(1)

    pkg_manager = shutil.which("pnpm") or shutil.which("npm")
    if pkg_manager is None:
        print(
            "\n  Neither 'pnpm' nor 'npm' found in PATH.\n"
            "  Install Node.js (and optionally pnpm) to run ui/.\n"
        )
        sys.exit(1)

    # --- API: same server as `_cmd_api`, running in its own thread ---
    server_config = uvicorn.Config(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    api_thread = threading.Thread(target=server.run, name="uvicorn-api", daemon=True)

    # --- Frontend: Vite dev server as subprocess ---
    ui_process: subprocess.Popen[bytes] | None = None

    # Vite serves on 5173 by default (or the next free port if occupied,
    # but there's no way to know without parsing its stdout). 5173 covers
    # the normal local development case.
    #
    # "localhost" and NOT "127.0.0.1": Node (and therefore Vite) resolves
    # the "localhost" host using Node's own DNS policy, which in recent
    # versions may prefer ::1 (IPv6) over 127.0.0.1 depending on the
    # system -- if Vite ended up bound to ::1, an explicit TCP probe to
    # 127.0.0.1 never connects and the timeout fires even though Vite is
    # ready (this used to happen: Vite's log would show "ready" but
    # _wait_for_port would still report a timeout). socket.create_connection
    # with a hostname (instead of a literal IP) tries all addresses
    # returned by getaddrinfo, in the same order the browser would prefer
    # -- matching whatever Vite actually bound to, whether IPv4 or IPv6.
    frontend_host = "localhost"
    frontend_port = 5173
    frontend_url = f"http://{frontend_host}:{frontend_port}"

    print(f"\n  Starting API at http://{settings.api_host}:{settings.api_port} ...")
    api_thread.start()

    print(f"  Starting frontend (ui/) with '{Path(pkg_manager).name} run dev' ...")
    ui_process = subprocess.Popen(
        [pkg_manager, "run", "dev"],
        cwd=str(ui_dir),
    )

    try:
        api_ready = _wait_for_port(settings.api_host, settings.api_port, timeout=20.0)
        if not api_ready:
            logger.warning("[web] API did not respond within the expected time.")

        ui_ready = _wait_for_port(frontend_host, frontend_port, timeout=30.0)
        if ui_ready:
            print(f"  Opening {frontend_url} in the browser...\n")
            webbrowser.open(frontend_url)
        else:
            print(
                f"\n  Frontend did not respond at {frontend_url} within the "
                "expected time. Check the Vite output above (it may be "
                "using a different port) and open the URL manually.\n"
            )

        # Blocks here with the lifetime of the Vite subprocess -- Ctrl+C
        # interrupts it and falls through to finally, which shuts everything
        # down in order.
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n  Closing frontend and API...")
    finally:
        if ui_process is not None and ui_process.poll() is None:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_process.kill()

        server.should_exit = True
        api_thread.join(timeout=10)
        print("  Done.\n")


def _cmd_ingest(args: list[str]) -> None:
    if len(args) != 2:
        print("Usage: python -m src.main ingest <category> <collection>")
        print("E.g.:  python -m src.main ingest sociologia Guy-Debord_La-sociedad")
        sys.exit(1)
    sys.argv = ["ingest", args[0], args[1]]
    from src.ingest.ingest import main as ingest_main

    ingest_main()


def _cmd_ingest_url(args: list[str]) -> None:
    """
    Ingests a single URL (one page).
    Equivalent to: python -m src.ingest.web_ingest <cat> <col> <url>
    """
    if len(args) != 3:
        print("Usage: python -m src.main ingest-url <category> <collection> <url>")
        print("E.g.:  python -m src.main ingest-url sociologia debord https://sitio.com/articulo")
        sys.exit(1)
    sys.argv = ["ingest-url", args[0], args[1], args[2]]
    from src.ingest.web_ingest import main as web_ingest_main

    web_ingest_main()


def _cmd_crawl(args: list[str]) -> None:
    """
    Crawls an entire site by following links (BFS) within the same domain.

    How it actually works (it's not a sitemap path scan, it doesn't
    infer /child/grandchild from URL structure):
      1. Visits start_url, extracts and chunks its content.
      2. Finds all <a href> on that page pointing to the same domain
         (strict netloc comparison -- 'www.x.com' != 'x.com').
      3. Enqueues those links and repeats until the queue is exhausted
         or settings.max_pages is reached.
    Only discovers pages effectively linked from some already-visited
    page -- it doesn't guess URLs that exist but aren't linked.

    Internally web_crawler.py receives a single 'collection' (without
    separating category), so here we compose f"{category}/{collection}"
    before invoking it -- same argument pattern as ingest and ingest-url.
    """
    if len(args) != 3:
        print("Usage: python -m src.main crawl <category> <collection> <start_url>")
        print("E.g.:  python -m src.main crawl react hooks https://react.dev/learn")
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
        print(f"\n  Unknown command: '{cmd}'")
        print("  Use: python -m src.main --help\n")
        sys.exit(1)

    fn, n_extra_args = COMMANDS[cmd]
    extra = args[1:]

    if n_extra_args == 0:
        fn()
    else:
        fn(extra)


if __name__ == "__main__":
    main()
