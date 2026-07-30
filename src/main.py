"""
Single entry point for the RAG system.

Usage:
  python -m src.main                                   interactive menu
  python -m src.main chat                              go directly to chat
  python -m src.main api                               start the REST API
  python -m src.main mcp                               start the MCP server (Streamable HTTP)
  python -m src.main server                            start API + MCP in a single process (recommended)

  python -m src.main ingest <cat> <col>                ingest local files
  python -m src.main ingest-url <cat> <col> <url>      ingest a single URL
  python -m src.main crawl <cat> <col> <start_url>     crawl an entire site

Examples:
  python -m src.main ingest sociologia Guy-Debord_La-sociedad-del-espectaculo
  python -m src.main ingest-url sociologia debord https://sitio.com/articulo
  python -m src.main crawl react hooks https://react.dev/learn
  python -m src.main server                             recommended — API + MCP in one process
  python -m src.main api
  python -m src.main mcp

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


def _cmd_server() -> None:
    """
    Unified server: REST API + MCP server in a single process.

    The API includes the MCP server mounted at /mcp on the FastAPI app
    (see src/api/app.py). The frontend connects to the API for both REST
    endpoints (/api/v1/...) and MCP tools (/mcp/...).

    Usage:
      python -m src.main server
    """
    import uvicorn

    from src.config.settings import settings

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",
    )


def _cmd_mcp() -> None:
    import uvicorn

    from src.config.settings import settings
    from src.mcp_server.server import create_app

    app = create_app()
    uvicorn.run(
        app,
        host=settings.mcp_host,
        port=settings.mcp_port,
        reload=False,
        log_level="warning",
    )





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
    "server": (_cmd_server, 0),
    "mcp": (_cmd_mcp, 0),
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
