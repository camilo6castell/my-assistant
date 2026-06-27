"""
Punto de entrada de la API.

    python -m src.api

Levanta uvicorn con host y port de settings, con reload en desarrollo.
"""

import uvicorn

from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",  # uvicorn logs; el RAG usa su propio logger
    )
