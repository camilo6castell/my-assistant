from pathlib import Path
from typing import Any

from src.config.settings import BASE_VECTOR_PATH

from src.context.selector import match_namespace

from src.ingest.core import load_collection

from src.utils.logger import logger


class ContextManager:

    def __init__(self) -> None:

        self.base_path: Path = Path(BASE_VECTOR_PATH)

        self.loaded_contexts: dict[str, dict[str, Any]] = {}

    # =====================================================
    # DISCOVERY
    # =====================================================

    def list_all(self) -> list[str]:

        contexts: list[str] = []

        if not self.base_path.exists():
            return contexts

        for namespace in self.base_path.iterdir():

            if not namespace.is_dir():
                continue

            for collection in namespace.iterdir():

                if collection.is_dir():
                    contexts.append(f"{namespace.name}/{collection.name}")

        return sorted(contexts)

    # =====================================================
    # RESOLUTION
    # =====================================================

    def resolve_pattern(self, pattern: str) -> list[str]:

        available: list[str] = self.list_all()

        return match_namespace(
            pattern=pattern,
            available_contexts=available,
        )

    # =====================================================
    # ACTIVATION
    # =====================================================

    def activate(self, pattern: str) -> list[str]:

        matches: list[str] = self.resolve_pattern(pattern)

        if not matches:
            return []

        loaded: list[str] = []

        for context_name in matches:

            if context_name in self.loaded_contexts:
                continue

            try:

                collection: dict[str, Any] = load_collection(context_name)

                collection["collection_name"] = context_name

                self.loaded_contexts[context_name] = collection

                loaded.append(context_name)

                logger.info(f"Contexto cargado: {context_name}")

            except Exception:
                logger.exception(f"Error cargando contexto: {context_name}")

        return loaded

    # =====================================================
    # DEACTIVATION
    # =====================================================

    def deactivate(self, pattern: str) -> list[str]:

        matches: list[str] = self.resolve_pattern(pattern)

        removed: list[str] = []

        for context_name in matches:

            if context_name in self.loaded_contexts:

                del self.loaded_contexts[context_name]

                removed.append(context_name)

                logger.info(f"Contexto descargado: {context_name}")

        return removed

    def clear(self) -> None:

        self.loaded_contexts.clear()

        logger.info("Todos los contextos fueron descargados")

    # =====================================================
    # GETTERS
    # =====================================================

    def get_active(self) -> list[str]:
        return list(self.loaded_contexts.keys())

    def get_loaded_collections(self) -> list[dict[str, Any]]:
        return list(self.loaded_contexts.values())
