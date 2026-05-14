from pathlib import Path

from src.config.settings import BASE_VECTOR_PATH

from src.context.selector import match_namespace

from src.ingest.core import load_collection

from src.utils.logger import logger


class ContextManager:

    def __init__(self):

        self.base_path = Path(BASE_VECTOR_PATH)

        self.loaded_contexts = {}

    # =====================================================
    # DISCOVERY
    # =====================================================

    def list_all(self):

        contexts = []

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

    def resolve_pattern(self, pattern: str):

        available = self.list_all()

        return match_namespace(
            pattern=pattern,
            available_contexts=available,
        )

    # =====================================================
    # ACTIVATION
    # =====================================================

    def activate(self, pattern: str):

        matches = self.resolve_pattern(pattern)

        if not matches:
            return []

        loaded = []

        for context_name in matches:

            if context_name in self.loaded_contexts:
                continue

            try:

                collection = load_collection(context_name)

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

    def deactivate(self, pattern: str):

        matches = self.resolve_pattern(pattern)

        removed = []

        for context_name in matches:

            if context_name in self.loaded_contexts:

                del self.loaded_contexts[context_name]

                removed.append(context_name)

                logger.info(f"Contexto descargado: {context_name}")

        return removed

    def clear(self):

        self.loaded_contexts.clear()

        logger.info("Todos los contextos fueron descargados")

    # =====================================================
    # GETTERS
    # =====================================================

    def get_active(self):
        return list(self.loaded_contexts.keys())

    def get_loaded_collections(self):
        return list(self.loaded_contexts.values())
