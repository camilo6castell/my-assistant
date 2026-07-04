from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from src.config.settings import settings
from src.context.selector import match_namespace
from src.ingest.core import (
    ChunkMetadata,
    CollectionPaths,
    RawCollection,
    load_collection,
)
from src.utils.logger import logger

# FIX #4: TYPE_CHECKING es el patrón estándar reconocido por mypy y Pylance.
# "if False:" es equivalente en teoría pero no todos los checkers lo procesan igual.
# Con from __future__ import annotations las anotaciones son strings lazy,
# por lo que estos imports NO se ejecutan en runtime: cero overhead.
if TYPE_CHECKING:
    import numpy as np
    from faiss import Index as FaissIndex


# ======================================================
# TIPOS
# ======================================================


class LoadedCollection(TypedDict):
    """
    Colección completamente cargada en memoria.
    Extiende RawCollection con el nombre lógico asignado por ContextManager.
    """

    index: FaissIndex | None
    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None
    paths: CollectionPaths
    collection_name: str


# ======================================================
# MANAGER
# ======================================================


class ContextManager:

    def __init__(self) -> None:
        self.base_path: Path = settings.vector_store_path_for_backend
        self.loaded_contexts: dict[str, LoadedCollection] = {}

    # =====================================================
    # DISCOVERY
    # =====================================================

    def list_all(self) -> list[str]:
        """
        Recorre base_path y devuelve todas las colecciones disponibles
        en disco con el formato 'namespace/coleccion', ordenadas
        alfabéticamente.
        """
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
        """
        Resuelve un patrón (namespace o colección exacta) contra
        las colecciones disponibles en disco.
        """
        available: list[str] = self.list_all()

        return match_namespace(
            pattern=pattern,
            available_contexts=available,
        )

    # =====================================================
    # ACTIVATION
    # =====================================================

    def activate(self, pattern: str) -> list[str]:
        """
        Carga en memoria las colecciones que coinciden con el patrón.
        Las ya cargadas se omiten sin error.

        Retorna los nombres de las colecciones efectivamente cargadas.
        """
        matches: list[str] = self.resolve_pattern(pattern)

        if not matches:
            return []

        loaded: list[str] = []

        for context_name in matches:
            if context_name in self.loaded_contexts:
                continue

            try:
                raw: RawCollection = load_collection(context_name)

                self.loaded_contexts[context_name] = LoadedCollection(
                    index=raw["index"],
                    metadata=raw["metadata"],
                    vectors=raw["vectors"],
                    paths=raw["paths"],
                    collection_name=context_name,
                )

                loaded.append(context_name)
                logger.info(f"Contexto cargado: {context_name}")

            except Exception:
                logger.exception(f"Error cargando contexto: {context_name}")

        return loaded

    # =====================================================
    # DEACTIVATION
    # =====================================================

    def deactivate(self, pattern: str) -> list[str]:
        """
        Descarga de memoria las colecciones que coinciden con el patrón.
        Las que no están activas se omiten sin error.

        Retorna los nombres de las colecciones descargadas.
        """
        matches: list[str] = self.resolve_pattern(pattern)
        removed: list[str] = []

        for context_name in matches:
            if context_name in self.loaded_contexts:
                del self.loaded_contexts[context_name]
                removed.append(context_name)
                logger.info(f"Contexto descargado: {context_name}")

        return removed

    def clear(self) -> None:
        """Descarga todos los contextos activos."""
        self.loaded_contexts.clear()
        logger.info("Todos los contextos fueron descargados")

    # =====================================================
    # GETTERS
    # =====================================================

    def get_active(self) -> list[str]:
        """Devuelve los nombres de los contextos activos."""
        return list(self.loaded_contexts.keys())

    def get_loaded_collections(self) -> list[LoadedCollection]:
        """Devuelve las colecciones cargadas en memoria."""
        return list(self.loaded_contexts.values())
