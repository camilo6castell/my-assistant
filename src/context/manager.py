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

# FIX #4: TYPE_CHECKING is the standard pattern recognized by mypy and Pylance.
# "if False:" is equivalent in theory but not all checkers process it the same.
# With from __future__ import annotations, annotations are lazy strings,
# so these imports do NOT execute at runtime: zero overhead.
if TYPE_CHECKING:
    import numpy as np
    from faiss import Index as FaissIndex


# ======================================================
# TYPES
# ======================================================


class LoadedCollection(TypedDict):
    """
    Collection fully loaded in memory.
    Extends RawCollection with the logical name assigned by ContextManager.
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
        Traverses base_path and returns all available collections on disk
        in the 'namespace/collection' format, sorted alphabetically.
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
        Resolves a pattern (namespace or exact collection) against
        the collections available on disk.
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
        Loads into memory the collections matching the pattern.
        Already loaded ones are skipped without error.

        Returns the names of the collections actually loaded.
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
                logger.info(f"Context loaded: {context_name}")

            except Exception:
                logger.exception(f"Error loading context: {context_name}")

        return loaded

    # =====================================================
    # DEACTIVATION
    # =====================================================

    def deactivate(self, pattern: str) -> list[str]:
        """
        Unloads from memory the collections matching the pattern.
        Those not active are skipped without error.

        Returns the names of the collections unloaded.
        """
        matches: list[str] = self.resolve_pattern(pattern)
        removed: list[str] = []

        for context_name in matches:
            if context_name in self.loaded_contexts:
                del self.loaded_contexts[context_name]
                removed.append(context_name)
                logger.info(f"Context unloaded: {context_name}")

        return removed

    def clear(self) -> None:
        """Unloads all active contexts."""
        self.loaded_contexts.clear()
        logger.info("All contexts have been unloaded")

    # =====================================================
    # GETTERS
    # =====================================================

    def get_active(self) -> list[str]:
        """Returns the names of active contexts."""
        return list(self.loaded_contexts.keys())

    def get_loaded_collections(self) -> list[LoadedCollection]:
        """Returns the collections loaded in memory."""
        return list(self.loaded_contexts.values())
