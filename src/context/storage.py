from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import faiss
import numpy as np

from src.config.settings import BASE_VECTOR_PATH

if TYPE_CHECKING:
    from faiss import Index as FaissIndex

from src.ingest.core import ChunkMetadata


class StorageData(TypedDict):
    """Datos crudos que devuelve CollectionStorage.load()."""

    index: FaissIndex | None
    metadata: list[ChunkMetadata]
    vectors: np.ndarray | None


class CollectionStorage:

    def __init__(self, collection_name: str) -> None:
        self.collection_name: str = collection_name
        self.base_path: Path = BASE_VECTOR_PATH / collection_name

        self.base_path.mkdir(parents=True, exist_ok=True)

        self.index_path: Path = self.base_path / "index.faiss"
        self.metadata_path: Path = self.base_path / "metadata.pkl"
        self.vectors_path: Path = self.base_path / "vectors.npy"

    def exists(self) -> bool:
        return self.index_path.exists()

    def load(self) -> StorageData:
        if not self.exists():
            return StorageData(index=None, metadata=[], vectors=None)

        index: FaissIndex = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            metadata: list[ChunkMetadata] = pickle.load(f)

        vectors: np.ndarray = np.load(self.vectors_path)

        return StorageData(index=index, metadata=metadata, vectors=vectors)

    def save(
        self,
        index: FaissIndex,
        metadata: list[ChunkMetadata],
        vectors: np.ndarray,
    ) -> None:
        faiss.write_index(index, str(self.index_path))
        np.save(self.vectors_path, vectors)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)
