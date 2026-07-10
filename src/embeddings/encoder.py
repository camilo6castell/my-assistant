"""
Capa de embeddings multi-backend.

Diseño:
  Un único punto de entrada: get_encoder() devuelve una instancia de
  EmbeddingEncoder cacheada. Toda la lógica de backend vive aquí —
  core.py y search.py solo importan get_encoder() y llaman a encode().

  Backends soportados:
    sentence_transformers  — carga el modelo HuggingFace en proceso Python.
                             Sin servidor externo. Default.
    ollama                 — HTTP al endpoint /api/embeddings de Ollama.
                             Ollama corre con soporte Vulkan (GPU AMD).
    fastflowlm             — HTTP al endpoint /v1/embeddings, compatible
                             con la API OpenAI. FastFlowLM usa la NPU.

  Ambos backends HTTP usan el mismo client OpenAI porque FastFlowLM expone
  /v1/embeddings (OpenAI-compatible) y Ollama también lo expone desde v0.1.
  Si en el futuro Ollama cambia la firma, se puede añadir un cliente nativo
  sin cambiar la interfaz pública de este módulo.

Extensibilidad:
  Agregar un backend nuevo = subclasear EmbeddingEncoder e implementar
  _encode_raw(). El resto del sistema no cambia.

Normalización:
  Todos los backends devuelven vectores L2-normalizados float32 C-contiguos.
  Esto garantiza que IndexFlatIP de FAISS produce similitud coseno correcta
  independientemente del backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from src.config.settings import settings
from src.utils.logger import logger

# ======================================================
# INTERFAZ BASE
# ======================================================


class EmbeddingEncoder(ABC):
    """
    Interfaz común para todos los backends de embedding.

    El único método que los consumidores (core.py, search.py) necesitan
    es encode(). La normalización y el cast a float32 se hacen aquí,
    una sola vez, independientemente del backend.
    """

    @abstractmethod
    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """
        Devuelve embeddings SIN normalizar, como floats de cualquier dtype.
        Cada subclase implementa solo esto.
        """
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Devuelve embeddings L2-normalizados, float32, C-contiguos.
        Este es el contrato con el resto del sistema.
        """
        raw = self._encode_raw(texts)
        arr = np.ascontiguousarray(raw, dtype=np.float32)

        # L2-normalización en numpy: arr / ||arr||₂ por fila.
        # Equivalente a faiss.normalize_L2 pero sin depender de faiss aquí.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # evita división por cero
        return cast(NDArray[np.float32], arr / norms)


# ======================================================
# BACKEND: sentence_transformers
# ======================================================


class SentenceTransformersEncoder(EmbeddingEncoder):
    """
    Carga el modelo HuggingFace directamente en el proceso Python.

    Ventajas: sin servidor externo, más fácil de instalar.
    Desventajas: la GPU no es accesible si torch no tiene soporte Vulkan/ROCm
    compilado — en Arch Linux con AMD esto depende de la build de torch.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            f"[embeddings] Backend: sentence_transformers | model={settings.embedding_model}"
        )
        self._model: SentenceTransformer = SentenceTransformer(settings.embedding_model)

    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        # normalize_embeddings=False porque la normalización la hace encode()
        result = self._model.encode(texts, normalize_embeddings=False)
        return np.asarray(result)


# ======================================================
# BACKEND: ollama / fastflowlm (HTTP OpenAI-compatible)
# ======================================================


class HttpEmbeddingEncoder(EmbeddingEncoder):
    """
    Backend HTTP para Ollama y FastFlowLM.

    Ambos exponen un endpoint /v1/embeddings compatible con la API
    OpenAI, por lo que comparten el mismo client. La diferencia entre
    ollama y fastflowlm es solo la base_url y el modelo configurados
    en .env.providers.

    Ollama: http://localhost:11434  (GPU Vulkan)
    FastFlowLM: http://127.0.0.1:52625  (NPU)

    El batch se parte en lotes de BATCH_SIZE para evitar timeouts
    en colecciones grandes — los runtimes locales suelen tener un
    límite de tokens por request menos generoso que la API de OpenAI.
    """

    BATCH_SIZE = 64

    def __init__(self) -> None:
        logger.info(
            f"[embeddings] Backend: {settings.embedding_backend} | "
            f"base_url={settings.embedding_base_url} | "
            f"model={settings.embedding_model}"
        )
        self._client: OpenAI = OpenAI(
            base_url=f"{settings.embedding_base_url.rstrip('/')}/v1",
            api_key=settings.embedding_api_key,
        )
        self._model: str = settings.embedding_model

    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = list(texts[i : i + self.BATCH_SIZE])
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # La API devuelve los embeddings en el mismo orden que el input.
            batch_embeddings: list[list[float]] = [
                item.embedding for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)


# ======================================================
# FACTORY
# ======================================================

_BACKEND_MAP: dict[str, type[EmbeddingEncoder]] = {
    "sentence_transformers": SentenceTransformersEncoder,
    "ollama": HttpEmbeddingEncoder,
    "fastflowlm": HttpEmbeddingEncoder,
}


@lru_cache(maxsize=1)
def get_encoder() -> EmbeddingEncoder:
    """
    Devuelve la instancia cacheada del encoder para el backend configurado.

    lru_cache(maxsize=1): el encoder se construye una sola vez por proceso.
    Esto evita recargar el modelo SentenceTransformer o recrear el client
    HTTP en cada llamada a encode_chunks() o encode_queries().

    Si el backend no está en _BACKEND_MAP, falla rápido con un ValueError
    claro en lugar de un AttributeError confuso más adelante.
    """
    backend = settings.embedding_backend.lower()

    if backend not in _BACKEND_MAP:
        valid = ", ".join(sorted(_BACKEND_MAP))
        raise ValueError(
            f"EMBEDDING_BACKEND={backend!r} no reconocido. " f"Valores válidos: {valid}"
        )

    encoder_cls = _BACKEND_MAP[backend]
    return encoder_cls()
