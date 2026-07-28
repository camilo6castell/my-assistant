"""
Multi-backend embedding layer.

Design:
  Single entry point: get_encoder() returns a cached EmbeddingEncoder
  instance. All backend logic lives here -- core.py and search.py only
  import get_encoder() and call encode().

  Supported backends (same alias as in .env.providers, see EMBEDDER in
  src/config/settings.py):
    sentence_transformers  -- loads the HuggingFace model in the Python
                              process. No external server. Default.
    ollama                 -- HTTP to Ollama's /api/embeddings endpoint.
                              Ollama runs with Vulkan support (AMD GPU).
    flm                    -- HTTP to the /v1/embeddings endpoint,
                              compatible with the OpenAI API. FastFlowLM
                              uses the NPU.

  Both HTTP backends use the same OpenAI client because FastFlowLM
  exposes /v1/embeddings (OpenAI-compatible) and Ollama also does since
  v0.1. If Ollama changes the signature in the future, a native client
  can be added without changing this module's public interface.

Extensibility:
  Adding a new backend = subclassing EmbeddingEncoder and implementing
  _encode_raw(). The rest of the system does not change.

Normalization:
  All backends return L2-normalized float32 C-contiguous vectors.
  This guarantees that FAISS IndexFlatIP produces correct cosine
  similarity regardless of the backend.
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
# BASE INTERFACE
# ======================================================


class EmbeddingEncoder(ABC):
    """
    Common interface for all embedding backends.

    The only method consumers (core.py, search.py) need is encode().
    Normalization and the cast to float32 are done here, once, regardless
    of the backend.
    """

    @abstractmethod
    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns un-normalized embeddings as floats of any dtype.
        Each subclass implements only this.
        """
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns L2-normalized, float32, C-contiguous embeddings.
        This is the contract with the rest of the system.
        """
        raw = self._encode_raw(texts)
        arr = np.ascontiguousarray(raw, dtype=np.float32)

        # L2 normalization in numpy: arr / ||arr||_2 per row.
        # Equivalent to faiss.normalize_L2 but without depending on faiss
        # here.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoids division by zero
        return cast(NDArray[np.float32], arr / norms)


# ======================================================
# BACKEND: sentence_transformers
# ======================================================


class SentenceTransformersEncoder(EmbeddingEncoder):
    """
    Loads the HuggingFace model directly in the Python process.

    Advantages: no external server, easier to install.
    Disadvantages: the GPU is not accessible if torch is not compiled
    with Vulkan/ROCm support -- on Arch Linux with AMD this depends on
    the torch build.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            f"[embeddings] Backend: sentence_transformers | model={settings.embedding_model}"
        )
        self._model: SentenceTransformer = SentenceTransformer(settings.embedding_model)

    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        # normalize_embeddings=False because normalization is handled by
        # encode()
        result = self._model.encode(texts, normalize_embeddings=False)
        return np.asarray(result)


# ======================================================
# BACKEND: ollama / fastflowlm (HTTP OpenAI-compatible)
# ======================================================


class HttpEmbeddingEncoder(EmbeddingEncoder):
    """
    HTTP backend for Ollama and FastFlowLM.

    Both expose a /v1/embeddings endpoint compatible with the OpenAI
    API, so they share the same client. The difference between ollama
    and flm is only the base_url (EMBEDDER_OLLAMA_URL /
    EMBEDDER_FLM_URL) and the model (EMBEDDER=<backend>,<model>)
    configured in .env.providers.

    Ollama: http://localhost:11434  (Vulkan GPU)
    FastFlowLM: http://127.0.0.1:52625  (NPU)

    The batch is split into chunks of BATCH_SIZE to avoid timeouts on
    large collections -- local runtimes typically have a less generous
    per-request token limit than the OpenAI API.
    """

    BATCH_SIZE = settings.embedding_batch_size

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
            # The API returns embeddings in the same order as the input.
            batch_embeddings: list[list[float]] = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)


# ======================================================
# FACTORY
# ======================================================

_BACKEND_MAP: dict[str, type[EmbeddingEncoder]] = {
    "sentence_transformers": SentenceTransformersEncoder,
    "ollama": HttpEmbeddingEncoder,
    "flm": HttpEmbeddingEncoder,
}


@lru_cache(maxsize=1)
def get_encoder() -> EmbeddingEncoder:
    """
    Returns the cached encoder instance for the configured backend.

    lru_cache(maxsize=1): the encoder is built once per process. This
    avoids reloading the SentenceTransformer model or recreating the HTTP
    client on every call to encode_chunks() or encode_queries().

    If the backend is not in _BACKEND_MAP, fails fast with a clear
    ValueError instead of a confusing AttributeError later.
    """
    backend = settings.embedding_backend.lower()

    if backend not in _BACKEND_MAP:
        valid = ", ".join(sorted(_BACKEND_MAP))
        raise ValueError(f"EMBEDDING_BACKEND={backend!r} not recognized. Valid values: {valid}")

    encoder_cls = _BACKEND_MAP[backend]
    return encoder_cls()
