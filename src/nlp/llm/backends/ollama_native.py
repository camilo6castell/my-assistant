"""
Native client for Ollama, using the official `ollama` package
(pip install ollama) instead of its OpenAI-compatible endpoint.

Why native and not OpenAI-compat: Ollama's /v1/chat/completions
endpoint has inconsistent and poorly documented support for
`think`/`reasoning_effort` (there's an open project issue where
think=true simply doesn't apply to several models). The native
client (`ollama.Client().chat(..., think=...)`) supports it directly
and in a type-safe way -- see ollama._types.ChatResponse in the
package.

The `ollama` import is lazy (inside __init__, not at module level):
so someone who only uses "openai_compat" providers (FastFlowLM,
Gemini) never needs the `ollama` package installed -- it's an
optional dependency, not a project-wide requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    pass


class OllamaNativeClient:
    """
    Pure transport, same as OpenAICompatClient -- doesn't decide what
    each field means, just unpacks `kwargs` (already built by
    src.config.models.ollama.build_kwargs(), shape
    {model, messages, think, options}) against `ollama.Client().chat()`.
    """

    def __init__(self, host: str) -> None:
        import ollama  # lazy: see module docstring

        self._client = ollama.Client(host=host)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        try:
            response = self._client.chat(**kwargs)
            content = response.message.content
            return content.strip() if content else None
        except Exception:
            # Unlike openai-python, ollama-python doesn't wrap transport
            # errors (server down, timeout) in its own hierarchy --
            # ResponseError/RequestError don't cover those cases.
            # Catching broadly here preserves the same contract as
            # OpenAICompatClient: never raises, the caller decides
            # the fallback.
            logger.exception("[ollama_native] Error querying LLM")
            return None
