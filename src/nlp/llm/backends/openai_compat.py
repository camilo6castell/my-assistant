"""
Client for any OpenAI-compatible server (FastFlowLM, Gemini via its
compatible endpoint, and in general any runtime that speaks the
/v1/chat/completions protocol).
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI, OpenAIError

from src.utils.logger import logger


class OpenAICompatClient:
    """
    Pure transport: knows nothing about concrete models or
    FastFlowLM/Gemini in particular -- serves any provider whose
    ProviderConfig.client is "openai_compat" (see src/nlp/llm/providers.py).
    `kwargs` is already assembled by src.config.models.build_kwargs() with
    the model, messages, and any overrides already resolved.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError:
            logger.exception("[openai_compat] Error calling LLM")
            return None

        # Some OpenAI-compatible runtimes (e.g. FastFlowLM on NPU) may
        # return HTTP 200 with a body missing 'choices' when the inference
        # process fails internally mid-generation -- this is not an error
        # the OpenAI SDK recognizes (not an OpenAIError), so blindly
        # indexing choices[0] crashes with an unhandled TypeError and
        # takes down the whole request with a 500. Treating it as an empty
        # response is consistent with the rest of complete()'s contract
        # (see base.py: "None if empty") and lets the caller do the normal
        # fallback ("Model returned no response", see generate.py) instead
        # of propagating an unhandled exception to the endpoint.
        if not response.choices:
            # Previously we only logged that 'choices' was missing, without
            # saying why -- that leaves blind exactly the case that most
            # needs diagnosing (see: FastFlowLM returning this consistently
            # for a RAG prompt without raising any HTTP error). The OpenAI
            # SDK models the response as a pydantic object, so we dump the
            # full JSON -- it may contain 'usage' (completion_tokens=0?
            # suggests the server cut generation before emitting text), an
            # embedded 'error' the runtime placed outside the standard
            # schema, or finish_reason -- any of those narrows the cause
            # much more than "no choices came back".
            try:
                raw_body = response.model_dump_json()
            except Exception:
                raw_body = repr(response)
            logger.warning(
                "[openai_compat] Server response has no 'choices' "
                "(possible internal runtime failure, e.g. timeout or abort mid-generation) "
                "-- treating as empty response. "
                f"Full body: {raw_body[:2000]}"
            )
            return None
        content = response.choices[0].message.content
        return str(content).strip() if content else None
