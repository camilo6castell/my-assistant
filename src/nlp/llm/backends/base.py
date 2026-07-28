"""
Common contract across LLM client implementations.

generate.py builds the kwargs dict ALREADY RESOLVED (model/messages/
max_tokens/think/extra already translated to the shape that backend
expects -- temperature doesn't pass through here, it travels fixed
inside that same dict from _MODELS[model], see
src/config/models/build_kwargs()) and hands it to any LLMClient
without knowing what's behind it -- OpenAI SDK, native Ollama client,
whatever. Each implementation only has to unpack that dict against its
own SDK (`**kwargs`); it never touches the mechanics of what each
field means.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict


class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    """Any form of completing a chat: OpenAI-compatible, native Ollama, etc."""

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        """
        `kwargs` is already built by src.config.models.build_kwargs() in
        the native shape of this backend. Returns the response text,
        or None if empty.
        """
        ...
