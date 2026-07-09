"""Configuration for Gemini models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.llm.backends.base import ChatTurn

_MODELS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash-lite": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 0.95,
    },
}


class ModelConfig:
    def __init__(self, model_name: str) -> None:
        try:
            self._config = deepcopy(_MODELS[model_name])
        except KeyError:
            raise ValueError(f"Unsupported model: {model_name}") from None

        self.model_name = model_name

    def build_kwargs(
        self,
        messages: list[ChatTurn],
    ) -> dict[str, Any]:
        kwargs = deepcopy(self._config)

        kwargs["model"] = self.model_name
        kwargs["messages"] = messages

        return kwargs

    def supports_thinking(self) -> bool:
        return False

    def set_thinking(self, enabled: bool) -> None:
        # Gemini OpenAI-compatible no expone actualmente
        # un equivalente a enable_thinking/think.
        return
