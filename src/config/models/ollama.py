"""Configuration for Ollama models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.llm.backends.base import ChatTurn

_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-r1": {
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
            "top_p": 0.95,
            "top_k": 20,
        },
        "think": False,
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
        return "think" in self._config

    def set_thinking(self, enabled: bool) -> None:
        if self.supports_thinking():
            self._config["think"] = enabled
