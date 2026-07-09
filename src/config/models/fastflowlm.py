"""Configuration for FastFlowLM models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.config.settings import settings
from src.llm.backends.base import ChatTurn

_MODELS: dict[str, dict[str, Any]] = {
    "qwen3.5:9b": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    },
    "gpt-oss:20b": {
        "timeout": 600,
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "qwen3.5:2b": {
        "timeout": 600,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "enable_thinking": False,
        },
    },
}


class FastFlowLMModelConfig:
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
        extra = self._config.get("extra_body", {})

        if "enable_thinking" in extra:
            return True

        chat_kwargs = extra.get("chat_template_kwargs", {})
        return "enable_thinking" in chat_kwargs

    def set_thinking(self, enabled: bool) -> None:
        if not self.supports_thinking():
            return

        extra = self._config.setdefault("extra_body", {})

        if "enable_thinking" in extra:
            extra["enable_thinking"] = enabled

        if "chat_template_kwargs" in extra:
            extra["chat_template_kwargs"]["enable_thinking"] = enabled


fastFlowLMModelConfig = FastFlowLMModelConfig(settings.local_model)
