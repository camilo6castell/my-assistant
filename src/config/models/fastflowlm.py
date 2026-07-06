"""
Capacidades de modelos servidos vía FastFlowLM (runtime OpenAI-compatible
en http://127.0.0.1:52625/v1 típicamente -- ver LOCAL_BASE_URL en .env).

FLM documenta el modo de razonamiento como un campo top-level "think" en
el payload del request en Server Mode (ver model card de cada modelo en
https://fastflowlm.com).
"""

from src.config.models import ModelCapabilities, ThinkMapping

MODELS: dict[str, ModelCapabilities] = {
    "qwen3:8b": ModelCapabilities(
        think=ThinkMapping(param="think", on=True, off=False, default=False),
    ),
}
