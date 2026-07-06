"""
Capacidades de modelos Gemini vía el endpoint OpenAI-compatible de Google.

Gemini 2.x tiene su propio "thinking budget" interno, pero no está
expuesto a través del endpoint OpenAI-compatible con un campo simple
on/off equivalente a think_mode -- por eso no se declara ThinkMapping
acá. Si en el futuro se agrega soporte, es una línea en este archivo.
"""

from src.config.models import ModelCapabilities

MODELS: dict[str, ModelCapabilities] = {
    "gemini-2.5-flash-lite": ModelCapabilities(),
}
