"""
Capacidades de modelos servidos vía Ollama, usando su cliente nativo
(src/llm/backends/ollama_native.py -- paquete `ollama`, no el endpoint
OpenAI-compatible). El endpoint OpenAI-compatible de Ollama tiene
comportamiento inconsistente/no documentado para think en varios
modelos (confirmado por un issue abierto del proyecto); el cliente
nativo sí soporta `think` de forma directa y tipada.
"""

from src.config.models import ModelCapabilities, ThinkMapping

MODELS: dict[str, ModelCapabilities] = {
    "qwen3": ModelCapabilities(
        think=ThinkMapping(param="think", on=True, off=False, default=False),
    ),
    # El cliente nativo de Ollama soporta think=True/False para deepseek-r1
    # según su documentación. Si en la práctica no logra desactivarlo del
    # todo para tu versión de modelo/Ollama, ajustar `off` acá -- es el
    # único lugar que haría falta tocar.
    "deepseek-r1": ModelCapabilities(
        think=ThinkMapping(param="think", on=True, off=False, default=False),
    ),
    # gpt-oss no soporta apagar el razonamiento del todo, solo bajarlo a
    # su nivel mínimo -- por eso `off` es "low", no False.
    "gpt-oss": ModelCapabilities(
        think=ThinkMapping(param="think", on="high", off="low", default=False),
    ),
}
