"""
Registro de capacidades por modelo -- reemplaza los antiguos
`ProviderConfig.supports`/`think_param`/`default_think` (strings planas
en .env) por estructuras tipadas, una por backend, en su propio archivo:

    src/config/models/fastflowlm.py
    src/config/models/ollama.py
    src/config/models/gemini.py

Por qué esto y no .env:
  Antes era posible declarar "think_mode" en LOCAL_SUPPORTS y olvidarse
  de configurar LOCAL_THINK_PARAM -- quedaban desincronizados porque eran
  dos strings independientes. Acá `supports_set()` se DERIVA de la misma
  estructura que define el comportamiento real (ThinkMapping), así que
  no puede desincronizarse: si un modelo no tiene ThinkMapping, "think_mode"
  simplemente no aparece en `supports`.

Agregar un modelo nuevo = una entrada en el dict MODELS del archivo de
ese backend. Agregar un backend nuevo = un archivo nuevo acá + un cliente
nuevo en src/llm/backends/ -- nunca hace falta tocar generate.py,
providers.py, ni los routers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.utils.logger import logger

# bool para modelos on/off (FastFlowLM, la mayoría vía Ollama);
# str de nivel para modelos que solo aceptan una escala (ej. gpt-oss).
ThinkValue = bool | Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ThinkMapping:
    """
    Cómo activar/desactivar el razonamiento para un modelo puntual.

    `param` es el nombre real del campo/kwarg que esa API espera --
    cada backend (src/llm/backends/*.py) decide cómo aplicarlo a su
    propio transporte (extra_body para OpenAI-compatible, kwarg nativo
    para Ollama). `on`/`off` son los valores lógicos; no siempre son
    True/False -- gpt-oss vía Ollama, por ejemplo, no se puede apagar
    del todo, solo bajar a su nivel mínimo.
    """

    param: str
    on: ThinkValue
    off: ThinkValue
    # Valor lógico a asumir cuando el request NO trae un override
    # explícito. None = no mandar el campo si no hay override (el
    # modelo/runtime decide su propio comportamiento). True/False = se
    # manda ese valor siempre que no venga un override -- necesario
    # porque algunos modelos (Qwen3) vienen con razonamiento ON por
    # defecto y no hay forma de "apagarlo" si nunca se manda el campo.
    default: bool | None = None


@dataclass(frozen=True)
class ModelCapabilities:
    """Qué acepta un modelo puntual, para un backend puntual."""

    supports_temperature: bool = True
    supports_max_tokens: bool = True
    # None = el modelo no tiene modo de razonamiento activable.
    think: ThinkMapping | None = None
    # Si True, GenerationOptions.extra (passthrough sin validar) se deja
    # pasar para este modelo. False por defecto -- opt-in explícito,
    # igual que think: nunca "cualquier JSON pasa a cualquier servidor".
    allows_extra: bool = False


def supports_set(caps: ModelCapabilities) -> frozenset[str]:
    """
    GenerationOptions que acepta este modelo, derivado de sus
    capacidades reales -- lo que antes era ProviderConfig.supports a
    mano. Usado por GET /api/v1/config/providers y por la validación
    de requests en el router de chat.
    """
    names: set[str] = set()
    if caps.supports_temperature:
        names.add("temperature")
    if caps.supports_max_tokens:
        names.add("max_tokens")
    if caps.think is not None:
        names.add("think_mode")
    if caps.allows_extra:
        names.add("extra")
    return frozenset(names)


# capabilities_key ("fastflowlm", "ollama", "gemini"...) -> {modelo: ModelCapabilities}
# Los imports son perezosos (dentro de la función) para que agregar un
# archivo nuevo en esta carpeta no requiera tocar este módulo salvo por
# esta única línea de registro.
def _registry() -> dict[str, dict[str, dict[str, dict[str, bool]]]]:
    from src.config.models import fastflowlm, gemini, ollama

    return {
        "fastflowlm": fastflowlm._MODELS,
        "ollama": ollama._MODELS,
        "gemini": gemini._MODELS,
    }


# def get_model_capabilities(capabilities_key: str, model: str) -> ModelCapabilities:
#     """
#     Busca las capacidades de `model` dentro del backend `capabilities_key`.

#     Fallback conservador (solo temperature/max_tokens) si el backend o el
#     modelo no están registrados -- preferible a un KeyError en medio de
#     un request; queda un warning en el log para que no pase desapercibido.
#     """
#     backend_models = _registry().get(capabilities_key)
#     if backend_models is None:
#         logger.warning(
#             f"[models] Backend de capacidades desconocido: {capabilities_key!r}. "
#             "Usando capacidades conservadoras (solo temperature/max_tokens)."
#         )
#         return ModelCapabilities()

#     caps = backend_models.get(model)
#     if caps is None:
#         logger.warning(
#             f"[models] Modelo '{model}' no registrado en '{capabilities_key}'. "
#             "Usando capacidades conservadoras (solo temperature/max_tokens)."
#         )
#         return ModelCapabilities()

#     return caps
