"""
Registro de capacidades por modelo -- reemplaza los antiguos
`ProviderConfig.supports`/`think_param`/`default_think` (strings planas
en .env) por un dict JSON por modelo, uno por backend, en su propio
archivo:

    src/config/models/flm.py
    src/config/models/ollama.py
    src/config/models/gemini.py

El nombre de archivo/clave de cada backend acá ("flm", "ollama",
"gemini") coincide 1:1 con el alias de backend usado en .env.providers
(LLM_FLM_URL, EMBEDDER_OLLAMA_URL, LLM_ROL_GENERATE=flm,..., EMBEDDER=
ollama,..., etc. -- ver src/config/settings.py) y con el `capabilities`
de cada ProviderConfig en src/nlp/llm/providers.py. Es a propósito: el
mismo string identifica al backend en las tres capas, así que agregar
un backend nuevo nunca implica inventar un alias distinto en cada capa.

Por qué esto y no .env:
  Antes era posible declarar "think_mode" en LOCAL_SUPPORTS y olvidarse
  de configurar LOCAL_THINK_PARAM -- quedaban desincronizados porque eran
  dos strings independientes. Acá `get_supports()` se DERIVA directo de
  `_MODELS[model]` (misma estructura que build_kwargs() usa para armar
  el request real), así que no puede desincronizarse: si un modelo no
  tiene campo de thinking en su dict, "think_mode" simplemente no
  aparece en `supports`.

  Mismo criterio para la temperatura: NO es un override por-request
  (build_kwargs() no acepta un parámetro `temperature`) -- es una
  propiedad fija de _MODELS[model]["temperature"] en el archivo del
  backend correspondiente. Para cambiarla se edita ese archivo, no hay
  otro lugar (ni .env, ni GenerationOptions, ni un slider en la UI) que
  pueda pisarla.

Agregar un modelo nuevo = una entrada en el dict `_MODELS` del archivo
de ese backend. Agregar un backend nuevo = un archivo nuevo acá (con las
mismas 4 funciones: build_kwargs, supports_thinking, default_think,
supports_max_tokens) + una línea nueva en `_registry()` + un cliente
nuevo en src/llm/backends/ -- nunca hace falta tocar generate.py,
providers.py, ni los routers.

Este módulo es solo un DISPATCHER hacia el archivo del backend correcto
-- no conoce el shape de ningún kwargs dict, ni muta nada. Cada función
de acá simplemente reenvía a `<backend>.<misma_función>(model_name, ...)`.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.nlp.llm.backends.base import ChatTurn


class _ModelBackend(Protocol):
    """
    Forma estructural que debe cumplir cada archivo de backend
    (fastflowlm.py/gemini.py/ollama.py) para poder registrarse acá --
    un módulo con estas funciones matchea este Protocol sin necesidad
    de heredar nada ni de un cast explícito (typing estructural: mypy
    compara la firma real del módulo contra esto). Le da tipado real a
    `_module(...).build_kwargs(...)` en vez de degradar a `Any` como
    pasaría devolviendo `ModuleType` a secas.
    """

    def build_kwargs(
        self,
        model_name: str,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = None,
        think: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def supports_thinking(self, model_name: str) -> bool: ...
    def default_think(self, model_name: str) -> bool | None: ...
    def supports_max_tokens(self, model_name: str) -> bool: ...
    def context_window(self, model_name: str) -> int | None: ...


def _registry() -> dict[str, _ModelBackend]:
    # Import perezoso (dentro de la función) para que agregar un archivo
    # nuevo en esta carpeta no requiera tocar nada acá salvo esta línea.
    from src.config.models import flm, gemini, ollama

    return {
        "flm": flm,
        "ollama": ollama,
        "gemini": gemini,
    }


def _module(capabilities_key: str) -> _ModelBackend:
    registry = _registry()
    try:
        return registry[capabilities_key]
    except KeyError:
        valid = ", ".join(sorted(registry))
        raise ValueError(
            f"Backend de capacidades desconocido: {capabilities_key!r}. Válidos: {valid}"
        ) from None


def build_kwargs(
    capabilities_key: str,
    model_name: str,
    messages: list[ChatTurn],
    *,
    max_tokens: int | None = None,
    think: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Arma el dict de kwargs LISTO para pasarle al SDK del backend
    correspondiente (`**kwargs` directo, sin transformación adicional en
    src/llm/backends/*.py). El shape exacto lo decide cada módulo de
    backend -- OpenAI-compatible (fastflowlm/gemini) devuelve kwargs
    planos; ollama devuelve {model, messages, think, options}.

    No acepta `temperature` a propósito: siempre queda el valor de
    _MODELS[model_name]["temperature"] tal como está escrito en el
    archivo del backend -- no hay override por-request para eso (ver
    docstring del módulo).
    """
    return _module(capabilities_key).build_kwargs(
        model_name,
        messages,
        max_tokens=max_tokens,
        think=think,
        extra=extra,
    )


def get_supports(capabilities_key: str, model_name: str) -> frozenset[str]:
    """
    GenerationOptions que acepta este modelo -- usado por
    GET /api/v1/config/providers (qué controles mostrar en la UI) y por
    la validación de requests en el router de chat.

    No incluye "temperature": no es una GenerationOption, es una
    propiedad fija del modelo (ver docstring del módulo).

    Fallback conservador (frozenset vacío) si el backend o el modelo no
    están registrados, en vez de un 500 -- se loguea como excepción
    normal más arriba en la pila si el caller no lo espera; acá alcanza
    con no reventar el descubrimiento de providers por un modelo mal
    configurado.
    """
    mod = _module(capabilities_key)
    names: set[str] = set()
    if mod.supports_max_tokens(model_name):
        names.add("max_tokens")
    if mod.supports_thinking(model_name):
        names.add("think_mode")
    # "extra" (passthrough) siempre se ofrece como opt-in -- cada backend
    # decide qué hacer con él (fastflowlm/gemini lo mergean a extra_body,
    # ollama lo ignora con warning, ver build_kwargs de cada uno).
    names.add("extra")
    return frozenset(names)


def get_context_window(capabilities_key: str, model_name: str) -> int | None:
    """
    Ventana de contexto (en tokens) del modelo activo, o None si no está
    documentada en su archivo de backend -- en ese caso
    check_context_fit() (src/llm/context_guard.py) no bloquea el
    request, solo lo deja pasar sin verificar (fail-open: preferible a
    romper la app por un modelo sin ese dato completado todavía).
    """
    return _module(capabilities_key).context_window(model_name)


def get_default_think(capabilities_key: str, model_name: str) -> bool | None:
    """
    El valor de thinking YA escrito en _MODELS[model_name] para este
    modelo -- None si el modelo no tiene modo de razonamiento en
    absoluto. Usado por GET /api/v1/config/providers para que el botón
    "Pensar" de la UI arranque reflejando el comportamiento real del
    modelo (ver ProviderInfo.default_think), en vez de arrancar en un
    estado fijo sin relación con la config real.
    """
    return _module(capabilities_key).default_think(model_name)
