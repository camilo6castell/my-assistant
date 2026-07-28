# Módulo 8 — Backends y Clientes LLM

Este módulo documenta la capa de transporte que se comunica con los proveedores LLM. Cada backend tiene su propio cliente que implementa la misma interfaz (`LLMClient`), permitiendo que `generate.py` haga consultas sin saber qué hay detrás.

---

## La interfaz base: `LLMClient`

```python
# src/nlp/llm/backends/base.py

class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class LLMClient(Protocol):
    """Cualquier forma de completar un chat: OpenAI-compatible, Ollama nativo, etc."""

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        """
        `kwargs` ya está construido por src.config.models.build_kwargs()
        en la forma nativa de este backend. Retorna el texto de respuesta,
        o None si está vacío.
        """
        ...
```

### Por qué `Protocol` y no una clase abstracta

`LLMClient` es un `Protocol` (structural subtyping) — cualquier clase que implemente `complete()` con la misma firma es un `LLMClient` válida, sin necesidad de heredar de una clase base. Esto permite que cada backend sea independiente: no importan `base.py`, no heredan de nada. Si mañana quieres un cliente para Claude que habla otro protocolo, solo implementas `complete()` con la firma correcta.

### El tipo `ChatTurn`

```python
ChatTurn = TypedDict con:
    role: "system" | "user" | "assistant"
    content: str
```

Es el tipo de cada mensaje en la lista que se envía al LLM. `build_messages()` en `generate.py` construye esta lista: primero el system prompt, luego el historial (pares user/assistant), y finalmente el prompt actual del usuario.

---

## `OpenAICompatClient` — el cliente multi-uso

```python
# src/nlp/llm/backends/openai_compat.py

class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError:
            logger.exception("[openai_compat] Error calling LLM")
            return None

        if not response.choices:
            logger.warning("[openai_compat] Server response has no 'choices' ...")
            return None

        content = response.choices[0].message.content
        return str(content).strip() if content else None
```

Este cliente sirve para **cualquier servidor compatible con el protocolo OpenAI** (`/v1/chat/completions`):

- **FastFlowLM** (`flm`) — runtime local de inferencia en NPU
- **Gemini** (`gemini`) — a través de su endpoint compatible con OpenAI
- **Cualquier otro** que hable el mismo protocolo

### Por qué envuelve `openai` Python

La librería `openai` ya maneja:
- Conexión HTTP, pool de conexiones, reintentos
- Serialización/deserialización de JSON
- Manejo de errores de red y de la API

No tiene sentido reimplementar esto. El cliente solo tiene que hacer `**kwargs` sobre lo que `build_kwargs()` ya construyó.

### El caso de `response.choices` vacío

Algunos runtimes (como FastFlowLM en NPU) pueden devolver HTTP 200 con un body que no tiene `choices` cuando el proceso de inferencia falla internamente a mitad de generación. Esto no es un `OpenAIError` que el SDK reconozca. Tratarlo como respuesta vacía es consistente con el contrato de `complete()` ("None if empty") y deja que el caller haga el fallback normal.

---

## `OllamaNativeClient` — el cliente directo de Ollama

```python
# src/nlp/llm/backends/ollama_native.py

class OllamaNativeClient:
    def __init__(self, host: str) -> None:
        import ollama  # lazy import
        self._client = ollama.Client(host=host)

    def complete(self, kwargs: dict[str, Any]) -> str | None:
        try:
            response = self._client.chat(**kwargs)
            content = response.message.content
            return content.strip() if content else None
        except Exception:
            logger.exception("[ollama_native] Error querying LLM")
            return None
```

### Por qué nativo y no OpenAI-compat

El endpoint `/v1/chat/completions` de Ollama tiene soporte inconsistente para el parámetro `think` (modo de razonamiento). Hay un issue abierto del proyecto donde `think=true` simplemente no aplica a varios modelos. El cliente nativo (`ollama.Client().chat(..., think=...)`) soporta esto directamente y de forma type-safe.

### Import lazy del paquete `ollama`

El `import ollama` está dentro de `__init__`, no a nivel de módulo. Esto garantiza que alguien que solo usa backends `openai_compat` (FastFlowLM, Gemini) nunca necesita tener el paquete `ollama` instalado — es una dependencia opcional, no un requisito del proyecto completo.

### Manejo de errores

A diferencia de `openai-python`, `ollama-python` no envuelve errores de transporte (servidor caído, timeout) en su propia jerarquía de excepciones. Por eso `complete()` captura `Exception` de forma amplia — preserva el mismo contrato que `OpenAICompatClient`: nunca lanza excepciones, el caller decide el fallback.

---

## `build_kwargs()` — cómo se construyen los kwargs

Antes de que un cliente reciba los kwargs, `generate.py` llama a `build_kwargs()`:

```python
kwargs = build_kwargs(
    config.capabilities,   # "flm" | "ollama" | "gemini"
    config.model,          # "qwen3.5:9b" | "gemini-2.0-flash"
    messages,              # list[ChatTurn]
    max_tokens=...,
    think=...,
    extra=...,
)
```

`build_kwargs()` está en `src/config/models/` y construye un dict en la **forma nativa del backend**:

- Para `openai_compat`: `{"model": "...", "messages": [...], "max_tokens": 1024, "extra_body": {...}}`
- Para `ollama_native`: `{"model": "...", "messages": [...], "think": True, "options": {...}}`

El cliente solo tiene que hacer `**kwargs` sobre ese dict. No necesita interpretar qué es `max_tokens` o `think` — eso lo resolvió `build_kwargs()`.

---

## Flujo completo: de `ask_llm()` al proveedor

```
ask_llm(prompt, provider="generate")
    │
    ▼
build_messages()  →  list[ChatTurn]
    │
    ▼
_complete(messages, provider_name="generate")
    │
    ├── get_client("generate")
    │       │
    │       ├── get_provider("generate")
    │       │       │
    │       │       └── _build_provider_table()
    │       │               │
    │       │               └── settings.role_spec(LLMRole.GENERATE)
    │       │                   → ("flm", "qwen3.5:9b")
    │       │
    │       └── _client_for(config)
    │               │
    │               └── OpenAICompatClient(base_url=..., api_key=...)
    │
    ├── build_kwargs("flm", "qwen3.5:9b", messages, ...)
    │       → {"model": "qwen3.5:9b", "messages": [...], ...}
    │
    └── client.complete(kwargs)
            │
            └── self._client.chat.completions.create(**kwargs)
```

---

## Cómo agregar un nuevo backend

### Caso 1: nuevo backend con un tipo de cliente existente

Si el nuevo backend habla el protocolo OpenAI (ej. OpenAI, Claude vía compatible endpoint):

1. **`settings.py`**: agregar `llm_<nuevo>_url: str = Field(default="")` y opcionalmente `<nuevo>_api_key: str = Field(default="")`
2. **`providers.py`**: agregar entrada en `_backend_url_table()`, `_backend_api_key_table()`, y `_BACKEND_CLIENT` (con `"openai_compat"`)
3. **`src/config/models/<nuevo>.py`**: crear archivo con capacidades de los modelos
4. **`.env.providers`**: configurar `LLM_<NUEVO>_URL=...` y `LLM_ROL_GENERATE=<nuevo>,<modelo>`

No se toca `_CLIENT_FACTORIES` porque `"openai_compat"` ya existe.

### Caso 2: nuevo backend con un tipo de cliente nuevo

Si el backend tiene su propio protocolo (ej. un cliente nativo de Claude):

1. Todo lo anterior, pero con `_BACKEND_CLIENT` apuntando a un nuevo tipo (ej. `"claude_native"`)
2. **`src/nlp/llm/backends/claude_native.py`**: crear archivo implementando `LLMClient`
3. **`providers.py`**: agregar entrada en `_CLIENT_FACTORIES`:
   ```python
   _CLIENT_FACTORIES["claude_native"] = lambda config: ClaudeNativeClient(...)
   ```

En ambos casos: cero cambios en `graph.py`, `nodes.py`, ni `generate.py`.

---

## Cómo agregar un modelo a un backend existente

Solo toca un archivo: `src/config/models/<backend>.py`. Agregar una entrada al diccionario `_MODELS`:

```python
_MODELS = {
    "qwen3.5:9b": {
        "temperature": 0.0,
        "max_tokens": 8192,
        "supports": ["max_tokens", "think_mode"],
    },
    "qwen3.5:2b": {
        "temperature": 0.0,
        "max_tokens": 4096,
        "supports": ["max_tokens"],
    },
}
```

El campo `supports` lista qué overrides de generación acepta ese modelo (verificados por `_validate_think()` en `generate.py` y por `_validate_generation_options()` en el router).
