# Módulo 8 — Multi-Proveedor y Reviewer

Este módulo documenta la arquitectura de proveedores LLM (provider-per-role) y el ciclo de auto-corrección review → correct. El sistema puede usar diferentes modelos para diferentes pasos del pipeline, todo configurado desde variables de entorno.

---

## El eje Backend y el eje Role

La arquitectura tiene dos ejes independientes:

**Backend** — un runtime concreto: FastFlowLM (`flm`), Ollama (`ollama`), Gemini (`gemini`). Define dónde conectarse (base_url), con qué credencial (api_key), y qué implementación de `LLMClient` usar. Es mecánica de transporte puro, no sabe nada de modelos concretos. Se configura una vez en `.env.providers`.

**Role** — cada paso del pipeline que necesita un LLM. El `LLMRole` identifica qué función cumple:

```python
class LLMRole(StrEnum):
    GENERATE      = "generate"       # genera la respuesta final
    REFORMULATE   = "reformulate"    # reescribe la query para mejorar recall
    REVIEW        = "review"         # evalúa grounding y citación
    WEB_SUPPLEMENT = "web_supplement" # complementa con información de la web
```

Cada role elige, en `.env.providers`, qué backend + qué modelo lo sirve. Los dos ejes son independientes: dos roles pueden compartir backend con diferentes modelos, o compartir modelo con diferentes backends.

---

## `providers.py` — el registro de proveedores

### `ProviderConfig` — la configuración de cada proveedor

```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str           # nombre del role que resolvió esta config (ej. "generate")
    backend: str        # "flm" | "ollama" | "gemini"
    base_url: str       # URL del backend
    api_key: str        # credencial
    model: str          # nombre del modelo
    client: str         # "openai_compat" | "ollama_native"
    capabilities: str   # key en src/config/models/ para las capacidades del modelo
```

Es un dataclass `frozen=True` (inmutable). El campo `name` tiene `compare=False`: dos roles con el mismo backend+modelo comparten el mismo cliente cacheado, independientemente de su nombre de role.

### La tabla de backends

Las URLs y credenciales se resuelven en funciones (no a nivel de módulo) para que los tests que muten `settings` vean valores actualizados:

```python
def _backend_url_table() -> dict[str, str]:
    return {
        "flm": settings.llm_flm_url,
        "ollama": settings.llm_ollama_url,
        "gemini": settings.llm_gemini_url,
    }

def _backend_api_key_table() -> dict[str, str]:
    return {
        "flm": "not-needed",      # FastFlowLM local no valida la key
        "ollama": "not-needed",   # Ollama local no valida la key
        "gemini": settings.gemini_api_key,  # Gemini sí necesita la real
    }
```

### El mapping backend → tipo de cliente

```python
_BACKEND_CLIENT: dict[str, str] = {
    "flm":     "openai_compat",    # FastFlowLM habla el protocolo OpenAI
    "ollama":  "ollama_native",    # Ollama usa su cliente nativo
    "gemini":  "openai_compat",    # Gemini tiene endpoint compatible con OpenAI
}
```

### Las fábricas de clientes

```python
_CLIENT_FACTORIES: dict[str, Callable[[ProviderConfig], LLMClient]] = {
    "openai_compat": lambda config: OpenAICompatClient(
        base_url=config.base_url, api_key=config.api_key
    ),
    "ollama_native": lambda config: OllamaNativeClient(host=config.base_url),
}
```

Agregar un nuevo tipo de cliente (ej. un cliente nativo de Claude) requiere: crear el archivo en `src/nlp/llm/backends/`, agregar una entrada aquí, y agregar la entrada en `_BACKEND_CLIENT`.

### Construcción de la tabla por role

```python
def _build_provider_table() -> dict[str, ProviderConfig]:
    return {role.value: _provider_config_for_role(role) for role in LLMRole}

def _provider_config_for_role(role: LLMRole) -> ProviderConfig:
    backend, model = settings.role_spec(role)
    # ... resuelve base_url, api_key, client del backend ...
    return ProviderConfig(
        name=role.value,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        client=client,
        capabilities=backend,
    )
```

`settings.role_spec(role)` lee el formato `"backend,model"` de `.env.providers`:

```env
LLM_ROL_GENERATE=flm,qwen3.5:9b
LLM_ROL_REFORMULATE=gemini,gemini-2.0-flash
LLM_ROL_REVIEW=gemini,gemini-2.0-flash
LLM_ROL_SUPPLEMENT=flm,qwen3.5:2b
```

### El cliente cacheado

```python
@cache
def _client_for(config: ProviderConfig) -> LLMClient:
    factory = _CLIENT_FACTORIES.get(config.client)
    if factory is None:
        raise ValueError(f"Unknown LLM client type: {config.client!r} ...")
    return factory(config)
```

`functools.cache` garantiza un único `LLMClient` por `(backend, base_url, api_key, model, client)`. El campo `name` no participa en la cache key (`compare=False` en el dataclass): dos roles con el mismo backend+modelo comparten una sola conexión.

### La interfaz pública

```python
def get_provider(name: str) -> ProviderConfig:
    """ Retorna la configuración para un role (ej. "generate"). """
    table = _build_provider_table()
    return table[name]

def get_client(name: str) -> tuple[LLMClient, ProviderConfig]:
    """ Retorna (client, config) listo para usar con LLMClient.complete(). """
    config = get_provider(name)
    return _client_for(config), config
```

Los nodos del grafo no importan `providers.py` directamente — acceden a él a través de `ask_llm()` y `ask_llm_internal()` en `generate.py`. Esto mantiene a `nodes.py` desacoplado de los detalles del transporte.

---

## Routing de role por nodo

Cada nodo pasa el role explícitamente al LLM:

```python
# reformulate_node — usa el role REFORMULATE
ask_llm_internal(
    system_prompt=build_reformulation_system_prompt(),
    prompt=reformulation_prompt,
    provider=LLMRole.REFORMULATE.value,
)

# generate_node — usa el role GENERATE
ask_llm(
    prompt=prompt,
    chat_memory=state["chat_memory"],
    provider=LLMRole.GENERATE.value,
    ...
)

# review_node — usa el role REVIEW
ask_llm_internal(
    system_prompt=build_review_system_prompt(),
    prompt=review_prompt,
    provider=LLMRole.REVIEW.value,
)

# correct_node — usa el role GENERATE (misma generación que el nodo principal)
ask_llm(
    prompt=correction_prompt,
    chat_memory=state["chat_memory"],
    provider=LLMRole.GENERATE.value,
    ...
)
```

**Por qué el role GENERATE siempre genera la respuesta final:** Los chunks de tus documentos indexados viajan en el prompt. Si ese nodo usara un proveedor externo, esos chunks — potencialmente privados — saldrían de tu máquina. El proveedor asignado a GENERATE garantiza que tus colecciones nunca se envían a un servicio externo (a menos que configures `LLM_ROL_GENERATE=gemini,...` deliberadamente).

---

## `generate.py` — la capa de orquestación

`generate.py` es el único módulo que importa `providers.py`. Construye los mensajes, valida overrides, y delega al cliente:

```python
def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None = None,
    think_mode: bool | None = None,
    extra: ExtraFields | None = None,
    system_prompt: str | None = None,
) -> str:
    messages = build_messages(
        prompt=prompt,
        chat_memory=chat_memory,
        system_prompt=system_prompt,
    )
    content = _complete(
        messages=messages,
        provider_name=provider,
        max_tokens=max_tokens,
        think_mode=think_mode,
        extra=extra,
    )
    return content if content is not None else "Model did not return a response."
```

`ask_llm_internal()` es la variante para operaciones internas (reformulación, review): no tiene chat_memory, y si falla devuelve `None` en vez de un mensaje de error al usuario.

```python
def ask_llm_internal(
    system_prompt: str,
    prompt: str,
    provider: str,
    max_tokens: int | None = None,
) -> str | None:
    messages = build_messages(prompt=prompt, chat_memory=[], system_prompt=system_prompt)
    content = _complete(messages=messages, provider_name=provider, ...)
    return content
```

---

## El ciclo review → correct

Este es el mecanismo de auto-corrección. El objetivo es reducir alucinaciones y forzar citación de fuentes sin que el usuario tenga que hacer seguimiento manual.

### Qué evalúa `review_node`

1. **Anclaje (Grounding):** ¿cada afirmación de la respuesta aparece en los chunks recuperados? Si el modelo dice algo que no está en el contexto, es una alucinación.
2. **Citas:** ¿la respuesta menciona de qué fuente viene la información? Un RAG que no cita no es útil para trabajo académico o de investigación.
3. **Voz (Voice):** ¿expone el mecanismo de recuperación? Frases como "según el contexto proporcionado" violan la naturalidad.

### El protocolo JSON

El reviewer responde con JSON puro:

```json
{"passed": true}
{"passed": false, "reason": "grounding", "feedback": "La respuesta afirma que Freud publicó 1984, pero eso no aparece en el contexto."}
```

### `MAX_REVIEW_ATTEMPTS`

```python
MAX_REVIEW_ATTEMPTS = settings.max_review_attempts  # default: 1
```

Si el reviewer rechaza y `correct_node` corrige, el ciclo vuelve a `review_node`. Si esta segunda revisión también falla, el cap fuerza `review_passed=True` — el loop se rompe y la respuesta llega al usuario. El valor `1` significa: máximo un reintento.

---

## Configuración en `.env.providers`

```env
# Backends (URLs)
LLM_FLM_URL=http://127.0.0.1:8080/v1
LLM_OLLAMA_URL=http://127.0.0.1:11434
LLM_GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_API_KEY=AIza...

# Roles (backend,model)
LLM_ROL_GENERATE=flm,qwen3.5:9b
LLM_ROL_REFORMULATE=gemini,gemini-2.0-flash
LLM_ROL_REVIEW=gemini,gemini-2.0-flash
LLM_ROL_SUPPLEMENT=flm,qwen3.5:2b
```

El formato `backend,model` es parseado por `_split_backend_model()` en `settings.py`. El split es solo en la primera coma — el nombre del modelo puede contener `:` legítimamente (tags de Ollama/FastFlowLM como `qwen3.5:9b`).

---

## Cómo agregar un nuevo modelo a un backend existente

Solo toca un archivo: `src/config/models/<backend>.py`. Agregar una entrada al diccionario `_MODELS` con el nombre, temperatura, y capacidades. Ningún otro archivo se modifica.

---

## Cómo agregar un nuevo backend

1. Agregar la URL en `settings.py` (`llm_<nuevo>_url`)
2. Agregar la entrada en `_backend_url_table()` y `_backend_api_key_table()` en `providers.py`
3. Agregar la entrada en `_BACKEND_CLIENT` (que tipo de cliente usa)
4. Si el tipo de cliente es nuevo, crear el archivo en `src/nlp/llm/backends/` y agregar la fábrica en `_CLIENT_FACTORIES`
5. Crear `src/config/models/<nuevo_backend>.py` con las capacidades de los modelos
6. Configurar `LLM_ROL_*=<nuevo_backend>,<modelo>` en `.env.providers`

Cero cambios en `graph.py`, `nodes.py`, ni `generate.py`.

---

## Talking points para entrevista

**"¿Cómo manejas múltiples proveedores LLM?"**

> Arquitectura provider-per-role: cada paso del pipeline (generate, reformulate, review, web_supplement) tiene su propio role configurado independientemente en `.env.providers`. El routing de role a backend+modelo se resuelve en `settings.role_spec()`, y los clientes se cachean por `(backend, url, key, model)` — no por role.

**"¿Cómo evitas que los documentos privados salgan de tu máquina?"**

> El proveedor asignado a GENERATE es el único que recibe los chunks de los documentos indexados en su prompt. Los otros roles solo reciben la pregunta del usuario (reformulación) o la respuesta generada (review) — nunca los chunks crudos.

**"¿Cómo manejas la posibilidad de que el reviewer falle?"**

> Fallo silencioso explícito: si el JSON no se puede parsear o el proveedor devuelve un error, se aprueba por defecto. Un loop cap (`MAX_REVIEW_ATTEMPTS`) evita loops infinitos. El log registra cada evento de fallback.
