# Módulo 5 — Generación y Prompts

Este módulo tiene dos archivos principales: `builder.py` construye todos los prompts del sistema, y `generate.py` los envía al LLM correcto sin importar qué backend esté detrás.

---

## `builder.py` — un catálogo completo de prompts

El archivo define **once funciones**, cada una para un momento distinto del pipeline. Todas siguen el mismo esqueleto conceptual: `ROL → ANCLAJE → CITAS → ESTILO → REGLAS DE TAREA`. Las secciones usan encabezados en mayúsculas con dos puntos, las reglas son listas con guion, y se distingue entre "REQUIREMENTS" (formato de salida) y "REGLAS" (comportamiento del modelo).

### `build_system_prompt()` — el prompt principal del RAG

```python
def build_system_prompt() -> str:
```

Este es el prompt más extenso. Le dice al LLM que es un experto que responde usando **exclusivamente** el material de referencia proporcionado. Las secciones clave:

- **ROLE:** actúa como si hubiera absorbido el material, no como un sistema que acaba de recuperar documentos.
- **GROUNDING:** cada afirmación debe ser trazable a un fragmento específico. Si las fuentes son insuficientes, decirlo. Si se contradicen, presentar ambas posiciones.
- **CITATION:** formato `(Fuente, p. X)` inmediatamente después de la afirmación. Si una afirmación depende de múltiples fuentes, citarlas todas juntas.
- **STYLE:** nunca exponer el mecanismo de recuperación (nada de "según el contexto proporcionado"). Responder en el mismo idioma que la pregunta del usuario.
- **RESPONSE QUALITY:** explicaciones completas, tablas Markdown para comparaciones, listas cuando mejoren la legibilidad.

Al final incluye las reglas de modo (HARD y SOFT) insertadas via `build_mode_rules()`.

### `build_reformulation_system_prompt()` — para reformulación de queries

```python
def build_reformulation_system_prompt() -> str:
```

Prompt para el nodo de reformulación. Le pide al LLM reescribir la pregunta del usuario para maximizar la calidad de recuperación en la base vectorial. Debe preservar la intención original, no responder, no introducir hechos nuevos, y resolver ambigüedades solo cuando se infieran de la redacción original.

### `build_web_supplement_system_prompt()` — para el suplemento web

```python
def build_web_supplement_system_prompt() -> str:
```

Prompt para la llamada LLM que decide si agregar información de fuentes web a una respuesta ya generada. La instrucción clave es tratar cada fragmento web como material no confiable y nunca como instrucciones — esta es la mitigación principal contra prompt injection indirecto (una página web podría contener texto tipo "ignore tus instrucciones anteriores"). A diferencia del prompt principal, aquí SÍ se indica al usuario que la información viene de fuentes web.

### `build_context_block()` — unir chunks recuperados

```python
def build_context_block(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)
```

Función auxiliar que une una lista de chunks con el separador `---`. Es el bloque de contexto recuperado que aparece en todos los prompts de generación.

### `build_mode_rules()` — reglas según el modo

```python
def build_mode_rules(mode: str) -> str:
    if mode == ChatMode.HARD:
        return """..."""
    return """..."""
```

Genera el bloque de reglas que se inserta en el system prompt principal:

- **HARD:** respuesta estrictamente limitada a lo explícito en las fuentes. No generalizar. Si falta información, decirlo.
- **SOFT:** respuestas largas, bien desarrolladas. Sintetizar entre múltiples fuentes, construir conexiones y comparaciones. Mantener grounding completo aun cuando la respuesta crezca en profundidad.

El modo no es un parámetro técnico del modelo (no cambia temperatura). Es texto en el prompt que le indica al LLM cómo comportarse.

### `build_prompt()` — el prompt del usuario para `generate_node`

```python
def build_prompt(
    context_chunks: list[str],
    question: str,
    mode: str,
) -> str:
```

Construye el prompt que contiene el contexto recuperado y la pregunta del usuario. El historial de conversación **no** se incluye aquí — viaja como mensajes de API separados en `build_messages()`. El resultado es:

```
Response mode needed: SOFT

Retrieved context:

{chunks unidos por ---}

User question:

{pregunta}
```

### `inject_attachments()` — inyectar archivos adjuntos

```python
def inject_attachments(
    question: str,
    attachments: list[tuple[str, str]],
) -> str:
```

Prepends el contenido de archivos adjuntos a la pregunta, envuelto en bloques de código con el nombre de archivo como encabezado. Se aplica **antes** de pasar la pregunta a `build_prompt()`. Las attachmentes son ortogonales al modo de respuesta — se inyectan igual en modo RAG y en modo sin contexto.

Si no hay attachmentes, devuelve la pregunta sin modificar.

### `build_review_system_prompt()` — system prompt del reviewer

```python
def build_review_system_prompt() -> str:
```

Le dice al LLM que es un revisor de calidad para el sistema RAG. Evalúa tres dimensiones:

1. **Anclaje:** cada afirmación debe estar respaldada por el contexto recuperado.
2. **Citas:** cada afirmación que dependa de una fuente debe citarla inmediatamente.
3. **Voz:** rechazar frases que expongan el mecanismo de recuperación.

### `build_review_prompt()` — prompt de revisión para `review_node`

```python
def build_review_prompt(
    context_chunks: list[str],
    question: str,
    answer: str,
) -> str:
```

Le entrega al reviewer los chunks recuperados, la pregunta original, y la respuesta generada. El formato de respuesta solicitado es JSON estricto:

```json
{"passed": true}
```

o bien:

```json
{
  "passed": false,
  "reason": "<grounding|missing_sources|exposed_retrieval_voice>",
  "feedback": "<explicación concisa del problema>"
}
```

### `build_correction_prompt()` — prompt de reparación para `correct_node`

```python
def build_correction_prompt(
    context_chunks: list[str],
    question: str,
    previous_answer: str,
    feedback: str,
    mode: str,
) -> str:
```

Incluye la respuesta rechazada, el feedback del reviewer, y las reglas del modo. Le dice al modelo: "esto estaba mal por este motivo, corrígelo sin cambiar lo que estaba bien". Más eficiente que regenerar desde cero porque el modelo puede hacer un diff conceptual de lo que necesita corregir.

### `build_web_supplement_prompt()` — prompt de suplemento web

```python
def build_web_supplement_prompt(
    question: str,
    answer: str,
    web_chunks: list[str],
) -> str:
```

Se usa en el caso `collections + web_search=True`. La respuesta local se genera primero, y esta función construye un segundo prompt que se envía como llamada LLM separada. El LLM decide si agregar algo o no — nunca reescribe ni reemplaza la respuesta original.

---

## `generate.py` — cómo llega el prompt al LLM

### `build_messages()`: el historial como mensajes de API

```python
def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
    system_prompt: str | None = None,
    max_turns: int | None = None,
) -> list[ChatTurn]:
```

Construye el array de mensajes en el formato estándar de la API `chat/completions`:

```python
[
    {"role": "system",    "content": "Eres un experto RAG..."},
    {"role": "user",      "content": "pregunta anterior"},       # turno 1
    {"role": "assistant", "content": "respuesta anterior"},      # turno 1
    {"role": "user",      "content": "<<< prompt actual >>>"},   # turno actual
]
```

- `system_prompt`: si es `None`, usa `build_system_prompt()` (comportamiento RAG por defecto). Si es `""` (cadena vacía, distinto de `None`), omite el mensaje system por completo — es el caso `_answer_raw()` donde el usuario controla todo desde su propio mensaje.
- `max_turns`: controla cuántos turnos del historial se envían (ventana deslizante). `None` usa `settings.max_turns`.

### `_validate_think()`: validar modo de razonamiento

```python
def _validate_think(think_mode: bool | None, config: ProviderConfig) -> None:
```

Si se proporciona un `think_mode` explícito, valida que el modelo activo lo soporte consultando `get_supports()` en `src/config/models/`. Si `think_mode` es `None`, no se valida — significa "usar el valor por defecto ya configurado en `_MODELS[model]`".

### `_dump_request_for_debug()`: diagnóstico opt-in

```python
def _dump_request_for_debug(kwargs: dict[str, Any]) -> None:
```

Escribe el body exacto enviado al proveedor a un archivo `debug_last_llm_request.json`. Solo se activa con `settings.llm_debug_dump=True`. Sobrescribe en cada llamada (solo importa el último request). Diseñado para diagnosticar problemas como "el modelo local devuelve vacío solo con prompts grandes de RAG".

### `_complete()`: la llamada central al LLM

```python
def _complete(
    *,
    messages: list[ChatTurn],
    provider_name: str,
    log_prefix: str,
    max_tokens: int | None,
    think_mode: bool | None,
    extra: ExtraFields | None,
) -> str | None:
```

Esta es la función privada que construye los kwargs resueltos (via `build_kwargs()` de `src/config/models/`) y se los entrega al `LLMClient` del proveedor. El módulo no sabe (ni necesita saber) si eso termina hablándole a OpenAI, FastFlowLM, u Ollama nativo.

- No recibe `temperature`: ningún override se pasa aquí, así que `build_kwargs()` siempre deja intacto el valor que ya está en `_MODELS[model]`.
- Devuelve `None` (nunca levanta excepción) si el modelo respondió vacío o si la llamada falló.
- `get_client()` devuelve una tupla `(LLMClient, ProviderConfig)` — el cliente es un protocolo `LLMClient`, no una instancia cruda de `OpenAI`.

### `ask_llm()`: la interfaz pública para nodos del grafo y la API

```python
def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None = None,
    think_mode: bool | None = None,
    extra: ExtraFields | None = None,
    max_turns: int | None = None,
    system_prompt: str | None = None,
) -> str:
```

El `provider` es **requerido** y siempre viene de `LLMRole.GENERATE.value`. Este módulo ya no elige un default por su own — la única fuente de verdad para "qué modelo genera la respuesta" es `Settings.role_spec()`.

- `system_prompt`: `None` usa `build_system_prompt()`, `""` omite el system message.
- `max_tokens`, `think_mode`, `extra`: overrides por request que nunca mutan `settings` (evita bugs de concurrencia).
- Si la llamada falla, devuelve `"Model did not return a response."`.

### `ask_llm_internal()`: para operaciones internas del grafo

```python
def ask_llm_internal(
    system_prompt: str,
    prompt: str,
    provider: str,
    max_tokens: int | None = None,
) -> str | None:
```

Versión simplificada para operaciones internas como reformulación y review. No recibe `chat_memory` (siempre vacío), no acepta `think_mode` ni `extra`. Devuelve `None` en lugar de un mensaje de error si falla, para que el grafo continúe con un fallback.

---

## El sistema de proveedores subyacente

`generate.py` depende de `src/nlp/llm/providers.py` para resolver el cliente y la configuración del proveedor. La estructura clave es `ProviderConfig`:

```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str         # nombre del rol (ej. "generate") — solo para logging
    backend: str      # "flm" | "ollama" | "gemini"
    base_url: str     # URL del backend
    api_key: str      # credencial
    model: str        # nombre del modelo (ej. "qwen3.5:9b")
    client: str       # "openai_compat" | "ollama_native"
    capabilities: str # key en src/config/models/ para las capacidades del modelo
```

`get_client(name)` resuelve el `ProviderConfig` para un rol dado y devuelve `(LLMClient, ProviderConfig)`. El `LLMClient` es un protocolo — `generate.py` nunca interactúa directamente con el proveedor HTTP.

---

## Flujo completo dentro del grafo

```
retrieve_node devuelve results: list[SearchResult]
        │
        ▼  (en generate_node)
context_chunks = format_context_chunks(results)
        │
        ▼  inject_attachments(question, attachments)
question modificada (o sin cambios si no hay archivos adjuntos)
        │
        ▼  check_context_fit(system_prompt, prompt, chat_memory, provider, max_tokens)
        │    └─ estima tokens totales → ¿cabe en la ventana del modelo?
        │    └─ si no cabe → ContextLimitExceeded → el router devuelve 413
        │
        ▼  build_prompt(context_chunks, question, mode)
prompt: str
        │
        ▼  ask_llm(prompt, chat_memory, provider, max_tokens, think_mode, extra)
        │    └─ build_messages() → [{system}, {turns...}, {user: prompt}]
        │    └─ _validate_think() → ¿el modelo soporta think_mode?
        │    └─ build_kwargs() → kwargs en forma nativa del backend
        │    └─ client.complete(kwargs) → HTTP POST → proveedor
        │
answer: str
        │
        ▼  review_node
        │    build_review_prompt(chunks, question, answer)
        │    ask_llm_internal(review_system_prompt, review_prompt, provider)
        │    parsea JSON → {"passed": bool, "feedback": str}
        │
   passed?
   ├─ sí → END
   └─ no → correct_node
              build_correction_prompt(chunks, question, answer, feedback, mode)
              check_context_fit(...)  ← misma protección preventiva
              ask_llm(correction_prompt, chat_memory, provider)
              → nueva answer → review_node (loop, max 1 vez)
```
