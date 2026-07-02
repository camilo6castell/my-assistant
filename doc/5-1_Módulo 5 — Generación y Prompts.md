# Módulo 5 — Generación y Prompts *(actualizado)*

Este módulo tiene tres cambios respecto a la versión anterior que vale la pena entender antes de leer el código:

1. **El historial ya no va en el texto del prompt.** Antes `build_prompt()` recibía `chat_memory` y lo embebía como texto dentro del string. Ahora el historial viaja como mensajes de API en `build_messages()`, que es la forma nativa de la interfaz `chat/completions`. El prompt resultante es más corto y no tiene el historial duplicado.

2. **`builder.py` exporta tres funciones en vez de una.** `build_prompt` sigue siendo la principal. `build_review_prompt` y `build_correction_prompt` son las nuevas, usadas por `review_node` y `correct_node` en el grafo.

3. **`client.py` fue eliminado.** Reemplazado por `providers.py`, que gestiona múltiples clientes LLM en vez de uno solo global. Ver Módulo 8.

---

## `builder.py` — tres funciones, tres momentos del grafo

### `build_prompt()` — para `generate_node`

```python
def build_prompt(
    context_chunks: list[str],
    question: str,
    mode: str,
) -> str:
```

El historial desapareció de la firma. La función ahora construye solo tres secciones: reglas del modo, contexto recuperado, y pregunta.

```
Eres un asistente RAG.
Tu tarea es responder usando EXCLUSIVAMENTE el contexto proporcionado.

{build_rules_block(mode)}

========================================
CONTEXTO
========================================

{build_context_block(context_chunks)}

========================================
PREGUNTA
========================================

{question}

========================================
RESPUESTA
========================================
```

El prompt termina con `RESPUESTA` y un salto de línea — señal visual para que el modelo sepa dónde empieza su output. Es una técnica estándar de prompt engineering.

`build_rules_block(mode)` varía según el modo:

```
HARD: "Usa únicamente el contenido del contexto. Si algo no está,
       dilo explícitamente. Cita la fuente y página."

SOFT: "Puedes conectar ideas, sintetizar, abstraer principios generales.
       Indica fuentes cuando sea posible."
```

El punto importante: **el modo no es un parámetro técnico del modelo** (no cambia temperatura ni el modelo en sí). Es texto en el prompt que le indica al LLM cómo comportarse. Combinado con el efecto en `search()` (más chunks, más queries en SOFT), el modo tiene dos palancas: qué llega al LLM y qué se le permite hacer con ello.

### `build_review_prompt()` — para `review_node`

```python
def build_review_prompt(
    context_chunks: list[str],
    question: str,
    answer: str,
) -> str:
```

Le entrega a Gemini los mismos chunks, la pregunta original, y la respuesta generada por el modelo local. Le pide que evalúe dos dimensiones:

- **Anclaje:** ¿cada afirmación está respaldada por el contexto o es una alucinación?
- **Citas:** ¿la respuesta menciona las fuentes cuando hace afirmaciones concretas?

El formato de respuesta solicitado es JSON estricto:

```json
{"passed": true}
{"passed": false, "feedback": "La respuesta afirma X pero eso no aparece en el contexto."}
```

JSON en vez de texto libre porque el parsing es determinístico. Gemini a veces envuelve el JSON en bloques de código Markdown aunque se le indique que no — `review_node` tiene una limpieza defensiva para eso.

### `build_correction_prompt()` — para `correct_node`

```python
def build_correction_prompt(
    context_chunks: list[str],
    question: str,
    previous_answer: str,
    feedback: str,
    mode: str,
) -> str:
```

Incluye la respuesta rechazada y el feedback del reviewer. Le dice al modelo local: "esto estaba mal por este motivo, corrígelo sin cambiar lo que estaba bien". Más eficiente que regenerar desde cero porque el modelo puede hacer un diff conceptual de lo que necesita corregir.

---

## `generate.py` — cómo llega el prompt al LLM

### `build_messages()`: el historial como mensajes de API

```python
def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
) -> list[ChatCompletionMessageParam]:

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in chat_memory:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": prompt})

    return messages
```

Esta es la estructura estándar de la API `chat/completions`:

```python
[
    {"role": "system",    "content": "Eres un asistente RAG..."},
    {"role": "user",      "content": "pregunta anterior"},       # turno 1
    {"role": "assistant", "content": "respuesta anterior"},      # turno 1
    {"role": "user",      "content": "<<< prompt actual >>>"},   # turno actual
]
```

El historial como mensajes de API es la forma nativa — el modelo los recibe como parte de la conversación, no como texto plano dentro del prompt del usuario. Antes estaban en los dos lugares (texto en el prompt + mensajes de API), lo cual era redundante. Ahora solo en los mensajes.

### `ask_llm()` y `ask_llm_internal()` — con `provider`

```python
def ask_llm(
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str | None = None,
) -> str:
    provider_name = provider or settings.generate_provider
    client, config = get_client(provider_name)

    response = client.chat.completions.create(
        model=config.model,
        messages=build_messages(prompt=prompt, chat_memory=chat_memory),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )
    ...
```

La firma ahora tiene `provider`. Si se pasa, usa ese proveedor. Si no, usa `settings.generate_provider` (por defecto `"local"`). El nodo del grafo lo pasa explícitamente:

```python
# en generate_node:
answer = ask_llm(prompt=prompt, chat_memory=..., provider=settings.generate_provider)

# en correct_node:
corrected = ask_llm(prompt=correction_prompt, chat_memory=..., provider=settings.generate_provider)
```

`ask_llm_internal()` es la misma idea pero para llamadas internas — reformulación y review — que no reciben `chat_memory` y usan `settings.reformulate_provider` por defecto (`"gemini"`). Si falla (503, timeout), devuelve el `prompt` original como fallback en lugar de levantar una excepción, para que el grafo continúe.

---

## El flujo completo de este módulo dentro del grafo

```
retrieve_node devuelve results: list[SearchResult]
        │
        ▼  (en generate_node)
context_chunks = [f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}" ...]
        │
        ▼  build_prompt(context_chunks, question, mode)
prompt: str
        │
        ▼  ask_llm(prompt, chat_memory, provider="local")
        │    └─ build_messages() → [{system}, {turns...}, {user: prompt}]
        │    └─ client.chat.completions.create()
        │         HTTP POST → http://127.0.0.1:52625/v1/chat/completions
        │
answer: str
        │
        ▼  review_node
        │    build_review_prompt(chunks, question, answer)
        │    ask_llm_internal(review_prompt, provider="gemini")
        │    parsea JSON → {"passed": bool, "feedback": str}
        │
   passed?
   ├─ sí → END
   └─ no → correct_node
              build_correction_prompt(chunks, question, answer, feedback, mode)
              ask_llm(correction_prompt, chat_memory, provider="local")
              → nueva answer → review_node (loop, max 1 vez)
```
