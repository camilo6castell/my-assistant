# Módulo 7 — El Grafo LangGraph

Este módulo documenta el grafo de estados con LangGraph: la pieza central del sistema RAG. Si entiendes este módulo, puedes explicar la arquitectura completa en tres minutos.

---

## El concepto central: grafo de estados

Un grafo de estados tiene tres componentes:

**Estado** — un TypedDict que representa toda la información del sistema en un momento dado. Cada nodo recibe el estado completo y devuelve solo los campos que modifica. LangGraph hace el merge.

**Nodos** — funciones que reciben el estado y producen un delta. No saben nada del grafo — no saben qué nodo viene después, no saben si se van a volver a ejecutar. Solo reciben estado y devuelven estado.

**Edges** — conexiones entre nodos. Los edges normales son fijos (`A → B`). Los edges condicionales ejecutan una función de routing que lee el estado y devuelve qué nodo viene a continuación.

```python
# Edge normal: retrieve siempre va a evaluate
graph.add_edge(_RETRIEVE, _EVALUATE)

# Edge condicional: evaluate puede ir a generate o a reformulate
graph.add_conditional_edges(
    _EVALUATE,
    route_after_evaluate,  # función que lee el estado
    {_REFORMULATE: _REFORMULATE, _GENERATE: _GENERATE},  # mapa de opciones
)
```

La lógica de control vive en los edges, no en los nodos. Los nodos son funciones puras.

---

## `state.py` — RAGState

El estado compartido entre todos los nodos. Tiene 15 campos:

```python
class RAGState(TypedDict):
    # Datos de entrada del usuario
    question: str  # pregunta actual (puede ser reformulada)
    mode: str  # ChatMode.SOFT | ChatMode.HARD
    collections: list[LoadedCollection]  # colecciones FAISS activas
    chat_memory: list[TurnMemory]  # historial de la conversación

    # Resultado de la recuperación
    results: list[SearchResult]  # chunks recuperados por retrieve_node
    confidence: float  # score promedio de top-K
    reformulated: bool  # True si ya se reformuló (evita loops)

    # Resultado de la generación
    answer: str  # respuesta generada

    # Ciclo de revisión
    review_passed: bool  # True si el reviewer aprobó
    review_feedback: str  # motivo de rechazo del reviewer
    review_attempts: int  # intentos de corrección (evita loops)

    # Overrides de generación (None = behavior por defecto)
    max_tokens: int | None  # límite de tokens de salida
    think_mode: bool | None  # modo de razonamiento (si el modelo lo soporta)
    extra: dict[str, Any] | None  # parámetros específicos del backend

    # Adjuntos
    attachments: list[tuple[str, str]]  # archivos adjuntos (nombre, contenido)
```

### `RAGStateUpdate` — el tipo de retorno de los nodos

Cada nodo retorna un `RAGStateUpdate`, que es un TypedDict con `total=False` — todos los campos son opcionales. LangGraph toma solo los campos presentes en el dict de retorno y los mergea con el estado existente:

```python
class RAGStateUpdate(TypedDict, total=False):
    question: str
    mode: str
    collections: list[LoadedCollection]
    chat_memory: list[TurnMemory]
    results: list[SearchResult]
    confidence: float
    reformulated: bool
    answer: str
    review_passed: bool
    review_feedback: str
    review_attempts: int
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None
    attachments: list[tuple[str, str]]
```

Esto reemplaza al genérico `dict[str, object]` anterior: cada campo tiene el mismo tipo que en `RAGState`, lo que permite type-checking estático sin perder la flexibilidad de que cada nodo solo escriba lo que necesita.

### Por qué los campos `max_tokens`, `think_mode` y `extra` viven aquí

Son overrides de generación que viajan desde el endpoint HTTP hasta los nodos del grafo. `None` significa "usar el default de settings/.env". Solo `generate_node` y `correct_node` los leen — `reformulate_node` y `review_node` siempre usan la configuración por defecto porque son tareas internas de un solo paso.

**Importante:** `temperature` NO vive aquí. Es una propiedad fija de cada modelo, definida en `src/config/models/<backend>.py`, y nunca se sobreescribe por request.

### Por qué `attachments` no viaja a `retrieve_node`

`retrieve_node` usa `state["question"]` para la búsqueda semántica. Los adjuntos son contenido de archivos que no deberían contaminar el embedding de búsqueda. Solo `generate_node` los inyecta en el prompt final usando `inject_attachments()`.

**Nota de implementación:** este módulo NO usa `from __future__ import annotations`. LangGraph llama a `get_type_hints(RAGState)` en tiempo de ejecución para inspeccionar los campos del estado. Con anotaciones postergadas, todos los tipos se vuelven strings lazy y `get_type_hints()` falla al resolver `LoadedCollection`.

---

## `nodes.py` — los seis nodos

### Constantes de nombres

Cada nodo tiene un nombre constante que se usa tanto al registrarse en el grafo como en los edges condicionales:

```python
_RETRIEVE = "retrieve"
_EVALUATE = "evaluate"
_REFORMULATE = "reformulate"
_GENERATE = "generate"
_REVIEW = "review"
_CORRECT = "correct"
```

---

### `retrieve_node`

```python
def retrieve_node(state: RAGState) -> RAGStateUpdate:
    results, confidence = search(
        question=state["question"],
        mode=state["mode"],
        collections=state["collections"],
    )
    return {"results": results, "confidence": confidence}
```

Llama a `search()` del Módulo 4. Devuelve solo `results` y `confidence` — LangGraph mergea con el estado existente. `state["question"]` puede ser diferente a la pregunta original si `reformulate_node` ya actuó — el grafo pasó por este nodo dos veces.

---

### `evaluate_node`

```python
def evaluate_node(state: RAGState) -> RAGStateUpdate:
    confidence = state["confidence"]
    reformulated = state["reformulated"]
    limit = settings.confidence_limit
    # ... solo loggea la decisión
    return {}
```

El único nodo que devuelve `{}`. Su único rol es loggear — la decisión real la toma `route_after_evaluate` en el edge. Esto es una decisión de diseño: mantiene la lógica de routing en el edge (donde pertenece según LangGraph) en vez de en el nodo.

---

### `reformulate_node`

```python
def reformulate_node(state: RAGState) -> RAGStateUpdate:
    reformulation_prompt = (
        f"A semantic search over [{collection_names}] returned results "
        f"with low relevance for this question:\n\n"
        f'"{original}"\n\n'
        f"Rewrite the question to maximize semantic similarity ..."
    )

    reformulated_question = ask_llm_internal(
        system_prompt=build_reformulation_system_prompt(),
        prompt=reformulation_prompt,
        provider=LLMRole.REFORMULATE.value,
    )

    return {
        "question": reformulated_question if reformulated_question else original,
        "reformulated": True,
    }
```

Sobrescribe `question` con la versión reformulada por el LLM del rol REFORMULATE. Marca `reformulated=True` para que `route_after_evaluate` no vuelva a reformular — sin esto habría un loop infinito entre reformulate → retrieve → evaluate → reformulate.

Usa `ask_llm_internal()` en vez de `ask_llm()`: no tiene chat_memory, es una operación interna del grafo, no un turno del usuario. Le pasa un system prompt específico de reformulación (`build_reformulation_system_prompt()`).

Después de este nodo, el edge fijo lleva de vuelta a `retrieve_node`, que ahora busca con la pregunta reformulada.

---

### `generate_node`

```python
def generate_node(state: RAGState) -> RAGStateUpdate:
    results = state["results"]
    if not results:
        return {"answer": "No relevant context was found for your question."}

    context_chunks = format_context_chunks(results)
    prompt = build_prompt(
        context_chunks=context_chunks,
        question=inject_attachments(state["question"], state["attachments"]),
        mode=state["mode"],
    )

    # Guard: verifica que el prompt no exceda el contexto del modelo
    check_context_fit(
        system_prompt=build_system_prompt(),
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
    )

    # Genera la respuesta
    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
    )

    return {"answer": answer}
```

Este nodo hace tres cosas:

1. **Inyecta adjuntos** en la pregunta via `inject_attachments()` — los archivos adjuntos viajan como contexto adicional, no como colección.
2. **Valida el contexto** con `check_context_fit()` antes de llamar al LLM — si el prompt excede el límite del modelo, lanza `ContextLimitExceeded` que el endpoint traduce a 413.
3. **Genera la respuesta** con los overrides `max_tokens`, `think_mode` y `extra` del estado.

---

### `review_node`

```python
def review_node(state: RAGState) -> RAGStateUpdate:
    attempts = state.get("review_attempts", 0)

    if attempts >= MAX_REVIEW_ATTEMPTS:
        return {"review_passed": True, "review_feedback": ""}

    # ... construye review_prompt ...

    raw = ask_llm_internal(
        system_prompt=build_review_system_prompt(),
        prompt=review_prompt,
        provider=LLMRole.REVIEW.value,
    )

    # Parseo defensivo de JSON
    try:
        clean = temp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        passed = bool(result.get("passed", True))
        feedback = str(result.get("feedback", ""))
    except (json.JSONDecodeError, AttributeError):
        passed = True  # fallback: aprueba si no puede parsear
        feedback = ""

    return {
        "review_passed": passed,
        "review_feedback": feedback,
        "review_attempts": attempts + 1,
    }
```

Primero verifica el cap de intentos (`MAX_REVIEW_ATTEMPTS`). Si ya se revisó esa cantidad de veces, aprueba por defecto — evita loops infinitos independientemente del resultado del reviewer.

Si no, llama al LLM del rol REVIEW con `ask_llm_internal()` y un system prompt específico (`build_review_system_prompt()`). El reviewer evalúa dos cosas: anclaje (¿cada afirmación aparece en el contexto?) y citas (¿cada claim cita su fuente?).

**Fallo silencioso explícito:** Si el JSON viene malformado (timeout, respuesta libre del modelo), `passed` se pone en `True` — mejor entregar la respuesta sin revisar que bloquear al usuario. El log deja evidencia del evento.

**Limpieza defensiva del JSON:** Algunos modelos envuelven JSON en bloques de código Markdown (` ```json ... ``` `) aunque se les pida que no. La limpieza con `removeprefix`/`removesuffix` resuelve esto antes del parseo.

---

### `correct_node`

```python
def correct_node(state: RAGState) -> RAGStateUpdate:
    correction_prompt = build_correction_prompt(
        context_chunks=context_chunks,
        question=state["question"],
        previous_answer=state["answer"],
        feedback=state["review_feedback"],
        mode=state["mode"],
    )

    corrected = ask_llm(
        prompt=correction_prompt,
        chat_memory=state["chat_memory"],
        provider=LLMRole.GENERATE.value,
        max_tokens=state.get("max_tokens"),
        think_mode=state.get("think_mode"),
        extra=state.get("extra"),
    )

    return {"answer": corrected, "review_passed": False}
```

Regenera la respuesta incorporando el feedback del reviewer. Es un nodo separado de `generate_node` (en vez de reusarlo con un flag) para que el grafo sea legible: generate produce, correct repara. Cada nodo tiene una única responsabilidad.

Devuelve `review_passed=False` para que el edge lleve de vuelta a `review_node` — el ciclo review → correct puede ocurrir hasta `MAX_REVIEW_ATTEMPTS` veces.

---

## `graph.py` — el wiring

```python
def build_rag_graph():
    graph = StateGraph(RAGState)

    # Nodos
    graph.add_node(_RETRIEVE, retrieve_node)
    graph.add_node(_EVALUATE, evaluate_node)
    graph.add_node(_REFORMULATE, reformulate_node)
    graph.add_node(_GENERATE, generate_node)
    graph.add_node(_REVIEW, review_node)
    graph.add_node(_CORRECT, correct_node)

    # Edges fijos
    graph.set_entry_point(_RETRIEVE)
    graph.add_edge(_RETRIEVE, _EVALUATE)
    graph.add_edge(_REFORMULATE, _RETRIEVE)  # reformulate → retrieve (segunda pasada)
    graph.add_edge(_GENERATE, _REVIEW)
    graph.add_edge(_CORRECT, _REVIEW)  # correct → review (loop)

    # Edges condicionales
    graph.add_conditional_edges(
        _EVALUATE,
        route_after_evaluate,
        {_REFORMULATE: _REFORMULATE, _GENERATE: _GENERATE},
    )
    graph.add_conditional_edges(
        _REVIEW,
        route_after_review,
        {"end": END, "correct": _CORRECT},
    )

    return cast(Any, graph.compile())
```

### `route_after_evaluate`

```python
def route_after_evaluate(state: RAGState) -> Hashable:
    confidence = state["confidence"]
    limit = settings.confidence_limit

    if state["reformulated"] or confidence >= limit:
        return _GENERATE
    return _REFORMULATE
```

La condición `state["reformulated"]` es el guard anti-loop. Sin ella: confidence baja → reformulate → retrieve → evaluate → confidence baja otra vez → reformulate → loop infinito.

### `route_after_review`

`route_after_review` es una función definida en `nodes.py` (no inline en `graph.py`):

```python
def route_after_review(state: RAGState) -> str:
    if state.get("review_passed", True):
        return "end"
    return "correct"
```

Simple: si pasó la revisión (o si el cap de intentos forzó `review_passed=True`), termina. Si no, corrige.

---

## El grafo completo visualizado

```
                     ┌─────────┐
        ──────────── │ retrieve│◄──────────────────┐
       │             └────┬────┘                   │
       │                  │                        │
       │             ┌────▼────┐                   │
       │             │evaluate │ (solo loggea)      │
       │             └────┬────┘                   │
       │                  │                        │
       │      ┌───────────┴───────────┐            │
       │  conf ok / ya reformulado   conf baja     │
       │      │                       │            │
       │ ┌────▼──────┐        ┌───────▼─────┐      │
       │ │ generate  │        │ reformulate │──────┘
       │ └────┬──────┘        └─────────────┘
       │      │
       │ ┌────▼──────┐
       │ │  review   │◄─────────────────┐
       │ └────┬──────┘                  │
       │      │                         │
       │ ┌────┴──────┐                  │
       │ passed   rechazado             │
       │ │              │               │
       │ END     ┌──────▼──────┐        │
       │         │   correct   │────────┘
       │         └─────────────┘
       │             (loop acotado por MAX_REVIEW_ATTEMPTS)
       │
       └─ (el edge REFORMULATE → RETRIEVE cierra este loop)
```

---

## Por qué LangGraph y no un `if/else` en Python

La pregunta válida es: ¿no se podía hacer esto con un `while` y unos `if`? Sí. Pero hay tres razones para preferir el grafo:

**Observabilidad.** LangGraph puede registrar cada transición de estado. Cada vez que el grafo toma una decisión (`route_after_evaluate` devuelve `_REFORMULATE`), esa decisión es un evento discreto que se puede loggear, trazar, y reproducir. Con `if/else`, la lógica de routing está entremezclada con la de procesamiento.

**Separación de responsabilidades.** Los nodos no saben nada del flujo — solo transforman estado. La lógica de control vive exclusivamente en los edges. Si cambias las condiciones de routing (por ejemplo, agregar un tercer intento de reformulación), tocas solo el edge, no los nodos.

**Extensibilidad.** Agregar un nodo nuevo es `graph.add_node()` + conectar los edges. Sin LangGraph, habría que reescribir el flujo de control del handler.

Dicho esto: LangGraph tiene costo de complejidad. Para un pipeline de 2 pasos sin condicionales, un `if/else` es más claro. El grafo justifica su existencia cuando hay más de un edge condicional — que es exactamente el caso aquí.
