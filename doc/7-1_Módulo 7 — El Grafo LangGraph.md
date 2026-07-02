# Módulo 7 — El Grafo LangGraph *(nuevo)*

Este módulo introduce la pieza más importante del refactor: el reemplazo del pipeline lineal `search() → build_prompt() → ask_llm()` por un grafo de estados con LangGraph. Si entiendes este módulo, puedes explicar la arquitectura completa del sistema en una entrevista en tres minutos.

---

## El problema que resuelve

El pipeline lineal original era determinístico: siempre hacía los mismos pasos en el mismo orden. Eso es suficiente para un RAG simple. Pero tiene un límite: no puede reaccionar a la calidad de lo que recuperó.

Imagina que buscas sobre "el papel de la censura en los sueños según Freud" y los vectores más cercanos tienen un score de 0.61 — por debajo del umbral de confianza. El pipeline lineal usaría esos chunks mediocres de todas formas y generaría una respuesta pobre o con alucinaciones.

Lo que necesitas es: detectar ese bajo score y, antes de generar, intentar mejorar la query automáticamente. Eso requiere **una decisión en tiempo de ejecución sobre el flujo del programa** — y eso es exactamente para lo que está diseñado LangGraph.

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
    route_after_evaluate,                          # función que lee el estado
    {_REFORMULATE: _REFORMULATE, _GENERATE: _GENERATE},  # mapa de opciones
)
```

La lógica de control vive en los edges, no en los nodos. Los nodos son funciones puras.

---

## `state.py` — RAGState

```python
class RAGState(TypedDict):
    question:        str                   # pregunta actual (puede ser reformulada)
    mode:            str                   # ChatMode.SOFT | ChatMode.HARD
    collections:     list[LoadedCollection]
    chat_memory:     list[TurnMemory]
    results:         list[SearchResult]    # chunks recuperados
    confidence:      float                 # score promedio de top-K
    reformulated:    bool                  # True si ya se reformuló (evita loops)
    answer:          str                   # respuesta generada
    review_passed:   bool                  # True si el reviewer aprobó
    review_feedback: str                   # motivo de rechazo del reviewer
    review_attempts: int                   # intentos de corrección (evita loops)
```

El estado es el contrato entre todos los nodos. Un nodo que produce `answer` no sabe nada de `review_passed` — solo escribe su campo. Un nodo que lee `review_feedback` no sabe quién lo escribió. El grafo es la capa que conecta las escrituras con las lecturas.

**Por qué TypedDict y no dataclass:** LangGraph inspecciona el tipo del estado en tiempo de inicialización con `get_type_hints()`. TypedDict funciona de forma nativa con esa introspección. Los dataclasses también funcionan pero requieren anotaciones adicionales. TypedDict es la convención del ecosistema LangGraph.

---

## `nodes.py` — los seis nodos

### `retrieve_node`

```python
def retrieve_node(state: RAGState) -> dict[str, object]:
    results, confidence = search(
        question=state["question"],
        mode=state["mode"],
        collections=state["collections"],
    )
    return {"results": results, "confidence": confidence}
```

Llama a `search()` del Módulo 4. Devuelve solo `results` y `confidence` — LangGraph mergea con el estado existente. Nota que `state["question"]` puede ser diferente a la pregunta original si `reformulate_node` ya actuó — el grafo pasó por este nodo dos veces.

### `evaluate_node`

```python
def evaluate_node(state: RAGState) -> dict[str, object]:
    confidence = state["confidence"]
    limit = settings.confidence_limit
    if confidence >= limit:
        logger.info(f"confidence={confidence:.4f} >= limit={limit} → generando")
    else:
        logger.info(f"confidence={confidence:.4f} < limit={limit} → reformulando")
    return {}   # no modifica el estado
```

El único nodo que devuelve `{}`. Su único rol es loggear — la decisión real la toma `route_after_evaluate` en el edge. Esto es una decisión de diseño: mantiene la lógica de routing en el edge (donde pertenece según LangGraph) en vez de en el nodo.

### `reformulate_node`

```python
def reformulate_node(state: RAGState) -> dict[str, object]:
    reformulation_prompt = f"Reescribe esta pregunta para búsqueda vectorial: {state['question']}"
    reformulated_question = ask_llm_internal(
        prompt=reformulation_prompt,
        provider=settings.reformulate_provider,   # gemini
    )
    return {"question": reformulated_question, "reformulated": True}
```

Sobrescribe `question` con la versión reformulada por Gemini. Marca `reformulated=True` para que `route_after_evaluate` no vuelva a reformular en la segunda pasada — sin esto habría un loop infinito entre `reformulate → retrieve → evaluate → reformulate`.

Después de este nodo, el edge fijo lleva de vuelta a `retrieve_node`, que ahora busca con la pregunta reformulada.

### `generate_node`

```python
def generate_node(state: RAGState) -> dict[str, object]:
    context_chunks = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in state["results"]
    ]
    prompt = build_prompt(context_chunks, state["question"], state["mode"])
    answer = ask_llm(
        prompt=prompt,
        chat_memory=state["chat_memory"],
        provider=settings.generate_provider,   # local
    )
    return {"answer": answer}
```

El modelo local genera la respuesta. Siempre usa `generate_provider` — el proveedor local. Los documentos indexados nunca salen de la máquina.

### `review_node`

```python
def review_node(state: RAGState) -> dict[str, object]:
    attempts = state.get("review_attempts", 0)

    if attempts >= MAX_REVIEW_ATTEMPTS:
        return {"review_passed": True, "review_feedback": ""}
```

Primero verifica el cap de intentos. Si ya se revisó `MAX_REVIEW_ATTEMPTS` veces, aprueba por defecto — evita loops infinitos independientemente del resultado del reviewer.

Si no, llama a Gemini con `build_review_prompt()` y parsea el JSON de respuesta. Si el JSON viene malformado (o Gemini devuelve un 503), aprueba por defecto — falla silenciosa explícita, mejor entregar la respuesta sin revisar que bloquear al usuario.

```python
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        passed = bool(result.get("passed", True))
        feedback = str(result.get("feedback", ""))
    except json.JSONDecodeError:
        passed = True   # fallback: aprueba si no puede parsear
        feedback = ""

    return {
        "review_passed": passed,
        "review_feedback": feedback,
        "review_attempts": attempts + 1,
    }
```

### `correct_node`

```python
def correct_node(state: RAGState) -> dict[str, object]:
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
        provider=settings.generate_provider,   # local
    )
    return {"answer": corrected, "review_passed": False}
```

Regenera la respuesta con el feedback del reviewer. Devuelve `review_passed=False` para que el edge lleve de vuelta a `review_node` — el ciclo review → correct puede ocurrir hasta `MAX_REVIEW_ATTEMPTS` veces.

---

## `graph.py` — el wiring

```python
def build_rag_graph():
    graph = StateGraph(RAGState)

    # Nodos
    graph.add_node(_RETRIEVE,   retrieve_node)
    graph.add_node(_EVALUATE,   evaluate_node)
    graph.add_node(_REFORMULATE, reformulate_node)
    graph.add_node(_GENERATE,   generate_node)
    graph.add_node(_REVIEW,     review_node)
    graph.add_node(_CORRECT,    correct_node)

    # Edges fijos
    graph.set_entry_point(_RETRIEVE)
    graph.add_edge(_RETRIEVE,   _EVALUATE)
    graph.add_edge(_REFORMULATE, _RETRIEVE)   # reformulate → retrieve (segunda pasada)
    graph.add_edge(_GENERATE,   _REVIEW)
    graph.add_edge(_CORRECT,    _REVIEW)      # correct → review (loop)

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

    return graph.compile()
```

### `route_after_evaluate`

```python
def route_after_evaluate(state: RAGState) -> str:
    if state["reformulated"] or state["confidence"] >= settings.confidence_limit:
        return _GENERATE
    return _REFORMULATE
```

La condición `state["reformulated"]` es el guard anti-loop. Sin ella: confidence baja → reformulate → retrieve → evaluate → confidence baja otra vez → reformulate → loop infinito.

### `route_after_review`

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

**Extensibilidad.** Agregar un nodo nuevo (por ejemplo, un `classify_node` que decida si la pregunta es factual o conceptual antes de retrieve) es `graph.add_node()` + conectar los edges. Sin LangGraph, habría que reescribir el flujo de control del handler.

Dicho esto: LangGraph tiene costo de complejidad. Para un pipeline de 2 pasos sin condicionales, un `if/else` es más claro. El grafo justifica su existencia cuando hay más de un edge condicional — que es exactamente el caso aquí.
