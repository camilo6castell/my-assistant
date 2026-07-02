# Módulo 6 — Sesión, Modos y CLI *(actualizado)*

Los cambios respecto a la versión anterior son tres: los modos se renombraron (RIGUROSO→HARD, INTERPRETATIVO→SOFT), `ChatSession` tiene un nuevo atributo `agent_active`, e `interface.py` ahora tiene dos handlers de preguntas — el pipeline lineal original y el nuevo via grafo LangGraph.

---

## `modes.py` — HARD y SOFT

```python
class ChatMode:
    SOFT: str = "SOFT"
    HARD: str = "HARD"
```

El renombre no es cosmético — refleja mejor qué hace cada modo:

- **HARD** = restricciones duras sobre el uso del contexto (cita o di que no está). Era RIGUROSO.
- **SOFT** = el modelo puede interpretar, conectar y sintetizar. Era INTERPRETATIVO.

Los valores son strings porque viajan hasta el prompt como texto (`build_rules_block(mode)` los compara con `ChatMode.HARD`). Un `Enum` real requeriría `.value` para extraer el string — solución más pragmática para un sistema donde el modo es tanto un identificador de código como texto legible.

---

## `ChatSession` — el estado completo de la sesión

```python
class ChatSession:
    def __init__(self) -> None:
        self.context_manager: ContextManager = ContextManager()
        self.soft_mode: bool = True        # True = SOFT, False = HARD
        self.agent_active: bool = True     # True = usa grafo LangGraph
        self.chat_memory: list[TurnMemory] = []
```

Cuatro atributos, cuatro responsabilidades:

```
context_manager  → qué colecciones están cargadas (Módulo 3)
soft_mode        → en qué modo de respuesta estamos
agent_active     → si las preguntas pasan por el grafo o por el pipeline lineal
chat_memory      → historial de turnos de esta sesión
```

`agent_active` es el único atributo nuevo respecto a la versión anterior. Controla cuál de los dos handlers de preguntas usa `interface.py`. Por defecto está en `True` — el grafo es el camino principal.

**El toggle de modo:**

```python
@property
def mode(self) -> str:
    return SOFT if self.soft_mode else HARD

def toggle_mode(self) -> str:
    self.soft_mode = not self.soft_mode
    return self.mode
```

`@property` hace que `session.mode` se lea como atributo pero se compute en el momento — nunca puede haber incoherencia entre `soft_mode` y `mode`.

**El header del input:**

```python
def get_prompt_header(self) -> str:
    # Ejemplo: [ CONTEXT: freud, debord | MODE: SOFT | AGENT: ON ]
    #          >
```

Muestra en cada línea de input: contextos activos (solo el basename, no el namespace completo), modo actual, y si el agente está encendido.

---

## `interface.py` — dos handlers, un dispatcher

El loop principal lee comandos y despacha:

```python
while True:
    command = input(session.get_prompt_header()).strip()

    if command == "/exit":      break
    if command == "/help":      ...
    if command == "/context":   _handle_context(...)
    if command == "/remove":    _handle_remove(...)
    if command == "/list":      _print_available(...)
    if command == "/active":    _print_active(...)
    if command == "/clear":     ...
    if command == "/reset":     session.reset_memory()
    if command == "/mode":      session.toggle_mode()
    if command == "/agent":     session.toggle_agent()  # ← nuevo
    if command.startswith("/"): # comando desconocido → error
    
    # cualquier texto sin "/" → es una pregunta
    if session.agent_active:
        _handle_agent_question(session, command)
    else:
        _handle_question(session, command)
```

Las últimas dos líneas son el dispatch real entre los dos caminos de ejecución.

---

## `_handle_question()` — el pipeline lineal (sin grafo)

```python
def _handle_question(session: ChatSession, question: str) -> None:
    results, confidence = search(
        question=question,
        mode=session.mode,
        collections=collections,
    )

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
        # chat_memory ya no se pasa aquí — viaja en ask_llm → build_messages
    )

    answer = ask_llm(prompt=prompt, chat_memory=session.chat_memory)
    session.add_to_memory(user=question, assistant=answer)
```

La secuencia es: `search()` → `build_prompt()` → `ask_llm()`. Lineal, sin condicionales de routing, sin reformulación automática, sin reviewer. Disponible para cuando quieres una respuesta rápida sin la latencia extra del grafo.

---

## `_handle_agent_question()` — el pipeline con grafo LangGraph

```python
def _handle_agent_question(session: ChatSession, question: str) -> None:
    graph = build_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "mode": session.mode,
        "collections": collections,
        "chat_memory": session.chat_memory,
        "results": [],
        "confidence": 0.0,
        "reformulated": False,
        "answer": "",
        "review_passed": False,
        "review_feedback": "",
        "review_attempts": 0,
    }

    final_state = graph.invoke(initial_state)

    answer = final_state["answer"]
    confidence = final_state["confidence"]
    reformulated = final_state["reformulated"]
```

En lugar de llamar directamente a `search()` y `ask_llm()`, construye el estado inicial y le pasa el control al grafo. El grafo devuelve el estado final con `answer` ya populado.

Si `reformulated=True`, el grafo detectó baja confianza en los resultados iniciales, reformuló la query con Gemini, y buscó de nuevo. El CLI lo muestra al usuario:

```
  [agent] Low confidence in initial search → query reformulated automatically.
```

El estado inicial siempre incluye los campos de review en cero (`review_passed=False`, `review_feedback=""`, `review_attempts=0`). El grafo los actualiza a medida que avanza.

---

## Comparación de los dos caminos

| | Pipeline lineal (`/agent OFF`) | Grafo LangGraph (`/agent ON`) |
|---|---|---|
| Reformulación automática | No | Sí, si confidence < límite |
| Review de alucinaciones | No | Sí, via Gemini |
| Corrección con feedback | No | Sí, modelo local corrige |
| Llamadas al LLM | 1 (generate) | 1-3 (generate + review + correct) |
| Latencia | Menor | Mayor |
| Ideal para | Iteración rápida, debug | Respuestas de producción |

---

## La visión completa de la sesión

```
main() instancia ChatSession()
         │
         ▼
   start_chat(session)
         │
         ▼
   loop: input(header)
         │
    ┌────┴──────────────────────────┐
    │  comandos                     │  preguntas (sin "/")
    │  /context, /mode, etc.        │
    └────────────────────────────── │
                                    │
                         ┌──────────┴──────────┐
                    agent OFF             agent ON
                         │                     │
              _handle_question()    _handle_agent_question()
                         │                     │
               search()            graph.invoke(initial_state)
               build_prompt()           │
               ask_llm()          (Módulos 7 y 8)
                         │                     │
                    answer: str           answer: str
                         │                     │
                    session.add_to_memory()
                    (actualiza chat_memory para el próximo turno)
```
