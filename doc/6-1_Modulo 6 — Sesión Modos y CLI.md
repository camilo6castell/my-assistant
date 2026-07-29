# Módulo 6 — Sesión, Modos y CLI

Este módulo gestiona la interacción del usuario con el sistema RAG desde la línea de comandos. Tiene tres archivos principales: `session.py` (estado de la sesión), `interface.py` (el loop de comandos), y los módulos de re-exportación `modes.py`/`types.py`.

---

## `ChatMode` y `TurnMemory` — los modelos de dominio

Los tipos fundamentales viven en `src/domain.models.py` y son importados desde allí por todos los demás módulos:

```python
class ChatMode(StrEnum):
    SOFT = "SOFT"
    HARD = "HARD"


class TurnMemory(BaseModel):
    model_config = ConfigDict(frozen=True)
    user: str
    assistant: str
```

`ChatMode` es un `StrEnum` — sus valores son strings que viajan directamente al prompt (los compara `build_mode_rules()` en `builder.py`). HARD restringe la respuesta a lo explícito en las fuentes; SOFT permite interpretar, conectar y sintetizar.

`TurnMemory` es un `BaseModel` congelado (`frozen=True`) — una vez registrado un turno, es inmutable. La lista `chat_memory` es append-only.

### Re-exportación desde `modes.py` y `types.py`

```python
# src/cli/modes.py
from src.domain.models import ChatMode

__all__ = ["ChatMode"]

# src/cli/types.py
from src.domain.models import TurnMemory

__all__ = ["TurnMemory"]
```

Estos archivos son re-exportaciones para compatibilidad. El código fuente canonical está en `src/domain.models`. Si estás importando estos tipos, preferí importar directamente desde `src.domain.models`.

---

## `ChatSession` — el estado completo de la sesión

```python
class ChatSession:
    def __init__(self) -> None:
        self.context_manager: ContextManager = ContextManager()
        self.soft_mode: bool = True
        self.agent_active: bool = True
        self.chat_memory: list[TurnMemory] = []
```

Cuatro atributos, cuatro responsabilidades:

| Atributo | Responsabilidad |
|---|---|
| `context_manager` | Qué colecciones están cargadas (Módulo 3) |
| `soft_mode` | En qué modo de respuesta estamos (`True` = SOFT, `False` = HARD) |
| `agent_active` | Si las preguntas pasan por el grafo LangGraph o por el pipeline lineal |
| `chat_memory` | Historial de turnos de esta sesión |

### Gestión de contextos

```python
def load_context(self, pattern: str) -> list[str]:
    return self.context_manager.activate(pattern)


def unload_context(self, pattern: str) -> list[str]:
    return self.context_manager.deactivate(pattern)


def clear_contexts(self) -> None:
    self.context_manager.clear()


def get_active_contexts(self) -> list[str]:
    return self.context_manager.get_active()


def get_loaded_collections(self) -> list[LoadedCollection]:
    return self.context_manager.get_loaded_collections()
```

Todos delegan en `ContextManager` (Módulo 3). El patrón es el mismo: `load_context` activa, `unload_context` desactiva, `clear_contexts` limpia todo.

### El toggle de modo

```python
@property
def mode(self) -> str:
    return ChatMode.SOFT if self.soft_mode else ChatMode.HARD


def toggle_mode(self) -> str:
    self.soft_mode = not self.soft_mode
    return self.mode
```

`@property` hace que `session.mode` se lea como atributo pero se compute en el momento. Nunca puede haber incoherencia entre `soft_mode` y `mode`.

### El toggle de agente

```python
def toggle_agent(self) -> None:
    self.agent_active = not self.agent_active
```

Alterna entre el pipeline lineal (sin grafo) y el pipeline con LangGraph.

### Memoria de conversación

```python
def reset_memory(self) -> None:
    self.chat_memory = []


def add_to_memory(self, user: str, assistant: str) -> None:
    self.chat_memory.append(TurnMemory(user=user, assistant=answer))
```

`add_to_memory` crea un `TurnMemory` (modelo congelado) y lo agrega a la lista. `reset_memory` limpia todo el historial.

### El header del input

```python
def get_prompt_header(self) -> str:
    """
    Ejemplo de salida:
      [ CONTEXT: debord, freud | MODE: SOFT | AGENT: ON ]
      >
    """
    active: list[str] = self.get_active_contexts()
    mode_label: str = "SOFT" if self.soft_mode else "HARD"

    if not active:
        ctx_label = "No-context"
    else:
        names = [ctx.split("/")[-1] for ctx in sorted(active)]
        ctx_label = ", ".join(names)

    return (
        f"[ CONTEXT: {ctx_label} " + f"| MODE: {mode_label}" + " "
        f"| AGENT: {'ON' if self.agent_active else 'OFF'} ]" + "\n> "
    )
```

Muestra en cada línea de input: contextos activos (solo el basename, no el namespace completo), modo actual, y si el agente está encendido.

---

## `interface.py` — dos handlers, un dispatcher

### El grafo como singleton lazy

```python
_compiled_graph = None


def _get_graph() -> CompiledStateGraph[RAGState]:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph
```

El grafo compilado se construye una sola vez por proceso y se reutiliza en todas las llamadas posteriores. Esto evita el costo de recompilar el grafo en cada pregunta.

### El loop principal

```python
def start_chat(session: ChatSession) -> None:
    while True:
        command = input(session.get_prompt_header()).strip()

        if command == "/exit":
            break
        if command == "/help":
            print(HELP)
        if command == "/context":
            _handle_context(session, ...)
        if command == "/remove":
            _handle_remove(session, ...)
        if command == "/list":
            _print_available(session)
        if command == "/active":
            _print_active(session)
        if command == "/clear":
            session.clear_contexts()
        if command == "/reset":
            session.reset_memory()
        if command == "/mode":
            session.toggle_mode()
        if command == "/agent":
            session.toggle_agent()
        if command.startswith("/"):
            print(f"Unknown command: {command!r}")

        # Texto sin "/" → es una pregunta
        if session.agent_active:
            _handle_agent_question(session, command)
        else:
            _handle_question(session, command)
```

Las últimas dos líneas son el dispatch real: si el agente está activo, la pregunta pasa por el grafo LangGraph; si no, por el pipeline lineal.

### `_handle_question()` — el pipeline lineal

```python
def _handle_question(session: ChatSession, question: str) -> None:
    collections = session.context_manager.get_loaded_collections()

    results, confidence = search(
        question=question,
        mode=session.mode,
        collections=collections,
    )

    context_chunks = format_context_chunks(results)

    prompt = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=session.chat_memory,
        provider=LLMRole.GENERATE.value,
    )

    session.add_to_memory(user=question, assistant=answer)
```

La secuencia es lineal: `search()` → `build_prompt()` → `ask_llm()`. Sin reformulación automática, sin reviewer, sin corrección. Disponible para iteración rápida sin la latencia del grafo.

### `_handle_agent_question()` — el pipeline con grafo LangGraph

```python
def _handle_agent_question(session: ChatSession, question: str) -> None:
    graph = _get_graph()

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
        "max_tokens": None,
        "think_mode": None,
        "extra": None,
        "attachments": [],
    }

    final_state = cast(RAGState, graph.invoke(initial_state))
    answer = final_state["answer"]
    confidence = final_state["confidence"]
    reformulated = final_state["reformulated"]
```

En lugar de llamar directamente a `search()` y `ask_llm()`, construye el estado inicial y le pasa el control al grafo. El grafo devuelve el estado final con `answer` ya populado.

Si `reformulated=True`, el grafo detectó baja confianza en los resultados iniciales, reformuló la query, y buscó de nuevo. El CLI lo muestra al usuario:

```
  [agent] Low confidence in initial search → query reformulated automatically.
```

El estado inicial incluye campos vacíos para review (`review_passed=False`, `review_feedback=""`, `review_attempts=0`) y campos de override (`max_tokens=None`, `think_mode=None`, `extra=None`). El grafo los actualiza a medida que avanza.

---

## Comandos disponibles

| Comando | Acción |
|---|---|
| `/context <tokens>` | Cargar contexto(s) |
| `/remove <tokens>` | Descargar contexto(s) |
| `/list` | Ver contextos disponibles |
| `/active` | Ver contextos activos |
| `/clear` | Descargar todos los contextos |
| `/reset` | Limpiar memoria de conversación |
| `/mode` | Alternar modo (SOFT / HARD) |
| `/agent` | Activar/desactivar modo agente (LangGraph) |
| `/help` | Mostrar ayuda |
| `/exit` | Salir |

La sintaxis de tokens permite mezclar namespaces y colecciones exactas:

```
sociologia                    → todo el namespace
sociologia/debord             → colección exacta
sociologia react              → dos namespaces
sociologia/debord react/hooks → mezcla de exacto y namespaces
```

---

## Comparación de los dos caminos

| | Pipeline lineal (`/agent OFF`) | Grafo LangGraph (`/agent ON`) |
|---|---|---|
| Reformulación automática | No | Sí, si confidence < límite |
| Review de alucinaciones | No | Sí, via LLM de review |
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
               search()            _get_graph() (cached)
               build_prompt()           │
               ask_llm()          graph.invoke(initial_state)
                         │                     │
                    answer: str           answer: str
                         │                     │
                    session.add_to_memory()
                    (actualiza chat_memory para el próximo turno)
```
