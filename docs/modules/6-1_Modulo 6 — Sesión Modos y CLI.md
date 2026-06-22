# Módulo 6 — Sesión, Modos y CLI

Este es el módulo que conecta todo lo anterior con el usuario. Si los módulos 1 al 5 son los órganos del sistema, este módulo es el sistema nervioso — coordina, mantiene estado, y traduce las acciones del usuario en llamadas a los módulos correctos.

---

## `modes.py` — el contrato más simple del proyecto

```python
class ChatMode:
    RIGOROUS = "RIGUROSO"
    INTERPRETATIVE = "INTERPRETATIVO"

RIGOROUS = ChatMode.RIGOROUS
INTERPRETATIVE = ChatMode.INTERPRETATIVE
```

Esto merece atención precisamente por su simplicidad. Son constantes de string que funcionan como un enum. ¿Por qué no usar `Enum` de Python directamente?

Porque estas constantes viajan como texto plano hasta el prompt (recuerda el Módulo 5: `build_rules_block(mode)` recibe un string y lo compara con `ChatMode.INTERPRETATIVE`). Un `Enum` real requeriría `.value` para extraer el string. Esta solución es más pragmática para un sistema donde el modo necesita ser tanto un identificador de código como texto legible en el prompt.

---

## `session.py` — el objeto que mantiene el estado vivo

```python
class ChatSession:

    def __init__(self) -> None:
        self.context_manager: ContextManager = ContextManager()
        self.interpretative_mode: bool = False
        self.chat_memory: list[TurnMemory] = []
```

`ChatSession` tiene exactamente tres responsabilidades y tres atributos que las reflejan:

```
context_manager    → qué colecciones están cargadas en memoria
interpretative_mode → en qué modo de respuesta estamos
chat_memory        → qué se ha hablado en esta sesión
```

Nada más. Es el objeto que `main.py` instancia una vez al arrancar el proceso y que se pasa a través de toda la aplicación. Es el único estado mutable del sistema — todo lo demás son funciones puras o estructuras inmutables.

Visualiza la relación entre `ChatSession` y los módulos anteriores:

```
ChatSession
    │
    ├── context_manager (Módulo 3)
    │       └── loaded_contexts: dict[str, LoadedCollection]
    │                                └── index, metadata, vectors (Módulo 2)
    │
    ├── chat_memory (Módulo 5)
    │       └── list[TurnMemory] → va al prompt en cada consulta
    │
    └── mode (Módulo 4 + 5)
            └── afecta build_queries() y build_rules_block()
```

---

## Los métodos de `ChatSession`

La mayoría son delegaciones directas a `ContextManager`, pero vale la pena ver el patrón:

```python
def load_context(self, pattern: str) -> list[str]:
    return self.context_manager.activate(pattern)

def unload_context(self, pattern: str) -> list[str]:
    return self.context_manager.deactivate(pattern)

def clear_contexts(self) -> None:
    self.context_manager.clear()

def get_active_contexts(self) -> list[str]:
    return self.context_manager.get_active()
```

`ChatSession` no reimplementa nada — delega. Su rol respecto a `ContextManager` es de fachada: expone una interfaz limpia al resto del sistema sin exponer los detalles internos del manager. Si mañana cambias la implementación de `ContextManager`, `interface.py` no necesita saber nada.

**El toggle de modo:**

```python
@property
def mode(self) -> str:
    return INTERPRETATIVE if self.interpretative_mode else RIGOROUS

def toggle_mode(self) -> str:
    self.interpretative_mode = not self.interpretative_mode
    return self.mode
```

`@property` hace que `session.mode` se lea como un atributo pero se compute como una función. El resultado es siempre coherente — no puedes tener `interpretative_mode = True` y `mode = "RIGUROSO"` simultáneamente porque `mode` se calcula en el momento de la lectura.

**La memoria conversacional:**

```python
def add_to_memory(self, user: str, assistant: str) -> None:
    self.chat_memory.append(TurnMemory(user=user, assistant=assistant))

def reset_memory(self) -> None:
    self.chat_memory = []
```

Cada turno completo (pregunta + respuesta) se agrega después de que el LLM responde. `reset_memory` simplemente reemplaza la lista con una vacía — la lista anterior queda sin referencias y Python la libera.

---

## `get_prompt_header()` — el header del input

```python
def get_prompt_header(self) -> str:
    active: list[str] = self.get_active_contexts()
    mode_label: str = "INTERP" if self.interpretative_mode else "RIG"

    if not active:
        ctx_label = "SIN-CONTEXTO"
    else:
        names = [ctx.split("/")[-1] for ctx in sorted(active)]
        ctx_label = ", ".join(names)

    return f"[{ctx_label} | {mode_label}] > "
```

Este método computa el string que ves en cada línea de input del chat:

```
[debord, freud | INTERP] >
```

`ctx.split("/")[-1]` extrae solo el basename de cada colección — `"sociologia/debord"` → `"debord"`. Es la misma solución que discutiste en una sesión anterior cuando el header se desbordaba con los nombres completos.

La información comprimida en este header es deliberada — al escribir cada mensaje puedes ver de un vistazo qué contextos están activos y en qué modo estás, sin necesitar ejecutar `/active` o `/mode`.

---

## `interface.py` — el loop central

Este archivo es la capa de presentación que une todo. Su estructura es un loop infinito con un dispatcher de comandos:

```python
def start_chat(session: ChatSession) -> None:
    print("\n  === CHAT ===")
    print("  Escribe /help para ver los comandos disponibles.\n")

    while True:
        try:
            command: str = input(session.get_prompt_header()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue

        if command == "/exit":
            break

        if command.startswith("/context"):
            _handle_context(session, command[len("/context"):].strip())
            continue

        # ... resto de comandos ...

        if command.startswith("/"):
            print(f"\n  Comando desconocido: {command!r}  (escribe /help)\n")
            continue

        _handle_question(session, command)    # ← cualquier texto sin "/" es una pregunta
```

El patrón de dispatch es secuencial — Python evalúa cada `if` en orden. El orden importa:

```
1. ¿Es "/exit"?          → salir del loop
2. ¿Empieza con "/"?     → es un comando
   ├─ /context, /remove, /list, /active, /clear, /reset, /mode, /help
   └─ /algo_desconocido  → mensaje de error
3. ¿No empieza con "/"?  → es una pregunta → _handle_question()
```

El guard final `if command.startswith("/")` actúa como catch-all para comandos mal escritos. Si no estuviera, un typo como `/contextoo` llegaría hasta `_handle_question()` y se trataría como una pregunta — confuso para el usuario.

---

## `_handle_question()` — donde convergen todos los módulos

```python
def _handle_question(session: ChatSession, question: str) -> None:

    collections: list[LoadedCollection] = session.context_manager.get_loaded_collections()

    if not collections:
        print("\n  Carga un contexto primero.  Ej: /context sociologia\n")
        return

    print("\n  Buscando...\n")

    results, confidence = search(
        question=question,
        mode=session.mode,
        chat_memory=session.chat_memory,
        collections=collections,
    )

    if not results:
        print("  No se encontró contexto relevante.\n")
        return

    context_chunks: list[str] = [
        f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
        for r in results
    ]

    prompt: str = build_prompt(
        context_chunks=context_chunks,
        question=question,
        mode=session.mode,
        chat_memory=session.chat_memory,
    )

    answer: str = ask_llm(
        prompt=prompt,
        chat_memory=session.chat_memory,
    )

    print("  Respuesta:\n")
    print(answer)
    print(f"\n  [confidence: {confidence:.4f}]\n")

    session.add_to_memory(user=question, assistant=answer)
```

Esta función de 30 líneas es el corazón operativo del sistema. Es literalmente la secuencia de los módulos 3, 4 y 5 ejecutándose en orden:

```
session.context_manager.get_loaded_collections()  ← Módulo 3
search(question, mode, memory, collections)        ← Módulo 4
build_prompt(chunks, question, mode, memory)       ← Módulo 5
ask_llm(prompt, memory)                            ← Módulo 5
session.add_to_memory(question, answer)            ← actualiza estado
```

La última línea es especialmente importante: **`add_to_memory` se llama después de obtener la respuesta**, no antes. Así el turno actual (pregunta + respuesta) estará disponible para la próxima consulta, pero no contamina el contexto de la consulta actual. Es una cola append-only que crece turno a turno.

---

## `main.py` — el punto de entrada

```python
def main():
    session = ChatSession()

    while True:
        show_main_menu()
        choice = input("> ").strip()

        if choice == "1":
            start_chat(session)
        elif choice == "2":
            show_contexts(session)
        elif choice == "3":
            show_modes(session)
        elif choice == "4":
            show_about(session)
        elif choice == "5":
            print("\n  Hasta luego.\n")
            break
        else:
            print("\n  Opción inválida.\n")
```

`ChatSession()` se instancia aquí, una sola vez, y se pasa por referencia a todo el sistema. Cada vez que vuelves al menú desde el chat (`/exit`), `session` sigue vivo con su estado intacto — los contextos cargados, el historial, el modo. No se reinicia hasta que el proceso termina o `/exit` llega al `main()`.

---

## La visión completa — todos los módulos conectados

Ahora puedes ver el sistema completo de extremo a extremo:

```
OFFLINE (ingest)                          ONLINE (runtime)
────────────────                          ────────────────

Archivos/URLs                             main() instancia ChatSession
     │                                         │
     ▼  Módulo 1                               ▼
chunk_text()                             show_main_menu()
encode_chunks()                               │
     │                                   "1" → start_chat(session)
     ▼  Módulo 2                               │
index.faiss ──────────────────────────→  /context sociologia
metadata.pkl                             session.load_context()
vectors.npy                              load_collection() ← lee disco
                                              │
                                         usuario escribe pregunta
                                              │
                                         _handle_question()
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                               Módulo 4             Módulo 3
                             search()          get_loaded_collections()
                           build_queries()         │
                           encode_queries()    loaded_contexts{}
                           retrieve() ─────→  index.search()
                           rerank()               │
                                └────────── SearchResult[]
                                                  │
                                             Módulo 5
                                           build_prompt()
                                             │
                                           ask_llm()
                                             │
                                    HTTP POST → LLM local
                                             │
                                          answer: str
                                             │
                                    add_to_memory()
                                    (estado actualizado
                                     para el próximo turno)
```

---

## Lo que puedes decir en una entrevista

Si te piden que describas este proyecto de principio a fin, tienes una narrativa clara:

> "Es un sistema RAG con dos fases separadas. En la fase de ingesta, los documentos se dividen en chunks con overlap, se convierten a embeddings con un modelo de sentence-transformers, y se persisten en FAISS junto con su metadata. En tiempo de consulta, la pregunta pasa por el mismo modelo de embeddings, se busca por similitud coseno en los índices FAISS de las colecciones activas, los chunks más relevantes se insertan en un prompt estructurado junto con el historial de conversación, y ese prompt se envía a un LLM local vía API OpenAI-compatible. El sistema soporta múltiples colecciones independientes que se pueden cargar y descargar en memoria durante la sesión, y dos modos de búsqueda que afectan tanto el número de queries generadas como las instrucciones en el prompt."

Eso es lo que distingue a alguien que construyó el sistema de alguien que solo lo usó.

---
