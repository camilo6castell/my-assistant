# Módulo 3 — Gestión de Contextos

Hasta ahora vimos cómo se crea y guarda una colección. El Módulo 3 responde una pregunta diferente: **¿cómo el sistema administra múltiples colecciones en memoria durante una sesión de chat?**

El problema concreto: tienes esto en disco:

```
vector_stores/
├── sociologia/
│   ├── debord/
│   └── veblen/
├── psicoanalisis/
│   └── freud/
└── react/
    └── hooks/
```

El usuario puede querer chatear con `sociologia` y `psicoanalisis` al mismo tiempo, luego descargar `sociologia` y agregar `react`. El `ContextManager` es el responsable de manejar ese estado.

---

## `models.py` — Los contratos de datos

Antes de ver el manager, los dos tipos que viajan por todo el sistema:

```python
@dataclass(slots=True)
class ContextSource:
    source_id: str
    source_type: str
    source_name: str

@dataclass(slots=True)
class SearchResult:
    score: float
    text: str
    source: str
    page: int
    collection: str
    chunk_index: int
```

`SearchResult` es el tipo que el sistema produce al final de una búsqueda — ya lo viste en el Módulo 2. Cada resultado contiene el texto del chunk, de dónde vino, y qué tan relevante fue (`score`).

`slots=True` es un detalle de Python que vale la pena entender: normalmente cada instancia de una clase guarda sus atributos en un diccionario interno (`__dict__`). Con `slots=True`, Python reserva memoria fija para cada atributo en lugar de usar un dict. El resultado es acceso más rápido y menos memoria consumida — importante cuando tienes miles de `SearchResult` en memoria simultáneamente.

---

## `manager.py` — El estado de la sesión

El `ContextManager` es esencialmente un diccionario que actúa como caché de colecciones cargadas:

```python
class ContextManager:

    def __init__(self) -> None:
        self.base_path: Path = Path(BASE_VECTOR_PATH)
        self.loaded_contexts: dict[str, LoadedCollection] = {}
```

`loaded_contexts` es la estructura central. Es un `dict` donde:

- **clave** → nombre lógico de la colección (`"sociologia/debord"`)
- **valor** → la colección completa en memoria (índice FAISS + metadata + vectores)

Visualizado:

```python
loaded_contexts = {
    "sociologia/debord": {
        "index":           <faiss.IndexFlatIP>,   # listo para buscar
        "metadata":        [{...}, {...}, ...],    # 1847 chunks
        "vectors":         np.ndarray(1847, 384),  # matriz en RAM
        "paths":           {...},                  # rutas en disco
        "collection_name": "sociologia/debord"
    },
    "psicoanalisis/freud": {
        "index":           <faiss.IndexFlatIP>,
        "metadata":        [{...}, {...}, ...],    # 2103 chunks
        ...
    }
}
```

Cuando el usuario escribe `/context sociologia/debord`, esa colección se carga desde disco a este dict y **permanece en RAM** para todas las consultas siguientes de esa sesión. No se vuelve a leer disco hasta que se descarga o el proceso termina.

Esto es un patrón clásico de caché — cargar una vez, usar muchas veces.

---

## `list_all()` — Descubrir qué existe en disco

```python
def list_all(self) -> list[str]:
    contexts: list[str] = []

    if not self.base_path.exists():
        return contexts

    for namespace in self.base_path.iterdir():
        if not namespace.is_dir():
            continue

        for collection in namespace.iterdir():
            if collection.is_dir():
                contexts.append(f"{namespace.name}/{collection.name}")

    return sorted(contexts)
```

Dos loops anidados que recorren la estructura de directorios. El primero itera los namespaces (`sociologia`, `psicoanalisis`), el segundo itera las colecciones dentro de cada namespace (`debord`, `veblen`).

El resultado es una lista plana de strings con el formato `"namespace/coleccion"`:

```python
["psicoanalisis/freud", "react/hooks", "sociologia/debord", "sociologia/veblen"]
```

No hay ningún archivo de registro ni base de datos de colecciones. **El sistema de archivos es la fuente de verdad**. Esto es una decisión de diseño deliberada: agregar o eliminar una colección es tan simple como crear o borrar una carpeta.

---

## `activate()` — Cargar a memoria

```python
def activate(self, pattern: str) -> list[str]:
    matches: list[str] = self.resolve_pattern(pattern)

    if not matches:
        return []

    loaded: list[str] = []

    for context_name in matches:
        if context_name in self.loaded_contexts:
            continue                          # ya está en caché, skip

        try:
            raw: RawCollection = load_collection(context_name)

            self.loaded_contexts[context_name] = LoadedCollection(
                index=raw["index"],
                metadata=raw["metadata"],
                vectors=raw["vectors"],
                paths=raw["paths"],
                collection_name=context_name,
            )

            loaded.append(context_name)

        except Exception:
            logger.exception(f"Error cargando contexto: {context_name}")

    return loaded
```

Tres partes:

**1. Resolver el patrón** — `resolve_pattern` convierte un string como `"sociologia"` en la lista `["sociologia/debord", "sociologia/veblen"]`. Lo veremos en detalle en `selector.py`.

**2. Guard de caché** — `if context_name in self.loaded_contexts: continue`. Si ya está cargado, no se toca disco. Idempotente igual que el ingest.

**3. Carga y registro** — `load_collection` lee los tres archivos de disco y los devuelve como `RawCollection`. Luego se guarda en `loaded_contexts` con el nombre lógico agregado como campo extra (`collection_name`). El `try/except` garantiza que si una colección falla al cargar, el resto continúa.

---

## `selector.py` — La resolución de patrones

Este archivo responde a la pregunta: cuando el usuario escribe `/context sociologia`, ¿cómo el sistema sabe que eso significa "todas las colecciones bajo el namespace sociologia"?

```python
def _match_single(pattern: str, available: list[str]) -> list[str]:

    pattern = pattern.strip()

    # Caso 1: "sociologia/*"  →  namespace explícito
    if pattern.endswith("/*"):
        namespace = pattern[:-2]
        return [ctx for ctx in available if ctx.startswith(namespace + "/")]

    # Caso 2: "sociologia/debord"  →  match exacto
    if "/" in pattern:
        return [pattern] if pattern in available else []

    # Caso 3: "sociologia"  →  namespace implícito (azúcar sintáctico)
    prefix = pattern + "/"
    matches = [ctx for ctx in available if ctx.startswith(prefix)]

    if not matches and pattern in available:
        return [pattern]

    return matches
```

Tres casos, con prioridad de arriba a abajo:

```
"sociologia/*"          →  Caso 1: todas las de sociologia/
"sociologia/debord"     →  Caso 2: exactamente esa
"sociologia"            →  Caso 3: igual que Caso 1, más cómodo de escribir
```

El Caso 3 es azúcar sintáctico — existe para que el usuario no tenga que recordar la sintaxis `/*`.

**`match_contexts()` — la versión multi-token:**

```python
def match_contexts(raw: str, available_contexts: list[str]) -> list[str]:
    tokens = raw.strip().split()      # "sociologia react" → ["sociologia", "react"]

    seen: set[str] = set()
    result: list[str] = []

    for token in tokens:
        for ctx in _match_single(token, available_contexts):
            if ctx not in seen:
                seen.add(ctx)
                result.append(ctx)

    return sorted(result)
```

Divide el input por espacios y aplica `_match_single` a cada token. El `set` `seen` garantiza que no haya duplicados si dos tokens resuelven a la misma colección.

```
"/context sociologia/debord react psicoanalisis"
    │
    ▼
tokens = ["sociologia/debord", "react", "psicoanalisis"]
    │
    ├── "sociologia/debord" → ["sociologia/debord"]
    ├── "react"             → ["react/hooks"]
    └── "psicoanalisis"     → ["psicoanalisis/freud"]
    │
    ▼
["psicoanalisis/freud", "react/hooks", "sociologia/debord"]
```

---

## El ciclo de vida completo de un contexto

```
DISCO                    MANAGER                  CONSULTA
─────                    ───────                  ────────

vector_stores/           loaded_contexts = {}
  sociologia/debord/
  psicoanalisis/freud/

                         activate("sociologia")
                              │
                         load_collection()
                         lee 3 archivos de disco
                              │
                         loaded_contexts = {
                           "sociologia/debord": <coleccion>
                         }
                                                  search() usa
                                                  loaded_contexts

                         deactivate("sociologia")
                              │
                         del loaded_contexts["sociologia/debord"]
                              │
                         loaded_contexts = {}     RAM liberada
```

`deactivate` y `clear` son simplemente `del` sobre el dict — cuando Python elimina la referencia, el garbage collector libera la RAM que ocupaban el índice FAISS y la matriz numpy.

---

## Por qué esta arquitectura importa en entrevista

Si te preguntan "¿cómo manejas múltiples fuentes de conocimiento en tu RAG?", la respuesta que da un desarrollador senior es exactamente esta:

> "Cada colección es independiente en disco. En memoria, uso un dict como caché que mapea nombre lógico a colección cargada. La búsqueda posterior opera sobre todas las colecciones activas en paralelo. Cargar y descargar contextos es O(1) en el dict, sin reindexación."

¿Avanzamos al Módulo 4 — el corazón del RAG: cómo la búsqueda semántica convierte tu pregunta en resultados concretos?
