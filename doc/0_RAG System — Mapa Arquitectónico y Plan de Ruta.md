# RAG System — Mapa Arquitectónico y Plan de Ruta

> Panorama completo del sistema RAG (Retrieval-Augmented Generation). Este documento
> describe **cómo funciona** el sistema actualmente: su arquitectura, su flujo de datos
> y todos los módulos que lo componen.

---

## 1. Diagrama de Flujo General

Un sistema RAG tiene dos grandes etapas: **ingesta** (offline, una vez) y **consulta**
(runtime, cada vez que el usuario pregunta). Este proyecto las implementa así:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        INGESTA  (offline)                                  │
│                                                                            │
│   Archivo / URL                                                            │
│       │                                                                    │
│       ▼                                                                    │
│   ┌──────────────┐     ┌────────────────┐     ┌────────────────────────┐   │
│   │ chunk_text() │───▶│encode_chunks() │───▶│ save_collection()      │   │
│   │ (core.py)    │     │ (core.py →     │     │ (faiss_store.py)       │   │
│   │              │     │  encoder.py)   │     │                        │   │
│   │ Divide en    │     │ Embeddings     │     │ Persiste en disco:     │   │
│   │ fragmentos   │     │ pluggables     │     │  index.faiss           │   │
│   │              │     │ (3 backends)   │     │  metadata.pkl          │   │
│   └──────────────┘     └────────────────┘     │  vectors.npy           │   │
│                                               └────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        CONSULTA  (runtime)                              │
│                                                                         │
│   Pregunta del usuario                                                  │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                    GRAFO LangGraph                               │  │
│   │                                                                  │  │
│   │  ┌────────────┐   ┌────────────┐   ┌──────────────────────┐      │  │
│   │  │  retrieve  │─▶│  evaluate  │─▶│ ¿confidence >= 0.79? │      │  │
│   │  │            │   │            │   └──────┬───────────────┘      │  │
│   │  │ search()   │   │  Log only  │          │                      │  │
│   │  │ en FAISS   │   │            │     Sí   ▼   No                 │  │
│   │  │ top-K      │   └────────────┘     │         │                 │  │
│   │  └────────────┘                      ▼         ▼                 │  │
│   │                               ┌──────────┐ ┌──────────────┐      │  │
│   │                               │ generate │ │ reformulate  │      │  │
│   │                               │          │ │ (reintenta)  │      │  │
│   │                               └────┬─────┘ └──────────────┘      │  │
│   │                                    │                             │  │
│   │                                    ▼                             │  │
│   │                               ┌──────────┐                       │  │
│   │                               │  review  │  ← Evalúa grounding   │  │
│   │                               │ (Gemini) │    y citaciones       │  │
│   │                               └────┬─────┘                       │  │
│   │                              passed│   │rejected                 │  │
│   │                                    ▼   ▼                         │  │
│   │                                  END ┌─────────┐                 │  │
│   │                                      │ correct │  ← Regenera     │  │
│   │                                      └────┬────┘    con feedback │  │
│   │                                           │                      │  │
│   │                                           ▼                      │  │
│   │                                        review  (máx N intentos)  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                 │
│       ▼                                                                 │
│   Respuesta al usuario (con fuentes, grounding verificado)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ¿Qué hace especial a este RAG?

Un RAG básico hace: buscar → generar. Este sistema añade capas que lo
hacen más robusto y flexible:

### 2.1 Recuperación adaptativa (módulo 7)

Cuando la confianza de los resultados es baja (`< 0.79`), el grafo
**reformula automáticamente** la pregunta del usuario usando el LLM
(y con vocabulario del dominio) y reintenta la búsqueda. Si el usuario
pregunta "¿Qué dijo Freud sobre la cultura?" y los resultados son pobres,
el sistema reescribe a algo como "Freud conceptualización de la cultura
en El malestar en la cultura" y busca de nuevo.

### 2.2 Multi-proveedor con roles (módulo 8)

Cada paso del pipeline puede usar un **proveedor LLM diferente**:
uno para generar respuestas, otro para reformular, otro para revisar.
Los proveedores son configurables en `.env.providers` y se resuelven
en `src/nlp/llm/providers.py` sin tocar el grafo.

### 2.3 Reviewer (módulo 8)

Después de generar una respuesta, un **segundo LLM** (Gemini) la evalúa
verificando:
- **Grounding**: ¿las afirmaciones están respaldadas por el contexto?
- **Citaciones**: ¿cita las fuentes correctamente?

Si la revisión falla, el sistema regenera la respuesta incorporando
el feedback del reviewer (máximo `MAX_REVIEW_ATTEMPTS` ciclos).

### 2.4 Embedders pluggables (módulo 1)

El sistema de embeddings es **multi-backend**: puede usar
`sentence_transformers` (local), `ollama` (GPU Vulkan) o `flm`
(NPU via FastFlowLM). Se configura con `EMBEDDER=ollama,bge-m3`
en `.env` y el encoder se resuelve en `get_encoder()` con cache LRU.

### 2.5 Modos de respuesta (módulo 6)

- **SOFT**: síntesis profunda, múltiples fuentes, respuestas largas.
- **HARD**: solo lo explícitamente dicho en las fuentes, sin inferencias.

### 2.6 Almacenamiento efímero y adjuntos (módulo 9)

Archivos subidos "solo para esta conversación" se indexan en memoria
(sobreviven al proceso pero no al reinicio). Los adjuntos se inyectan
raw en el prompt sin indexar.

### 2.7 Validación preventiva de contexto (módulo 11)

Antes de llamar al LLM, `context_guard.py` estima los tokens totales
y los compara con la ventana del modelo activo. Si no cabe, rechaza
con 413 en lugar de esperar una respuesta truncada.

---

## 3. Plan de Ruta — Todos los Módulos

### Módulo 0: Esta Visión General

Este documento. No contiene código; describe la arquitectura completa.

---

### Módulo 1 — Ingesta y Embeddings

Archivos: `src/ingest/core.py`, `src/ingest/ingest.py`,
`src/ingest/web_ingest.py`, `src/ingest/web_crawler.py`,
`src/ingest/http.py`

**Qué hace**: Toma documentos (PDF, HTML, TXT) o URLs, los divide en
fragmentos, genera embeddings y los almacena en FAISS.

| Función | Ubicación | Responsabilidad |
|---------|-----------|-----------------|
| `chunk_text()` | `core.py` | Divide texto en fragmentos de `chunk_size` con `chunk_overlap` |
| `encode_chunks()` | `core.py` | Wrapper delgada sobre `get_encoder().encode()` |
| `build_metadata()` | `core.py` | Construye `ChunkMetadata` (modelo Pydantic) |
| `extract_main_content()` | `http.py` | Descarga URL y extrae contenido con `readability-lxml` |
| `main()` | `ingest.py` | Orquesta ingesta de archivos locales |
| `main()` | `web_ingest.py` | Orquesta ingesta de una URL individual |
| `main()` | `web_crawler.py` | Crawls BFS dentro de un dominio |

Ver Módulo 1 detallado: `doc/1-1_Módulo 1 — Ingesta y Embeddings_core.py_.md`
y `doc/1-2_Módulo 1 — ingest.py_El Director de Orquesta.md`

---

### Módulo 2 — Almacenamiento Vectorial

Archivos: `src/storage/faiss_store.py`

**Qué hace**: Capa unificada de persistencia FAISS + pickle + numpy.
Punto único de verdad para cargar, guardar, reconstruir y eliminar
colecciones.

| Tipo | Descripción |
|------|-------------|
| `ChunkMetadata` | Modelo Pydantic frozen con: `source`, `source_type`, `page`, `text`, `chunk_index`, `collection`, `file_id` |
| `RawCollection` | TypedDict con: `index`, `metadata`, `vectors`, `paths` |
| `load_collection()` | Carga index.faiss + metadata.pkl + vectors.npy |
| `save_collection()` | Apunta embeddings/metadata nuevos y persiste |
| `delete_by_sources()` | Elimina chunks por fuente |
| `vacuum_collection()` | Compacta tras eliminaciones |

Archivos en disco por colección:
```
/srv/ai/vector_stores/<backend>/<model_safe>/<category>/<collection>/
    index.faiss      ← Índice FAISS (IndexFlatIP)
    metadata.pkl     ← Lista de ChunkMetadata serializados
    vectors.npy      ← Array numpy de embeddings
```

---

### Módulo 3 — Gestión de Contextos

Archivos: `src/context/manager.py`, `src/context/selector.py`,
`src/context/models.py`, `src/context/delete.py`

**Qué hace**: Descubre colecciones en disco, las carga en memoria
para la consulta, resuelve patrones de namespace y gestiona eliminación.

| Clase/Función | Descripción |
|---------------|-------------|
| `ContextManager` | Carga/descarga colecciones en memoria. `activate("sociologia")` carga todo el namespace. |
| `match_namespace()` | Resuelve patrones: `"sociologia"` → todas las colecciones bajo `sociologia/` |
| `match_contexts()` | Resuelve múltiples tokens separados por espacio |
| `SearchResult` | Modelo Pydantic frozen: `score`, `text`, `source`, `page`, `collection`, `chunk_index` |
| `list_sources()` | Resume qué fuentes hay indexadas en una colección |
| `delete_by_source()` | Elimina una fuente específica |

---

### Módulo 4 — Recuperación Semántica

Archivos: `src/retrieval/search.py`

**Qué hace**: Convierte la pregunta en embeddings, busca en FAISS,
reordena por relevancia y calcula un score de confianza.

Flujo detallado:
```
question
    │
    ▼
build_queries(question, mode)
    │  HARD: 1 query literal
    │  SOFT: 3 queries (literal + 2 variantes semánticas)
    ▼
encode_queries(queries)        ← get_encoder().encode()
    │
    ▼
retrieve(embeddings, collections, top_k_initial)
    │  Búsqueda en cada colección FAISS
    │  IndexFlatIP → coseno similarity
    ▼
rerank(results)                ← Dedup + sort por score
    │
    ▼
results[:top_k_final], confidence
```

---

### Módulo 5 — Generación y Prompts

Archivos: `src/prompts/builder.py`, `src/nlp/llm/generate.py`

**Qué hace**: Construye los prompts del sistema (RAG, reformulación,
review, corrección, web supplement) y orquesta las llamadas al LLM.

| Función en `builder.py` | Para qué |
|--------------------------|----------|
| `build_system_prompt()` | Prompt principal RAG: ROLE → GROUNDING → CITATION → STYLE |
| `build_reformulation_system_prompt()` | Reescritura de queries |
| `build_review_system_prompt()` | Validación del reviewer |
| `build_correction_prompt()` | Regeneración post-review |
| `build_web_supplement_prompt()` | Complemento con fuentes web |

| Función en `generate.py` | Para qué |
|---------------------------|----------|
| `build_messages()` | Construye array `[system, history..., user]` para la API |
| `ask_llm()` | Llamada LLM para respuesta al usuario |
| `ask_llm_internal()` | Llamada LLM para operaciones internas (reformulación, review) |

---

### Módulo 6 — Sesión, Modos y CLI

Archivos: `src/cli/session.py`, `src/cli/interface.py`,
`src/cli/modes.py`

**Qué hace**: Interfaz de línea de comandos para interactuar con el
sistema RAG sin API web.

| Clase | Descripción |
|-------|-------------|
| `ChatSession` | Estado de la sesión: contextos activos, modo (SOFT/HARD), historial, agente ON/OFF |
| `interface.py` | Loop de comandos: `/context`, `/remove`, `/list`, `/active`, `/mode`, `/agent` |

Modos disponibles en `ChatMode` (`src/domain/models.py`):
- `SOFT` → respuestas sintéticas, largas, multi-fuente
- `HARD` → solo lo explícito en las fuentes

---

### Módulo 7 — El Grafo LangGraph

Archivos: `src/graph/graph.py`, `src/graph/nodes.py`,
`src/graph/state.py`

**Qué hace**: Grafo de estado que orquesta el pipeline RAG completo
con recuperación adaptativa y review.

Nodos del grafo:

| Nodo | Función | Qué hace |
|------|---------|----------|
| `retrieve` | `retrieve_node()` | Ejecuta `search()` sobre colecciones activas |
| `evaluate` | `evaluate_node()` | Evalúa confianza (solo log, el routing es en `route_after_evaluate`) |
| `reformulate` | `reformulate_node()` | Reescribe la pregunta con el LLM |
| `generate` | `generate_node()` | Construye prompt + llama al LLM + valida contexto |
| `review` | `review_node()` | Gemini evalúa grounding + citaciones |
| `correct` | `correct_node()` | Regenera con feedback del reviewer |

`RAGState` (`TypedDict`): el estado que viaja entre nodos.
LangGraph aplica cada retorno como merge parcial.

---

### Módulo 8 — Multi-Proveedor y Reviewer

Archivos: `src/nlp/llm/providers.py`, `src/nlp/llm/generate.py`,
`src/graph/nodes.py`

**Qué hace**: Resolución de proveedores LLM por rol. Cada paso del
pipeline (GENERATE, REFORMULATE, REVIEW, WEB_SUPPLEMENT) puede usar
un backend + modelo diferente.

```
LLMRole.GENERATE   → settings.role_spec() → ("flm", "qwen3.5:9b")
LLMRole.REVIEW     → settings.role_spec() → ("gemini", "gemini-2.0-flash")
LLMRole.REFORMULATE → settings.role_spec() → ("ollama", "qwen3.5:2b")
```

`ProviderConfig` (dataclass frozen):
```python
name: str  # "generate", "review", etc.
backend: str  # "flm", "ollama", "gemini"
base_url: str
api_key: str
model: str
client: str  # "openai_compat", "ollama_native"
capabilities: str  # → archivo en src/config/models/
```

Los clientes concretos (`LLMClient`) se cachean por
`(backend, base_url, api_key, model, client)` — no por rol.

---

### Módulo 9 — Almacenamiento Efímero y Adjuntos

Archivos: `src/context/ephemeral.py`, `src/context/attachments.py`

**Qué hace**: Gestiona archivos subidos por el usuario durante una
conversación, con dos enfoques diferentes:

| Almacén | Persistencia | Indexado | Uso |
|---------|-------------|----------|-----|
| `EphemeralStore` | Memoria del proceso (no sobrevive restart) | FAISS en memoria | Archivos para RAG semántico |
| `AttachmentStore` | Un solo envío | Sin indexar | Inyección raw en el prompt |

`EphemeralStore`:
- Un `conversation_id` = un `_ConversationStore` (metadata + vectors + index + files)
- `add_file()`: chunk → embed → agregar al índice
- `remove_file()`: filtra por `file_id`, reconstruye índice
- `sweep_expired()`: limpieza periódica por TTL

`AttachmentStore`:
- Archivos de texto plano (.py, .js, .md, .json, etc.)
- Se inyectan raw en el prompt via `inject_attachments()` en `builder.py`
- Se consumen automáticamente después de cada consulta

---

### Módulo 10 — Backends y Clientes LLM

Archivos: `src/nlp/llm/backends/base.py`,
`src/nlp/llm/backends/openai_compat.py`,
`src/nlp/llm/backends/ollama_native.py`

**Qué hace**: Implementaciones concretas de clientes LLM. Cada una
implementa el protocolo `LLMClient` (método `complete()`).

| Backend | SDK | Para quién |
|---------|-----|------------|
| `OpenAICompatClient` | `openai` (OpenAI SDK) | FastFlowLM (NPU), Gemini (endpoint compatible) |
| `OllamaNativeClient` | `ollama` (paquete nativo) | Ollama (GPU Vulkan) — soporte nativo de `think` |

```python
# src/nlp/llm/backends/base.py
class LLMClient(Protocol):
    def complete(self, kwargs: dict[str, Any]) -> str | None: ...
```

Los kwargs ya vienen resueltos por `src/config/models/build_kwargs()`:
el backend solo desempaqueta contra su SDK, sin saber qué modelo
está detrás.

---

### Módulo 11 — Validación de Contexto

Archivos: `src/nlp/llm/context_guard.py`, `src/utils/tokens.py`

**Qué hace**: Estimación preventiva de tokens para evitar que el LLM
reciba un prompt que excede su ventana de contexto.

Flujo:
```
system_prompt + prompt + chat_memory
        │
        ▼
  estimate_tokens()  ← tiktoken (cl100k_base) o heurística chars/3.5
        │
        ▼
  × (1 + SAFETY_MARGIN_RATIO)   ← margen de seguridad (10%)
        │
        ▼
  ¿estimated > limit - reserve?
        │
    Sí  → ContextLimitExceeded → HTTP 413
    No  → continúa正常mente
```

`ContextLimitExceeded` es una excepción con `as_detail()` que el router
de FastAPI traduce a una respuesta HTTP 413 con detalles del error.

---

## 4. Tipos de Dominio Compartidos

Archivo: `src/domain/models.py`

```python
class TurnMemory(BaseModel, frozen=True):
    """Un turno de la conversación."""

    user: str
    assistant: str


class ChatMode(StrEnum):
    SOFT = "SOFT"  # Síntesis profunda
    HARD = "HARD"  # Solo lo explícito


class LLMRole(StrEnum):
    GENERATE = "generate"
    REFORMULATE = "reformulate"
    REVIEW = "review"
    WEB_SUPPLEMENT = "web_supplement"
```

---

## 5. Ruta de Configuración

El sistema se configura exclusivamente via archivos de entorno:

| Archivo | Contenido |
|---------|-----------|
| `.env` | Configuración general: chunk_size, top_k, paths, etc. |
| `.env.providers` | Backends LLM, URLs, API keys, role assignments |

Nunca se leen variables de entorno directamente con `os.getenv()`.
Todo pasa por `src/config/settings.py` (pydantic-settings) que
valida tipos, constrains y relaciones entre campos al iniciar.

---

## 6. Diagrama de Dependencias entre Módulos

```
    CLI / API
        │
        ▼
    graph.py  ←── LangGraph
        │
        ├── nodes.py
        │      ├── retrieval/search.py
        │      ├── prompts/builder.py
        │      ├── nlp/llm/generate.py
        │      └── nlp/llm/context_guard.py
        │
        ▼
    retrieval/search.py
        │
        ├── nlp/embedders/encoder.py
        ├── context/manager.py
        └── context/models.py
        │
        ▼
    nlp/embedders/encoder.py     ←── settings (EMBEDDER=backend,model)
        │
        ▼
    storage/faiss_store.py
        │
        ▼
    /srv/ai/vector_stores/<backend>/<model_safe>/<collection>/
```
