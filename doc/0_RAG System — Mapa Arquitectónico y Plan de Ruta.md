# RAG System — Mapa Arquitectónico y Plan de Ruta

## El flujo completo en una imagen

```
INGESTA (offline)                         CONSULTA (runtime)
─────────────────                         ──────────────────

Archivo / URL                              Tu pregunta
     │                                          │
     ▼                                          ▼
Dividir en chunks                      LangGraph RAG Graph
(500 chars, overlap 100)                        │
     │                               ┌──────────▼──────────┐
     ▼                               │    retrieve_node    │
Convertir a vectores                 │  FAISS + rerank     │
(384 floats por chunk)               └──────────┬──────────┘
     │                                          │
     ▼                               ┌──────────▼──────────┐
Guardar en FAISS ◄──────────────►   │   evaluate_node     │
(index.faiss)                        │  confidence check   │
+ metadata.pkl                       └────┬────────────────┘
+ vectors.npy                             │
                               ┌──────────┴──────────┐
                          baja conf.            conf. ok
                               │                     │
                    ┌──────────▼──────┐    ┌─────────▼────────┐
                    │ reformulate_node│    │  generate_node   │
                    │ Gemini reescribe│    │  LLM local genera│
                    │ la query        │    └─────────┬────────┘
                    └──────────┬──────┘              │
                               │ (vuelve a retrieve) │
                                                     ▼
                                           ┌─────────────────┐
                                           │   review_node   │
                                           │ Gemini evalúa:  │
                                           │ anclaje + citas │
                                           └────┬────────────┘
                                                │
                                   ┌────────────┴──────────┐
                                aprobado               rechazado
                                   │                       │
                                  END             ┌────────▼────────┐
                                                  │  correct_node   │
                                                  │ LLM local corr. │
                                                  └────────┬────────┘
                                                           │
                                                      review_node
                                                   (loop acotado)
```

La idea central que lo une todo: **el significado del texto se convierte en coordenadas en un espacio matemático**. Buscar información relevante es encontrar qué coordenadas están más cerca de la coordenada de tu pregunta. Todo lo demás — el grafo, los proveedores, el reviewer — es infraestructura alrededor de esa idea.

---

## Qué tiene este sistema que un RAG básico no tiene

Un RAG mínimo es: embed pregunta → buscar FAISS → construir prompt → llamar LLM. Eso funciona. Este sistema va más lejos en tres direcciones:

**Adaptive retrieval** — antes de generar, evalúa si lo que recuperó es suficientemente relevante. Si no, reformula la pregunta automáticamente y vuelve a buscar. Implementado como grafo de estados con LangGraph.

**Multi-provider con responsabilidad separada** — el modelo local siempre genera la respuesta final (tus documentos no salen de tu máquina). Gemini se usa solo para tareas cortas y sin contexto sensible: reformular queries y revisar respuestas. Arquitectura de proveedores extensible vía registro en `providers.py`.

**Reviewer automático** — después de generar, Gemini evalúa si la respuesta está anclada en los chunks recuperados y si cita sus fuentes. Si no, el modelo local recibe el feedback y corrige. Loop acotado por `MAX_REVIEW_ATTEMPTS`.

---

## Plan de Ruta

Los módulos 1 a 3 no cambiaron con el refactor LangGraph. Los módulos 4 y 5 sí tienen cambios relevantes (se documentan). Los módulos 7 y 8 son completamente nuevos.

### Módulo 1 — Ingesta y Embeddings
`src/ingest/core.py` · `src/ingest/ingest.py` · `src/ingest/web_ingest.py` · `src/ingest/web_crawler.py`

Cómo el sistema convierte texto plano en vectores numéricos. Qué es un embedding, por qué dividimos en chunks, qué significa `normalize_embeddings=True`. **Sin cambios desde la versión original.**

### Módulo 2 — Almacenamiento Vectorial
`src/context/storage.py` · `src/ingest/core.py` (funciones de persistencia)

Por qué guardamos tres archivos (`index.faiss`, `metadata.pkl`, `vectors.npy`), qué rol cumple cada uno, y cómo FAISS hace la búsqueda. **Sin cambios desde la versión original.**

### Módulo 3 — Gestión de Contextos
`src/context/manager.py` · `src/context/selector.py` · `src/context/models.py`

Cómo el sistema organiza múltiples colecciones independientes y las carga/descarga de memoria. **Sin cambios desde la versión original.**

### Módulo 4 — Recuperación Semántica *(actualizado)*
`src/retrieval/search.py`

El corazón del RAG: FAISS search, similitud coseno, multi-query en modo SOFT, rerank. **Cambios:** `search()` ya no recibe `chat_memory` (el historial migró a los mensajes de API), modos renombrados a HARD/SOFT, 3 queries en SOFT en lugar de 4.

### Módulo 5 — Generación y Prompts *(actualizado)*
`src/prompts/builder.py` · `src/llm/generate.py`

Cómo se construye el prompt y cómo se llama al LLM. **Cambios:** `builder.py` tiene tres funciones en vez de una (`build_prompt`, `build_review_prompt`, `build_correction_prompt`). El historial ya no va en el texto del prompt — viaja como mensajes de API. `client.py` fue eliminado y reemplazado por `providers.py`.

### Módulo 6 — Sesión, Modos y CLI *(actualizado)*
`src/chat/session.py` · `src/chat/interface.py` · `src/chat/modes.py`

Cómo el estado de la sesión mantiene todo cohesionado. **Cambios:** modos renombrados a HARD/SOFT, `interface.py` ahora delega en el grafo LangGraph en lugar de llamar directamente a `search()` y `ask_llm()`.

### Módulo 7 — El Grafo LangGraph *(nuevo)*
`src/graph/graph.py` · `src/graph/nodes.py` · `src/graph/state.py`

Cómo funciona LangGraph, qué es `RAGState`, cómo se conectan los nodos, y por qué el routing condicional reemplaza el `if/else` explícito que tenía `interface.py`. El módulo más importante para hablar del proyecto en una entrevista técnica.

### Módulo 8 — Multi-Proveedor y Reviewer *(nuevo)*
`src/llm/providers.py` · `src/llm/generate.py` · `src/graph/nodes.py` (review_node, correct_node)

El registro de proveedores LLM, la arquitectura provider-per-node, y cómo funciona el ciclo review → correct. Por qué el modelo local siempre genera la respuesta final. Decisiones de diseño relevantes para entrevista.
