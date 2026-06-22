# RAG System — Mapa Arquitectónico y Plan de Ruta

## El flujo completo en una imagen

```
INGESTA (offline)                    CONSULTA (runtime)
─────────────────                    ──────────────────

 Archivo/URL                          Tu pregunta
     │                                     │
     ▼                                     ▼
 Dividir en chunks                   Convertir a vector
     │                                     │
     ▼                                     ▼
 Convertir a vectores            Buscar vectores similares
 (números que representan            en FAISS
  el significado del texto)          │
     │                               ▼
     ▼                          Tomar los chunks
 Guardar en FAISS ◄─────────►   más relevantes
 (index.faiss)                       │
 + metadata.pkl                      ▼
 + vectors.npy               Construir prompt
                              (contexto + pregunta)
                                     │
                                     ▼
                               Enviar al LLM
                                     │
                                     ▼
                                  Respuesta
```

La idea central que lo une todo: **el significado del texto se convierte en coordenadas en un espacio matemático**. Buscar información relevante es simplemente encontrar qué coordenadas están más cerca de la coordenada de tu pregunta. Todo lo demás es infraestructura alrededor de esa idea.

---

## Plan de Ruta

### Módulo 1 — Ingesta y Embeddings
`src/ingest/core.py` · `src/ingest/ingest.py` · `src/ingest/web_ingest.py` · `src/ingest/web_crawler.py`

Cómo el sistema convierte texto plano (PDFs, TXTs, URLs) en vectores numéricos que una máquina puede comparar. Aquí entenderás qué es un embedding, por qué dividimos el texto en chunks, y qué significa `normalize_embeddings=True`.

### Módulo 2 — Almacenamiento vectorial
`src/context/storage.py` · `src/ingest/core.py` (funciones de persistencia)

Por qué guardamos tres archivos (`index.faiss`, `metadata.pkl`, `vectors.npy`) en lugar de uno solo, qué rol cumple cada uno, y qué es FAISS internamente — explicado como una estructura de datos que ya conoces.

### Módulo 3 — Gestión de contextos
`src/context/manager.py` · `src/context/selector.py` · `src/context/models.py`

Cómo el sistema organiza múltiples colecciones independientes (sociología, psicoanálisis, etc.) y las carga/descarga de memoria en tiempo de ejecución. Por qué esto importa para el rendimiento.

### Módulo 4 — Recuperación semántica
`src/retrieval/search.py` · `src/retrieval/vectorstore.py`

El corazón del RAG: cómo se convierte tu pregunta en un vector, cómo FAISS hace la búsqueda, qué es la similitud coseno, y por qué el modo INTERPRETATIVO genera 4 queries en lugar de 1. Aquí está la diferencia entre un buscador de palabras clave y un buscador semántico.

### Módulo 5 — Generación y prompts
`src/prompts/builder.py` · `src/llm/generate.py` · `src/llm/client.py`

Cómo se ensambla el prompt final, por qué la estructura exacta del prompt importa, y qué viaja por el cable hacia tu LLM local (FastFlowLM/Ollama). El rol del historial de conversación (`chat_memory`) en el contexto.

### Módulo 6 — Sesión, modos y CLI
`src/chat/session.py` · `src/chat/interface.py` · `src/chat/modes.py` · `src/cli/`

Cómo el estado de la sesión mantiene todo cohesionado, qué cambia realmente entre modo RIGUROSO e INTERPRETATIVO a nivel de código, y cómo el loop de comandos conecta con todo lo anterior.

---

¿Apruebas este plan o quieres reorganizar algún módulo antes de empezar?
