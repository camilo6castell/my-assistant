# Módulo 4 — Recuperación Semántica *(actualizado)*

Este módulo documenta el estado actual de `src/retrieval/search.py`. Si leíste la versión anterior, los cambios principales son: modos renombrados a HARD/SOFT, `search()` ya no recibe `chat_memory`, y son 3 queries en SOFT en vez de 4.

---

## El punto de entrada

```python
def search(
    question: str,
    mode: str,
    collections: list[LoadedCollection],
) -> tuple[list[SearchResult], float]:
```

Notar que `chat_memory` ya **no** es un parámetro. En la versión anterior, el historial viajaba como una variante más de query en modo INTERPRETATIVO. Eso causaba que cada turno de la conversación influyera en la búsqueda vectorial, lo cual en la práctica añadía ruido. El historial ahora viaja solo como mensajes de API en `build_messages()` — impacta al LLM al generar la respuesta, pero no al buscar los chunks. El comportamiento resultante es más predecible.

---

## Paso 1 — `build_queries()`: una pregunta, ¿o tres?

```python
def build_queries(question: str, mode: str) -> list[str]:
    if mode == ChatMode.HARD:
        return [question]

    return [
        question,
        f"Explica el concepto: {question}",
        f"Relaciona ideas sobre: {question}",
    ]
```

**Modo HARD:** una sola query, la pregunta literal. Para búsquedas factuales donde la precisión importa más que el recall.

**Modo SOFT:** tres variantes. Esta es la técnica de **multi-query retrieval** — lanzar el mismo intento de búsqueda desde ángulos semánticos distintos para ampliar el pool de candidatos.

```
"¿Qué dice Freud sobre los sueños?"

query 1: "¿Qué dice Freud sobre los sueños?"           → busca la pregunta literal
query 2: "Explica el concepto: ¿Qué dice Freud..."     → sesga hacia definiciones
query 3: "Relaciona ideas sobre: ¿Qué dice Freud..."   → sesga hacia conexiones
```

El costo: 3x el trabajo de embedding y búsqueda. Por eso solo se activa en SOFT.

---

## Paso 2 — `encode_queries()`: texto → vectores

```python
def encode_queries(queries: list[str]) -> np.ndarray:
    embeddings = model.encode(queries, normalize_embeddings=True)
    return np.ascontiguousarray(embeddings, dtype=np.float32)
```

El mismo modelo `BAAI/bge-small-en-v1.5` que se usó al ingestar los documentos. Esto es crítico: **la pregunta y los documentos deben pasar por el mismo modelo**. Si indexaste con modelo A y consultas con modelo B, los vectores viven en espacios distintos y las distancias no significan nada — como comparar coordenadas GPS con coordenadas de un mapa de fantasía.

`np.ascontiguousarray` garantiza que la memoria esté organizada de forma contigua — un requisito técnico de los bindings C++ de FAISS.

---

## Paso 3 — `retrieve()`: el doble loop

```python
def retrieve(
    query_embeddings: np.ndarray,
    collections: list[LoadedCollection],
    top_k_initial: int,
) -> list[SearchResult]:
```

Hay dos loops anidados: uno sobre las colecciones activas, otro sobre los embeddings de query.

Con 2 colecciones activas, modo SOFT (3 queries, `top_k_initial=25`):

```
sociologia/debord
    query 1 → index.search() → 25 candidatos
    query 2 → index.search() → 25 candidatos
    query 3 → index.search() → 25 candidatos

psicoanalisis/freud
    query 1 → index.search() → 25 candidatos
    query 2 → index.search() → 25 candidatos
    query 3 → index.search() → 25 candidatos

Total bruto: 150 SearchResult (con duplicados)
```

`index.search(query, top_k_initial)` devuelve dos arrays paralelos:

```python
scores  = [[0.89, 0.85, 0.81, ...]]   # similitudes coseno, de mayor a menor
indices = [[47,   203,  891,  ...]]   # posiciones en metadata[]
```

`idx == -1` es el caso borde: si el índice tiene menos vectores que `top_k_initial`, FAISS rellena con `-1`. Se descarta.

`metadata[idx]` reconecta el número puro con texto legible. En la versión actual, `metadata` contiene instancias de `ChunkMetadata` (Pydantic BaseModel), por lo que el acceso es por atributo:

```python
# versión actual:
text=item.text,
source=item.source,
page=item.page,

# versión anterior usaba dict:
text=item["text"],
```

---

## Paso 4 — `rerank()`: de 150 candidatos a los mejores sin duplicados

```python
def rerank(results: list[SearchResult]) -> list[SearchResult]:
    results.sort(key=lambda x: x.score, reverse=True)

    dedup: list[SearchResult] = []
    seen: set[tuple[str, str, int]] = set()

    for r in results:
        key = (r.collection, r.source, r.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    return dedup
```

**Sort primero, deduplica después.** El orden importa: como la lista ya está ordenada por score antes de deduplicar, la primera ocurrencia de cada chunk (la que queda) es siempre la de mayor score.

La clave de deduplicación es la tupla `(collection, source, chunk_index)` — el identificador único global de un chunk. Solo `chunk_index` no alcanza porque puede haber chunks con el mismo índice en colecciones distintas.

---

## El cierre: top_k_final y confidence

```python
final_results = results[:top_k_final]   # HARD: 5, SOFT: 7

confidence = sum(r.score for r in final_results) / len(final_results)
```

`confidence` es el promedio de similitud coseno de los chunks que llegan al prompt. Es una heurística simple, no una probabilidad estadística. El grafo LangGraph la usa en `evaluate_node` para decidir si reformular la query:

```
confidence < settings.confidence_limit  → reformulate_node (Gemini reescribe la query)
confidence >= settings.confidence_limit → generate_node
```

Lo que ves en el log `[confidence: 0.8021]` es este valor.

---

## Flujo completo

```
question (str)
     │
     ▼  build_queries(mode)
[q1, q2, q3]  (SOFT) / [q1]  (HARD)
     │
     ▼  encode_queries()
matriz (3, 384) de floats normalizados
     │
     ▼  retrieve(collections, top_k_initial)
       loop: colecciones × queries × FAISS.search()
       → lista cruda de SearchResult (con duplicados)
     │
     ▼  rerank()
       sort by score + dedup por (collection, source, chunk_index)
     │
     ▼  [:top_k_final]
final_results: list[SearchResult]  (5 o 7 chunks)
     │
     ▼
(final_results, confidence: float)
```

Estos chunks son lo que `generate_node` convierte en `context_chunks` para construir el prompt.
