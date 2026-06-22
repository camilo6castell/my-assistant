# Módulo 4 — Recuperación Semántica

Este es el módulo donde todo lo anterior converge. Tienes contextos cargados en memoria (Módulo 3), cada uno con su índice FAISS (Módulo 2), construido a partir de chunks vectorizados (Módulo 1). Ahora: llega una pregunta del usuario. ¿Qué pasa exactamente?

```python
def search(
    question: str,
    mode: str,
    chat_memory: list[TurnMemory],
    collections: list[LoadedCollection],
) -> tuple[list[SearchResult], float]:
```

Esta única función es la frontera entre "tengo una pregunta en texto" y "tengo los fragmentos relevantes para responderla". Vamos a desarmarla en sus 4 pasos internos.

---

## Paso 1 — `build_queries()`: una pregunta, ¿o varias?

```python
def build_queries(
    question: str,
    mode: str,
    memory: list[TurnMemory],
) -> list[str]:

    if mode == ChatMode.RIGOROUS:
        return [question]

    history = " ".join(
        f"{turn['user']} {turn['assistant']}" for turn in memory[-MAX_TURNS:]
    )

    return [
        question,
        f"{history} {question}".strip(),
        f"Explica el concepto: {question}",
        f"Relaciona ideas sobre: {question}",
    ]
```

**Modo RIGUROSO:** una sola query, la pregunta tal cual. Simple y directo.

**Modo INTERPRETATIVO:** cuatro variantes de la misma pregunta. Esto es la técnica llamada **multi-query retrieval**. ¿Por qué generar 4 versiones?

Piénsalo así: si preguntas "¿qué relación hay entre el espectáculo y el inconsciente?", una sola búsqueda vectorial encontrará chunks que hablen literalmente de "espectáculo" e "inconsciente" juntos. Pero quizás el documento de Freud tiene un párrafo brillante sobre represión que es *conceptualmente* relevante sin usar esas palabras exactas.

Cada variante apunta el "rayo de búsqueda" en una dirección ligeramente distinta del espacio semántico:

```
"question"                          → busca la pregunta literal
"{history} {question}"              → incorpora contexto conversacional previo
"Explica el concepto: {question}"   → sesga hacia definiciones/explicaciones
"Relaciona ideas sobre: {question}" → sesga hacia conexiones entre temas
```

Es como lanzar 4 redes de pesca en lugares distintos del mismo lago en lugar de una sola red — más probabilidad de capturar algo relevante que la búsqueda literal sola no encontraría.

El costo: 4x el trabajo de embedding y búsqueda. Por eso solo se activa en modo INTERPRETATIVO — el modo RIGUROSO prioriza velocidad y precisión literal.

---

## Paso 2 — `encode_queries()`: texto → vectores

```python
def encode_queries(queries: list[str]) -> np.ndarray:
    embeddings = model.encode(queries, normalize_embeddings=True)
    return np.ascontiguousarray(embeddings, dtype=np.float32)
```

Exactamente la misma operación que vimos en el Módulo 1 para los chunks — el mismo modelo `BAAI/bge-small-en-v1.5`, los mismos 384 floats normalizados. Esto es crítico: **la pregunta y los documentos deben pasar por el mismo modelo de embeddings**. Si usaras un modelo distinto para indexar y otro para consultar, los vectores vivirían en "espacios" diferentes y las distancias no significarían nada — sería como comparar coordenadas GPS con coordenadas de un videojuego.

```
queries = ["pregunta original", "pregunta + historial", "Explica...", "Relaciona..."]
                    │
                    ▼
        encode_queries()
                    │
                    ▼
        matriz (4, 384)  ← 4 vectores, uno por variante
```

`np.ascontiguousarray` asegura que la memoria esté organizada de forma contigua — un requisito de los stubs de FAISS que ya viste en el módulo de tipado.

---

## Paso 3 — `retrieve()`: la búsqueda real

Aquí ocurre el doble loop más importante del sistema:

```python
def retrieve(
    query_embeddings: np.ndarray,
    collections: list[LoadedCollection],
    top_k_initial: int,
) -> list[SearchResult]:

    results: list[SearchResult] = []

    for collection in collections:
        index = collection["index"]

        if index is None:
            continue

        metadata = collection["metadata"]
        collection_name = collection["collection_name"]

        for q_emb in query_embeddings:
            query = np.ascontiguousarray([q_emb], dtype=np.float32)
            scores, indices = index.search(query, top_k_initial)

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                item = metadata[idx]

                results.append(
                    SearchResult(
                        score=float(score),
                        text=item["text"],
                        source=item["source"],
                        page=item["page"],
                        collection=collection_name,
                        chunk_index=item["chunk_index"],
                    )
                )

    return results
```

Desenrollemos esto con un ejemplo concreto. Supongamos:
- 2 colecciones activas: `sociologia/debord`, `psicoanalisis/freud`
- Modo INTERPRETATIVO → 4 query embeddings
- `top_k_initial = 25` (modo interpretativo busca más candidatos)

```
Loop externo (colecciones):
  ├─ sociologia/debord
  │     Loop interno (4 queries):
  │       ├─ query 1 → index.search() → 25 resultados
  │       ├─ query 2 → index.search() → 25 resultados
  │       ├─ query 3 → index.search() → 25 resultados
  │       └─ query 4 → index.search() → 25 resultados
  │
  └─ psicoanalisis/freud
        Loop interno (4 queries):
          ├─ query 1 → index.search() → 25 resultados
          ├─ query 2 → index.search() → 25 resultados
          ├─ query 3 → index.search() → 25 resultados
          └─ query 4 → index.search() → 25 resultados

Total: 2 colecciones × 4 queries × 25 resultados = 200 SearchResult
```

`index.search(query, top_k_initial)` es la llamada que dispara la búsqueda FAISS que discutimos en el Módulo 2 — calcula el producto interno (similitud coseno) entre `query` y cada vector del índice, y devuelve los `top_k_initial` más altos.

**`scores, indices = index.search(...)`** retorna dos arrays paralelos:

```python
scores  = [[0.89, 0.85, 0.81, ...]]   # similitudes, de mayor a menor
indices = [[47,   203,  891,  ...]]   # posiciones en metadata
```

`idx == -1` es el caso borde: si el índice tiene menos vectores que `top_k_initial` solicitados, FAISS rellena con `-1` para completar el array. Se descarta.

`metadata[idx]` es el momento exacto donde el número puro (`idx = 47`) se convierte en información legible (texto, fuente, página) — la reconexión que mencionamos en el Módulo 2.

---

## Paso 4 — `rerank()`: de 200 candidatos a los mejores, sin duplicados

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

Dos operaciones:

**1. `sort(reverse=True)`** — ordena los 200 resultados por `score` descendente. El más similar primero.

**2. Deduplicación con `set` de tuplas** — esto es necesario porque, recordando el ejemplo anterior, **el mismo chunk puede aparecer múltiples veces**. Si las 4 variantes de query en modo interpretativo son semánticamente parecidas, es muy probable que el chunk 47 de `sociologia/debord` aparezca en los resultados de la query 1 *y* de la query 3.

La clave de deduplicación es la tupla `(collection, source, chunk_index)` — el identificador único de un chunk en todo el sistema (recuerda: la posición es la clave, como vimos en Módulo 2, pero aquí necesitamos también `collection` porque puede haber chunks con el mismo `chunk_index` en colecciones distintas).

Como la lista ya está ordenada por score antes de deduplicar, **la primera ocurrencia de cada chunk es siempre la de mayor score** — así que el `dedup` se queda con la mejor versión de cada chunk único.

---

## El cierre: de 200 a 7 (o 5)

```python
def search(...) -> tuple[list[SearchResult], float]:

    if mode == ChatMode.INTERPRETATIVE:
        top_k_initial = INTERPRETATIVE_TOP_K_INITIAL   # 25
        top_k_final = INTERPRETATIVE_TOP_K_FINAL       # 7
    else:
        top_k_initial = BASE_TOP_K_INITIAL             # 15
        top_k_final = BASE_TOP_K_FINAL                 # 5

    queries = build_queries(question, mode, chat_memory)
    embeddings = encode_queries(queries)
    results = rerank(retrieve(embeddings, collections, top_k_initial))
    final_results = results[:top_k_final]

    if not final_results:
        return [], 0.0

    confidence = sum(r.score for r in final_results) / len(final_results)

    return final_results, confidence
```

`results[:top_k_final]` — slicing simple, te quedas con los N mejores después de ordenar y deduplicar.

**`confidence`** es el promedio de los scores de los resultados finales. Es lo que viste en el chat como `[confidence: 0.8280]`. No es una probabilidad estadística rigurosa — es una heurística simple: si los chunks que vas a usar tienen scores altos, el contexto recuperado probablemente es relevante; si son bajos, probablemente la pregunta no tenía buena cobertura en los documentos cargados.

---

## El flujo completo de extremo a extremo

```
"¿Cómo se relacionan el espectáculo y el inconsciente?"
         │
         ▼  build_queries() [modo INTERPRETATIVO]
4 variantes de la pregunta
         │
         ▼  encode_queries()
matriz (4, 384) — 4 vectores
         │
         ▼  retrieve() — doble loop
2 colecciones × 4 queries × 25 = hasta 200 SearchResult
         │
         ▼  rerank() — sort + dedup por (collection, source, chunk_index)
~120 SearchResult únicos, ordenados por score
         │
         ▼  [:7]
7 SearchResult finales
         │
         ▼
(resultados, confidence=0.74)
```

Estos 7 `SearchResult` son lo que `interface.py` convierte en `context_chunks` para construir el prompt — que es exactamente el Módulo 5.

---

## `vectorstore.py` — ¿qué queda ahí?

Vale la pena mencionarlo brevemente porque puede parecer que falta:

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def normalize_embedding(embedding: np.ndarray | list[float]) -> np.ndarray:
    arr = np.array(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return np.asarray(arr / norm, dtype=np.float32)
```

Estas son utilidades de propósito general que **no se usan en el flujo principal actual** — `search.py` delega toda la normalización a `model.encode(normalize_embeddings=True)` y toda la similitud a FAISS internamente. Este archivo existe como caja de herramientas reutilizable (por ejemplo, si en el futuro quisieras comparar dos embeddings manualmente sin pasar por FAISS).

---

¿Avanzamos al Módulo 5 — cómo estos `SearchResult` se convierten en el prompt final que recibe el LLM?
