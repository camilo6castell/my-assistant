# Módulo 4 — Recuperación Semántica

El Módulo 4 es el corazón del RAG: convierte una pregunta en texto en resultados concretos de chunks relevantes. Todo lo que viste hasta ahora — ingestar, almacenar, gestionar contextos — converge aquí.

---

## El punto de entrada

```python
def search(
    question: str,
    mode: str,
    collections: list[LoadedCollection],
    top_k_initial: int | None = None,
    top_k_final: int | None = None,
) -> tuple[list[SearchResult], float]:
```

`search()` es la función pública que el grafo LangGraph llama desde `retrieve_node`. Recibe la pregunta, el modo de búsqueda, y la lista de colecciones activas (el `loaded_contexts` del Módulo 3).

Los parámetros `top_k_initial` y `top_k_final` son opcionales — cuando llegan como `None` (el caso normal desde el API), se usan los valores configurados en `.env` según el modo:

```python
if mode == ChatMode.SOFT:
    default_initial = settings.soft_top_k_initial
    default_final = settings.soft_top_k_final
else:
    default_initial = settings.hard_top_k_initial
    default_final = settings.hard_top_k_final

top_k_initial = default_initial if top_k_initial is None else top_k_initial
top_k_final = default_final if top_k_final is None else top_k_final
```

¿Por qué dos valores de `top_k`? Porque la búsqueda funciona en dos etapas: primero recoge un amplio pool de candidatos (`top_k_initial`), luego recorta a los mejores (`top_k_final`). Esto es el patrón **retrieve-then-rerank**.

---

## `ChatMode` — HARD vs SOFT

```python
class ChatMode(StrEnum):
    SOFT = "SOFT"
    HARD = "HARD"
```

`ChatMode` es un `StrEnum` — hereda de `str` y de `Enum`, lo que significa que sus valores son strings válidos y puedes compararlos directamente con `==`. No es una clase con constantes de tipo string; es una enumeración con semantics.

- **HARD**: una sola query literal. Para búsquedas factuales donde la precisión importa más que el recall.
- **SOFT**: tres variantes semánticas. Para búsquedas exploratorias donde quieres ampliar el recall.

---

## Paso 1 — `build_queries()`: una pregunta, ¿o tres?

```python
def build_queries(question: str, mode: str) -> list[str]:
    if mode == ChatMode.HARD:
        return [question]

    return [
        question,
        f"Explain the concept: {question}",
        f"Relate ideas about: {question}",
    ]
```

**Modo HARD:** una sola query, la pregunta literal. Directo y preciso.

**Modo SOFT:** tres variantes. Esta es la técnica de **multi-query retrieval** — lanzar el mismo intento de búsqueda desde ángulos semánticos distintos para ampliar el pool de candidatos.

```
"¿Qué dice Freud sobre los sueños?"

query 1: "¿Qué dice Freud sobre los sueños?"           → busca la pregunta literal
query 2: "Explain the concept: ¿Qué dice Freud..."     → sesga hacia definiciones
query 3: "Relate ideas about: ¿Qué dice Freud..."      → sesga hacia conexiones
```

El costo: 3x el trabajo de embedding y búsqueda. Por eso solo se activa en SOFT.

> **Nota:** el historial de la conversación **no** viaja como una variante de query. Viaja como mensajes de API al LLM durante la generación. Esto evita que cada turno influya en la búsqueda vectorial — que en la práctica añadía ruido.

---

## Paso 2 — `encode_queries()`: texto → vectores

```python
def encode_queries(queries: list[str]) -> np.ndarray:
    return get_encoder().encode(queries)
```

El módulo importa `get_encoder` de `src/nlp/embedders/encoder.py`. Esta función devuelve una instancia cacheada del backend de embeddings configurado (sentence_transformers, Ollama, o FastFlowLM).

**Lo crítico**: la pregunta y los documentos deben pasar por el mismo modelo. Si indexaste con modelo A y consultas con modelo B, los vectores viven en espacios distintos y las distancias no significan nada — como comparar coordenadas GPS con coordenadas de un mapa de fantasía.

`get_encoder().encode()` retorna vectores L2-normalizados en float32, C-contiguos — un requisito técnico de los bindings C++ de FAISS para que la similitud coseno funcione correctamente.

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

`metadata[idx]` reconecta el número puro con texto legible. Como `ChunkMetadata` es un `BaseModel` de Pydantic, el acceso es por atributo:

```python
item = metadata[idx]
results.append(
    SearchResult(
        score=float(score),
        text=item.text,
        source=item.source,
        page=item.page,
        collection=collection_name,
        chunk_index=item.chunk_index,
    )
)
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

## Paso 5 — `format_context_chunks()`: de SearchResult a texto para el prompt

```python
def format_context_chunks(results: list[SearchResult]) -> list[str]:
    return [
        f"SOURCE: {r.source}\nCOLLECTION: {r.collection}\nPAGE: {r.page}\n\n{r.text}"
        for r in results
    ]
```

Convierte los `SearchResult` en strings formateados que el generador de prompts inserta directamente en el contexto. Cada chunk lleva encabezados de metadata (`FUENTE`, `COLECCIÓN`, `PÁGINA`) para que el LLM sepa de dónde viene cada fragmento.

```
SOURCE: freud_capitulo3.pdf
COLLECTION: psicoanalisis/freud
PAGE: 42

La interpretación de los sueños requiere considerar...
```

---

## El cierre: top_k_final y confidence

```python
final_results = results[:top_k_final]   # HARD: 5, SOFT: 7

if not final_results:
    return [], 0.0

confidence = sum(r.score for r in final_results) / len(final_results)
```

`confidence` es el promedio de similitud coseno de los chunks que llegan al prompt. Es una heurística simple, no una probabilidad estadística. El grafo LangGraph la usa en `evaluate_node` para decidir si reformular la query:

```
confidence < settings.confidence_limit  → reformulate_node (LLM reescribe la query)
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
matriz (N, 384) de floats normalizados
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
     ▼  format_context_chunks()
["SOURCE: ...\n\n{text}", "SOURCE: ...\n{text}", ...]
     │
     ▼
context_chunks → generate_node → prompt del LLM
```

¿Quieres ver cómo se almacenan archivos subidos por el usuario en memoria para usarlos como contexto efímero? Eso es el Módulo 4-2.
