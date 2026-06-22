# Módulo 1 — `ingest.py`: El Director de Orquesta

Antes de ver el código, el mapa mental de lo que hace este archivo:

```
DISCO                    MEMORIA                      DISCO
─────                    ───────                      ─────

Archivos          →   Leer texto    →   Chunkear   →   Vectorizar   →   Guardar
(PDF/TXT/HTML)        por páginas       (500 chars)    (embeddings)     en FAISS
```

No hay nada de IA aquí todavía — `ingest.py` es pura orquestación. Llama funciones de `core.py` en el orden correcto.

---

## La entrada al sistema

```python
def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python -m src.ingest.ingest <categoria> <coleccion>")
        return

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    collection: str = f"{category}/{collection_name}"
```

Cuando ejecutas:
```bash
python -m src.ingest.ingest sociologia debord
```

`collection` queda como `"sociologia/debord"`. Ese string es el identificador único de todo lo que sigue — determina en qué carpeta de `vector_stores/` se va a guardar todo.

---

## Paso 1 — Cargar lo que ya existe

```python
collection_data: RawCollection = load_collection(collection)
existing_sources: set[str] = {m["source"] for m in collection_data["metadata"]}
```

Antes de procesar cualquier archivo, el sistema pregunta: **¿qué ya está indexado?**

`load_collection` lee el `metadata.pkl` que ya existe en disco (si existe) y devuelve la colección actual. Luego se extrae un `set` con los nombres de todas las fuentes ya procesadas.

El `set` aquí no es casual — la búsqueda de pertenencia en un `set` es O(1). Si tienes 10.000 chunks de 500 archivos y quieres saber si el archivo 501 ya fue procesado, no recorres los 10.000 metadatos: buscas en el set de 500 fuentes instantáneamente.

**¿Qué pasa si es la primera vez?** `load_collection` no encuentra nada en disco y devuelve una `RawCollection` vacía: `index=None`, `metadata=[]`, `vectors=None`. El sistema arranca desde cero sin errores.

---

## Paso 2 — Leer los archivos

```python
for file in files:
    if file.name in existing_sources:
        logger.info(f"Omitiendo ya indexado: {file.name}")
        continue

    pages: list[tuple[int, str]] = read_file(file)
```

El guard `if file.name in existing_sources` es **idempotencia** — puedes correr el ingest 10 veces sobre la misma carpeta y el resultado es siempre el mismo. No hay duplicados. Esto importa en producción cuando agregas un archivo nuevo a una colección que ya tiene 50.

`read_file` despacha a tres funciones según extensión:

```python
def read_file(path: Path) -> list[tuple[int, str]]:
    suffix: str = path.suffix.lower()

    if suffix == ".pdf":   return read_pdf(path)
    if suffix == ".html":  return read_html(path)
    if suffix == ".txt":   return read_txt(path)
```

Todas retornan el mismo tipo: `list[tuple[int, str]]` — una lista de pares `(número_de_página, texto)`. Esta interfaz uniforme es lo que permite que el resto del pipeline no sepa ni le importe si el origen era un PDF o un TXT.

**¿Por qué guardar el número de página?**

```python
SearchResult(
    score=...,
    text=...,
    source="Debord.pdf",
    page=47,          # ← esto
    ...
)
```

Cuando el RAG responde, puede decirte exactamente en qué página encontró la información. Trazabilidad.

---

## Paso 3 — Construir los chunks y su metadata

Este es el paso más importante de entender en detalle:

```python
for page_number, text in pages:
    chunks: list[str] = chunk_text(text)

    for i, chunk in enumerate(chunks):
        new_chunks.append(chunk)
        new_metadata.append(
            build_metadata(
                source=file.name,
                source_type="file",
                page=page_number,
                chunk=chunk,
                chunk_index=i,
                collection=collection,
            )
        )
```

Aquí se construyen **dos listas en paralelo** que deben mantenerse sincronizadas:

```
new_chunks    = ["texto del chunk 0",  "texto del chunk 1",  "texto del chunk 2",  ...]
new_metadata  = [{source, page, i=0},  {source, page, i=1},  {source, page, i=2},  ...]
               ↑ índice 0              ↑ índice 1              ↑ índice 2
```

El índice de posición en ambas listas es el contrato que une el vector con su origen. Cuando FAISS te devuelva "el vector en la posición 47 es el más relevante", irás a `metadata[47]` para saber de qué archivo y página vino ese chunk.

**Esta sincronización es la invariante más crítica del sistema.** Si en algún momento se desincroniza (por ejemplo, agregando a una sin la otra), el RAG devolvería texto de una fuente pero la metadata de otra. El `vacuum_collection` en `delete.py` existe precisamente para reparar esta sincronización si se corrompe.

---

## Paso 4 — Vectorizar y guardar

```python
embeddings: np.ndarray = encode_chunks(new_chunks)

save_collection(
    collection_data=collection_data,
    new_embeddings=embeddings,
    new_metadata=new_metadata,
)
```

`encode_chunks` convierte la lista de strings en una matriz numpy de shape `(N, 384)`. Luego `save_collection` hace el merge con lo que ya existía:

```python
# Dentro de save_collection en core.py:
if existing_vectors is not None:
    all_vectors = np.vstack([existing_vectors, new_embeddings])
else:
    all_vectors = new_embeddings
```

`np.vstack` es un `append` de matrices — apila las nuevas filas debajo de las existentes:

```
existing_vectors  shape (1000, 384)   ← vectores que ya estaban
new_embeddings    shape (200,  384)   ← vectores nuevos
                  ─────────────────
all_vectors       shape (1200, 384)   ← resultado
```

Y el FAISS index recibe solo los vectores nuevos (no el total), porque el índice ya contiene los anteriores y solo necesita los que se agregan:

```python
index.add(_to_f32(new_embeddings))  # solo los nuevos
```

---

## El flujo completo de una sola ejecución

```
python -m src.ingest.ingest sociologia debord
           │
           ▼
   collection = "sociologia/debord"
           │
           ▼
   ¿Existe ya? → cargar estado actual
           │
           ▼
   Para cada archivo en data/:
     ├─ ¿Ya indexado? → skip
     └─ Leer páginas
           │
           ▼
   Para cada página:
     └─ chunk_text() → ["chunk0", "chunk1", ...]
           │
           ▼
   build_metadata() para cada chunk
   (dos listas paralelas, mismo índice)
           │
           ▼
   encode_chunks() → matriz (N, 384)
           │
           ▼
   save_collection():
     ├─ vstack con vectores existentes → vectors.npy
     ├─ extend metadata existente    → metadata.pkl
     └─ index.add() nuevos vectores  → index.faiss
```

---

`web_ingest.py` y `web_crawler.py` siguen exactamente el mismo flujo, con una única diferencia: en lugar de `read_file(path)`, llaman a `extract_main_content(url)` que descarga el HTML, extrae el texto principal con la librería `readability`, y lo devuelve como `[(1, texto)]`. El resto del pipeline es idéntico.

¿Avanzamos al Módulo 2 — los tres archivos que guarda el sistema y por qué FAISS es una estructura de datos especial — o quieres que revisemos algo de este módulo?
