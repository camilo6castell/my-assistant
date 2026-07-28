# Módulo 1 — `ingest.py`: El Director de Orquesta

> `ingest.py` es el módulo que coordina todo el proceso de ingesta: lee archivos
> del disco, delega el chunking y encoding a `core.py`, y guarda el resultado
> en FAISS. También cubre `web_ingest.py`, `web_crawler.py` y `http.py`.

---

## 1. El problema que resuelve

`core.py` sabe cortar texto y generar embeddings. Pero no sabe:
- Cómo leer un PDF
- Cómo extraer texto de un HTML
- Cómo orquestar el flujo completo de "archivo → chunks → embeddings → disco"

Eso es trabajo de `ingest.py`: **el director que coordina a los músicos**.
Si `core.py` es el instrumento, `ingest.py` es quien toca la partitura.

---

## 2. `ingest.py` — Ingesta de archivos locales

### Ejecución

```bash
python -m src.ingest.ingest <category> <collection>
```

Ejemplo:
```bash
python -m src.ingest.ingest psicologia freud
```

Esto crea/actualiza la colección `psicologia/freud` leyendo los archivos
de `/srv/ai/data/`.

### Flujo completo

```
Archivos en /srv/ai/data/
    │
    ▼
┌────────────────────────────────────────────────────┐
│  load_collection("psicologia/freud")               │
│  → RawCollection (faiss_store.py)                  │
│  → existing_sources = {m.source for m in metadata} │
└──────────────┬─────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────┐
│  Para cada archivo en data/:                        │
│                                                     │
│  ¿file.name in existing_sources?                   │
│    Sí → skip (ya indexado)                         │
│    No →继续:                                        │
│                                                     │
│    read_file(path) → [(page_num, text), ...]        │
│      ├── read_pdf()  → pypdf                        │
│      ├── read_html() → BeautifulSoup                │
│      └── read_txt()  → open()                       │
│                                                     │
│    Para cada página:                                │
│      chunks = chunk_text(text)                      │
│      Para cada chunk:                               │
│        build_metadata(source=file.name, ...)        │
└──────────────┬─────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────┐
│  encode_chunks(new_chunks)                          │
│  → embeddings np.ndarray                            │
└──────────────┬─────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────┐
│  save_collection(collection_data, embeddings, meta) │
│  → Persiste en disco                               │
└────────────────────────────────────────────────────┘
```

### Código clave

```python
# src/ingest/ingest.py:11-21 (imports)

from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
```

Observa que `load_collection` y `save_collection` se importan de
`src.storage.faiss_store`, no de `core.py`. `core.py` solo provee
las primitivas de procesamiento; la persistencia vive en `faiss_store.py`.

### Lectores de archivos

| Función | Formato | Librería | Retorna |
|---------|---------|----------|---------|
| `read_pdf()` | `.pdf` | `pypdf.PdfReader` | `[(page_num, text), ...]` |
| `read_html()` | `.html` | `BeautifulSoup` | `[(1, text)]` (1 página) |
| `read_txt()` | `.txt` | `open()` | `[(1, text)]` (1 página) |
| `read_file()` | Dispatcher | — | Llama al lector correcto según extensión |

Los números de página empiezan en 1 (no en 0) para alinearse con la
convención humana de conteo de páginas.

### Deduplicación

El sistema **no re-indexa archivos ya procesados**:

```python
# src/ingest/ingest.py:85
existing_sources: set[str] = {m.source for m in collection_data["metadata"]}

# src/ingest/ingest.py:98
if file.name in existing_sources:
    logger.info(f"Skipping already indexed: {file.name}")
    continue
```

`existing_sources` se construye leyendo los campos `source` de la metadata
ya persistida. Si un archivo tiene el mismo nombre, se asume que ya fue indexado.

---

## 3. `web_ingest.py` — Ingesta de una URL

### Ejecución

```bash
python -m src.ingest.web_ingest <category> <collection> <url>
```

Ejemplo:
```bash
python -m src.ingest.web_ingest web-articulos neurociencia https://ejemplo.com/articulo
```

### Flujo

```
URL
    │
    ▼
extract_main_content(url)     ← http.py (readability-lxml)
    │
    ▼
chunk_text(text)              ← core.py
    │
    ▼
build_metadata(source=url, source_type="url", page=1, ...)  ← core.py
    │
    ▼
encode_chunks(chunks)         ← core.py → encoder.py
    │
    ▼
save_collection(...)          ← faiss_store.py
```

### Código clave

```python
# src/ingest/web_ingest.py:7-18 (imports)

from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.ingest.http import extract_main_content
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
```

Diferencia clave con `ingest.py`: usa `extract_main_content()` en lugar
de `read_file()`. Toda la lógica de descarga HTTP y extracción de contenido
está centralizada en `http.py`.

### Deduplicación

```python
# src/ingest/web_ingest.py:34
if url in existing_sources:
    logger.warning("URL already indexed.")
    print("URL already indexed.")
    return
```

---

## 4. `web_crawler.py` — Crawl BFS dentro de un dominio

### Ejecución

```bash
python -m src.ingest.web_crawler <category> <collection> <start_url>
```

Ejemplo:
```bash
python -m src.ingest.web_crawler web-docs python https://docs.python.org/3/
```

### Flujo

```
start_url
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  BFS (Breadth-First Search)                           │
│                                                       │
│  to_visit = [start_url]                               │
│                                                       │
│  while to_visit and len(visited) < max_pages:         │
│    url = to_visit.pop(0)                              │
│    visited.add(url)                                   │
│                                                       │
│    text = extract_main_content(url)   ← http.py       │
│                                                       │
│    if text:                                           │
│      chunks = chunk_text(text)       ← core.py        │
│      new_chunks.extend(chunks)                        │
│      new_metadata.extend(build_metadata(...))         │
│                                                       │
│    for link in get_links(url, domain):                │
│      if link not in visited:                          │
│        to_visit.append(link)                          │
│                                                       │
│    time.sleep(settings.delay)  ← crawl礼节            │
│                                                       │
│  encode_chunks(new_chunks)           ← core.py        │
│  save_collection(...)                 ← faiss_store.py │
└──────────────────────────────────────────────────────┘
```

### Configuración

| Parámetro | Default | Env Var | Descripción |
|-----------|---------|---------|-------------|
| `max_pages` | 50 | `MAX_PAGES` | Máximo de páginas a crawlear |
| `delay` | 1.0 | `DELAY` | Segundos entre requests (cortesía) |

### Extracción de links

```python
# src/ingest/web_crawler.py:27
def get_links(url: str, domain: str) -> set[str]:
```

Solo sigue links **dentro del mismo dominio** (`parsed.netloc == domain`).
Esto evita que el crawler se salga del sitio web objetivo.

### Deduplicación

El crawler mantiene un `visited: set[str]` que evita procesar una URL
dos veces. Además, `load_collection()` carga las fuentes existentes
para no re-indexar contenido ya procesado.

---

## 5. `http.py` — Extracción de contenido web

### ¿Qué hace?

Cuando le pides al sistema que indexe una URL, necesitas extraer **solo el
contenido principal** de la página, no el menú, los anuncios, ni el footer.
`http.py` usa **readability-lxml** (la misma librería que usa Readability
de Firefox) para hacer eso.

```python
# src/ingest/http.py:18

def extract_main_content(url: str) -> str | None:
    try:
        logger.info(f"Downloading content: {url}")
        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        doc: Document = Document(response.text)       # readability-lxml
        soup: BeautifulSoup = BeautifulSoup(doc.summary(), "html.parser")
        text: str = soup.get_text(separator="\n")

        logger.info(f"Content extracted successfully: {url}")
        return text

    except requests.RequestException as e:
        logger.error(f"HTTP error on {url}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None
```

### Flujo visual

```
HTML crudo de la página
    │
    ▼
readability.Document(html)    ← Extrae el "article" principal
    │
    ▼
Document.summary()            ← HTML limpio (solo contenido)
    │
    ▼
BeautifulSoup(summary).get_text(separator="\n")
    │
    ▼
Texto plano limpio
```

### ¿Por qué readability-lxml?

Una página web típica tiene:
- 30% contenido real del artículo
- 70% nav, footer, sidebars, ads, scripts

Sin readability, el sistema indexaría todo el ruido junto con el contenido.
`readability-lxml` detecta automáticamente el "article" principal de la
misma forma que lo haría un navegador al "modo lectura".

### Compartido entre dos módulos

`http.py` es importado por:
- `web_ingest.py` — para indexar una URL individual
- `web_crawler.py` — para extraer contenido de cada página del crawl

Ambos usan la misma función, manteniendo la lógica de extracción
centralizada y sin duplicación.

---

## 6. Imports y dependencias — Mapa rápido

```
ingest.py
    ├── src.ingest.core          → chunk_text, encode_chunks, build_metadata
    ├── src.storage.faiss_store  → load_collection, save_collection, ChunkMetadata
    └── src.config.settings      → settings.data_path

web_ingest.py
    ├── src.ingest.core          → chunk_text, encode_chunks, build_metadata
    ├── src.ingest.http          → extract_main_content
    ├── src.storage.faiss_store  → load_collection, save_collection, ChunkMetadata
    └── (no necesita settings)

web_crawler.py
    ├── src.ingest.core          → chunk_text, encode_chunks, build_metadata
    ├── src.ingest.http          → extract_main_content
    ├── src.storage.faiss_store  → load_collection, save_collection, ChunkMetadata
    └── src.config.settings      → settings.max_pages, settings.delay

http.py
    ├── requests                → HTTP GET
    ├── readability.Document    → Extracción de contenido principal
    ├── bs4.BeautifulSoup       → Conversión a texto plano
    └── src.utils.logger        → logger
```

---

## 7. Diferencia clave: archivos locales vs URLs

| Aspecto | `ingest.py` | `web_ingest.py` / `web_crawler.py` |
|---------|-------------|--------------------------------------|
| Fuente | Archivos en `/srv/ai/data/` | URLs web |
| Lectura | `read_pdf()`, `read_html()`, `read_txt()` | `extract_main_content()` |
| `source_type` | `"file"` | `"url"` |
| Paginación | Real (número de página del PDF) | Siempre `page=1` |
| Múltiples páginas | Sí (PDF con N páginas) | No (contenido principal = 1 bloque) |
| Crawl | No | `web_crawler.py` hace BFS |

---

## 8. Nota sobre `ChunkMetadata` y `build_metadata()`

Tanto `ingest.py`, `web_ingest.py` como `web_crawler.py` usan la misma
función para construir metadata:

```python
build_metadata(
    source=...,        # nombre de archivo o URL
    source_type=...,   # "file" o "url"
    page=...,          # número de página
    chunk=...,         # texto del fragmento
    chunk_index=...,   # índice dentro del archivo/URL
    collection=...,    # "category/collection_name"
)
```

`build_metadata()` retorna un `ChunkMetadata` (modelo Pydantic frozen),
no un dict. Esto garantiza que:
- Los campos se validan en tiempo de construcción
- El objeto no se puede modificar después
- La serialización (pickle) es manejada por Pydantic

---

## 9. Ejemplo completo de uso

```bash
# 1. Ingesta de archivos locales
#    Asegúrate de que los archivos estén en /srv/ai/data/
python -m src.ingest.ingest psicologia freud

# 2. Ingesta de una URL
python -m src.ingest.web_ingest web-articulos neurociencia https://ejemplo.com/articulo

# 3. Crawl de un sitio web completo
python -m src.ingest.web_crawler web-docs python https://docs.python.org/3/
```

Después de estos comandos, las colecciones están listas para ser consultadas
desde la CLI o la API.

---

## 10. Resumen: las responsabilidades de cada archivo

| Archivo | Responsabilidad |
|---------|-----------------|
| `core.py` | Chunking + encoding + metadata (primitivas puras) |
| `ingest.py` | Orquesta ingesta de archivos locales |
| `web_ingest.py` | Orquesta ingesta de una URL |
| `web_crawler.py` | Orquesta crawl BFS de un dominio |
| `http.py` | Descarga HTTP + extracción de contenido principal |
| `faiss_store.py` | Persistencia FAISS (load/save/delete) |

La separación es clara: `core.py` no sabe de archivos ni URLs.
`ingest.py` no sabe cortar texto ni generar embeddings.
`http.py` no sabe de FAISS. Cada módulo hace una cosa y la hace bien.
