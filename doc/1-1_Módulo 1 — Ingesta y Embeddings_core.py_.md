# Módulo 1 — Ingesta y Embeddings: `core.py`

> Este documento explica el corazón del pipeline de ingesta: cómo un documento
> se transforma en fragmentos indexables, cómo se generan los embeddings y cómo
> se construye la metadata que accompany a cada chunk.

---

## 1. ¿Por qué existe `core.py`?

Antes de hablar de código, pensemos en el problema:

Tienes un PDF de 200 páginas. Quieres poder hacer preguntas sobre su contenido.
No puedes buscar en el documento entero de una vez — los modelos de lenguaje
tienen una ventana de contexto limitada. Tampoco puedes buscar en páginas
completas — una página puede tener 500 palabras que hablan de 3 temas distintos.

La solución: **dividir el texto en fragmentos pequeños** (chunks), convertir cada
fragmento en un **vector numérico** (embedding) que captura su significado
semántico, y almacenar esos vectores en un **índice** que permita búsquedas
por similitud.

`core.py` hace exactamente eso, con una responsabilidad clara y acotada:

> **Divide el texto. Genera los embeddings. Construye la metadata.**
> Todo lo demás (persistencia, orquestación) vive en otros módulos.

---

## 2. Las 3 funciones

`core.py` tiene solo 3 funciones. Cada una hace una cosa:

```
chunk_text(text)  →  [fragmento1, fragmento2, ...]

encode_chunks(chunks)  →  np.ndarray (matriz de embeddings)

build_metadata(source, ...)  →  ChunkMetadata (Pydantic model)
```

Veamos cada una en detalle.

---

## 3. `chunk_text()` — El cortador de texto

```python
# src/ingest/core.py:28


def chunk_text(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + settings.chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += settings.chunk_size - settings.chunk_overlap

    return chunks
```

### Cómo funciona

Imagina que tienes una cinta adhesiva larga (el texto) y una regla de
`chunk_size` caracteres. Cortas un pedazo, avanzas `chunk_size - chunk_overlap`
caracteres, y cortas otro. El `chunk_overlap` asegura que no se pierda
información en los bordes entre fragmentos.

**Ejemplo** con `chunk_size=500`, `chunk_overlap=100`:

```
Texto: "ABC...Z" (2000 caracteres)

Fragmento 1: caracteres [0, 500]
Fragmento 2: caracteres [400, 900]    ← empieza 100 antes del final del anterior
Fragmento 3: caracteres [800, 1300]
Fragmento 4: caracteres [1200, 1700]
Fragmento 5: caracteres [1600, 2000]
```

### Configuración

Los valores vienen de `settings` (leídos de `.env`):

| Parámetro | Default | Env Var |
|-----------|---------|---------|
| `chunk_size` | 500 | `CHUNK_SIZE` |
| `chunk_overlap` | 100 | `CHUNK_OVERLAP` |

La validación en `Settings` garantiza que `chunk_overlap < chunk_size`
para evitar un bucle infinito.

---

## 4. `encode_chunks()` — El traductor a lenguaje de máquinas

```python
# src/ingest/core.py:54


def encode_chunks(chunks: list[str]) -> np.ndarray:
    logger.info(
        f"Generating embeddings for {len(chunks)} chunks | "
        f"backend={settings.embedding_backend} | model={settings.embedding_model}"
    )
    return get_encoder().encode(chunks)
```

### ¿Qué es un embedding?

Un embedding es un **vector de números** (lista de 384, 768, 1024...
dependiendo del modelo) que representa el significado semántico del texto.
Dos textos con significado similar tendrán vectores cercanos en el espacio
de embeddings.

```
"Freud habló del inconsciente"  →  [0.12, -0.34, 0.56, ...]  (768 números)
"El inconsciente según Freud"   →  [0.11, -0.33, 0.57, ...]  (cercano al anterior)

"Los gatos duermen mucho"       →  [-0.78, 0.22, -0.45, ...]  (lejano)
```

### El encoder es pluggable

`encode_chunks()` es un wrapper delgada. Toda la lógica de backend vive
en `src/nlp/embedders/encoder.py`. La función `get_encoder()` resuelve
cuál usar basándose en la variable `EMBEDDER` de `.env`:

```
EMBEDDER=ollama,bge-m3          → HttpEmbeddingEncoder (Ollama via HTTP)
EMBEDDER=sentence_transformers,... → SentenceTransformersEncoder (local)
EMBEDDER=flm,bge-small-en-v1.5   → HttpEmbeddingEncoder (FastFlowLM via HTTP)
```

`get_encoder()` está decorada con `@lru_cache(maxsize=1)`: se construye
una sola vez por proceso. No se recarga en cada llamada a `encode_chunks()`.

### Normalización

Todos los backends devuelven vectores **L2-normalizados** y en `float32`.
Esto garantiza que `FAISS IndexFlatIP` (producto interno) produzca
similitud coseno correcta sin importar qué backend se use.

```python
# src/nlp/embedders/encoder.py:70
def encode(self, texts: Sequence[str]) -> np.ndarray:
    raw = self._encode_raw(texts)
    arr = np.ascontiguousarray(raw, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # evita división por cero
    return cast(NDArray[np.float32], arr / norms)
```

### Los 3 backends

| Backend | Tipo | Hardware | Ejemplo de configuración |
|---------|------|----------|--------------------------|
| `sentence_transformers` | In-process (Python) | CPU / GPU si torch lo soporta | `EMBEDDER=sentence_transformers,bge-small-en-v1.5` |
| `ollama` | HTTP (vía Ollama) | GPU Vulkan (AMD) | `EMBEDDER=ollama,bge-m3` |
| `flm` | HTTP (vía FastFlowLM) | NPU | `EMBEDDER=flm,bge-small-en-v1.5` |

Los backends HTTP (`ollama` y `flm`) usan el **mismo cliente OpenAI**
porque ambos exponen `/v1/embeddings` compatible con la API de OpenAI.
La diferencia es solo la URL base (`EMBEDDER_OLLAMA_URL` / `EMBEDDER_FLM_URL`).

### Agregar un nuevo backend

Subclase `EmbeddingEncoder` e implementa `_encode_raw()`:

```python
class MiNuevoEncoder(EmbeddingEncoder):
    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        # tu lógica aquí
        return embeddings


# Registrar en el factory
_BACKEND_MAP["mi_backend"] = MiNuevoEncoder
```

Ningún otro módulo necesita cambios — `core.py` y `search.py` solo
llaman a `get_encoder().encode()`.

---

## 5. `build_metadata()` — El casillero de identificación

```python
# src/ingest/core.py:67


def build_metadata(
    source: str,
    source_type: str,
    page: int,
    chunk: str,
    chunk_index: int,
    collection: str,
    file_id: str | None = None,
) -> ChunkMetadata:
    return ChunkMetadata(
        source=source,
        source_type=source_type,
        page=page,
        text=chunk,
        chunk_index=chunk_index,
        collection=collection,
        file_id=file_id,
    )
```

### ¿Para qué sirve?

Cuando un usuario hace una pregunta y el sistema encuentra un chunk relevante,
necesita saber: ¿de dónde viene este texto? ¿Qué página? ¿En qué colección
está? `build_metadata()` construye ese "casillero de identificación" que
acompaña a cada chunk.

### `ChunkMetadata` — Modelo Pydantic frozen

```python
# src/storage/faiss_store.py:47


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str  # Nombre del archivo o URL
    source_type: str  # "file", "url"
    page: int  # Número de página
    text: str  # Texto del fragmento
    chunk_index: int  # Índice del chunk dentro del archivo/URL
    collection: str  # Colección a la que pertenece (e.g. "sociologia/1984")
    file_id: str | None  # Solo para archivos efímeros (None en ingesta normal)
```

**`frozen=True`**: una vez creado, el objeto no se puede modificar.
Esto es intencional — los chunks no deben mutarse después de la ingesta.
Además, habilita hashing (útil para deduplicación).

`ChunkMetadata` se importa en `core.py` desde `src/storage/faiss_store.py`
y se re-exporta para compatibilidad:

```python
from src.storage.faiss_store import ChunkMetadata  # re-export for backward compatibility
```

---

## 6. ¿Qué NO hace `core.py`?

Es igual de importante saber qué **no** está aquí:

| Funcionalidad | Dónde vive |
|---------------|-----------|
| Persistencia FAISS (load/save) | `src/storage/faiss_store.py` |
| Lógica de archivos (PDF, HTML, TXT) | `src/ingest/ingest.py` |
| Descarga y extracción de URLs | `src/ingest/web_ingest.py`, `http.py` |
| Configuración (chunk_size, etc.) | `src/config/settings.py` |
| El encoder concreto | `src/nlp/embedders/encoder.py` |

Esta separación permite que cada módulo sea testeado y modificado
independientemente. `core.py` no sabe si el texto viene de un PDF o
de una URL. No sabe si el encoder es Ollama o sentence_transformers.
No sabe dónde se guarda el resultado.

---

## 7. Flujo completo de ingesta (una mirada de pájaro)

```
Documento / URL
    │
    ▼
    ┌──────────────────────────────────────────────┐
    │  chunk_text(text)                             │
    │  → ["fragmento1", "fragmento2", ...]          │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  encode_chunks(chunks)                        │
    │  → get_encoder().encode(chunks)               │
    │  → np.ndarray (matriz N×768)                  │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  build_metadata(source, ...)                  │
    │  → ChunkMetadata(source="...", page=1, ...)  │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼
           save_collection()  ← src/storage/faiss_store.py
```

---

## 8. Ejemplo práctico

```python
from src.ingest.core import chunk_text, encode_chunks, build_metadata

# 1. Texto de entrada
text = "Freud desarrolló la teoría del inconsciente..."  # 2000 chars

# 2. Dividir en chunks
chunks = chunk_text(text)
print(f"Se generaron {len(chunks)} fragments")

# 3. Generar embeddings
embeddings = encode_chunks(chunks)
print(f"Shape: {embeddings.shape}")  # e.g. (5, 768)

# 4. Construir metadata para cada chunk
for i, chunk in enumerate(chunks):
    meta = build_metadata(
        source="freud_intro.pdf",
        source_type="file",
        page=1,
        chunk=chunk,
        chunk_index=i,
        collection="psicologia/freud",
    )
    print(f"Chunk {i}: {meta.source} → p.{meta.page}")
```

---

## 9. Nota para el desarrollador

Si necesitas modificar la ingesta, piensa en qué capa afecta:

- **Cambiar cómo se divide el texto** → modifica `chunk_text()` en `core.py`
- **Cambiar el modelo de embeddings** → cambia `EMBEDDER` en `.env`
- **Agregar un formato de archivo** → agrega un `read_*()` en `ingest.py`
- **Cambiar la persistencia** → modifica `faiss_store.py`
- **Cambiar la metadata** → modifica `ChunkMetadata` en `faiss_store.py`
  y `build_metadata()` en `core.py`

La regla de oro: `core.py` es el punto donde **texto se convierte en
datos estructurados listos para FAISS**. Todo lo que entra por arriba
es texto crudo; todo lo que sale por abajo son embeddings + metadata.
