# Módulo 4-2 — Almacenamiento Efímero y Archivos Adjuntos

Los Módulos anteriores asumen que el contexto viene de colecciones persistidas en disco — archivos ingestados previamente con el CLI. Pero los usuarios del API también quieren subir un PDF o un fragmento de código **durante** la conversación, sin tener que correr un comando de ingestación. Este módulo explica cómo funciona ese sistema.

Hay dos almacenes distintos que resuelven problemas distintos:

| Almacén | ¿Dónde vive? | ¿Para qué? | ¿Dura? |
|---|---|---|---|
| `EphemeralStore` | RAM | Archivos indexados semánticamente para retrieval | Mientras viva la conversación |
| `AttachmentStore` | RAM | Archivos de texto inyectados raw en el prompt | Un solo envío |

---

## 1. EphemeralStore — Archivos indexados en memoria

**Archivo fuente:** `src/context/ephemeral.py`

### El problema que resuelve

Un usuario sube un PDF desde el frontend. Quiere que el sistema lo use como contexto semántico — es decir, que haga embedding de su contenido y lo busque vectorialmente junto con las colecciones existentes. Pero no quiere que ese archivo se guarde permanentemente en disco.

La diferencia con `ContextManager`: `ContextManager` carga colecciones que viven en disco bajo `vector_stores/`. Una colección efímera **nunca se escribe a disco** — vive solo en RAM, indexada por `conversation_id`. No sobrevive un reinicio del servidor.

### ¿Por qué no en el navegador?

Chunking + embeddings requieren `sentence_transformers` o FAISS corriendo en el backend Python, no algo viable en un navegador. El cliente solo sube el archivo; toda la computación pasa por el mismo pipeline que usa `ingest/ingest.py` (`chunk_text`, `encode_chunks`, `build_metadata`) — la única diferencia es que `save_collection()` nunca se llama.

### La estructura interna

```python
@dataclass
class _ConversationStore:
    metadata: list[ChunkMetadata] = field(default_factory=list)
    vectors: np.ndarray | None = None
    index: FaissIndex | None = None
    files: dict[str, EphemeralFileInfo] = field(default_factory=dict)
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))
```

Cada conversación tiene su propio `_ConversationStore`. El campo `last_used` se actualiza con cada operación y se usa para la limpieza por TTL.

### `EphemeralFileInfo`

```python
class EphemeralFileInfo(BaseModel):
    file_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime
```

Lo que la API devuelve al cliente después de subir un archivo. `file_id` es un identificador opaco de 12 caracteres que se usa para eliminar archivos específicos.

### Métodos principales

#### `add_file()` — Subir y procesar

```python
def add_file(
    self,
    conversation_id: str,
    filename: str,
    source_type: str,
    pages: list[tuple[int, str]],
) -> EphemeralFileInfo:
```

El flujo:

```
pages (texto extraído del PDF/HTML/TXT)
    │
    ▼  chunk_text()
[chunk1, chunk2, ...]
    │
    ▼  build_metadata()  ← con file_id único
[ChunkMetadata, ChunkMetadata, ...]
    │
    ▼  encode_chunks()
np.ndarray (N, 384)  ← vectores normalizados
    │
    ▼  np.vstack() con vectores existentes
vectores actualizados
    │
    ▼  rebuild_index_from_vectors()
nuevo FaissIndex
```

Si la conversación ya tiene archivos, el nuevo se **agrega** a los existentes — no los reemplaza. Los vectores se apilan con `np.vstack` y el índice FAISS se reconstruye completo. Para el tamaño típico de una colección efímera (un par de archivos por conversación), esto es trivialmente barato.

#### `remove_file()` — Eliminar un archivo específico

```python
def remove_file(self, conversation_id: str, file_id: str) -> bool:
```

Filtra metadata y vectores por `file_id` (recordemos que cada `ChunkMetadata` tiene un campo `file_id`) y reconstruye el índice con lo que queda. Si era el último archivo, elimina toda la conversación.

```python
keep = [i for i, m in enumerate(store.metadata) if m.file_id != file_id]

store.metadata = [store.metadata[i] for i in keep]
store.vectors = store.vectors[keep]
store.index = _rebuild_index(store.vectors)
```

¿Por qué reconstruir el índice en lugar de usar `faiss.IndexIDMap.remove_ids()`? Porque no todos los tipos de índice soportan `remove_ids`, y para el tamaño típico de efímero reconstruir es igual de rápido.

#### `get_collection()` — Listo para `search()`

```python
def get_collection(self, conversation_id: str) -> LoadedCollection | None:
    store = self._conversations.get(conversation_id)
    if store is None or store.index is None:
        return None

    store.last_used = datetime.now(UTC)
    return LoadedCollection(
        index=store.index,
        metadata=store.metadata,
        vectors=store.vectors,
        paths=_placeholder_paths(conversation_id),
        collection_name=f"_ephemeral/{conversation_id}",
    )
```

Este es el punto de integración con el Módulo 4. `get_collection()` devuelve un `LoadedCollection` — exactamente el mismo tipo que `ContextManager.get_loaded_collections()`. Así, la búsqueda semántica no distingue entre colecciones efímeras y persistidas: ambas entran al mismo `retrieve()`.

Los `paths` son placeholders (nunca se tocan del disco) — `retrieve()` solo usa `index`, `metadata` y `collection_name`.

#### `remove_conversation()` — Limpiar todo

```python
def remove_conversation(self, conversation_id: str) -> bool:
    existed = conversation_id in self._conversations
    self._conversations.pop(conversation_id, None)
    return existed
```

Limpia toda la colección efímera de una conversación. El cliente lo llama al cerrar o eliminar una conversación en la UI.

#### `sweep_expired()` — Limpieza por TTL

```python
def sweep_expired(self, ttl: timedelta) -> int:
    cutoff = datetime.now(UTC) - ttl
    expired = [cid for cid, s in self._conversations.items() if s.last_used < cutoff]

    for cid in expired:
        del self._conversations[cid]

    return len(expired)
```

Sin esto, las colecciones efímeras crecerían sin límite mientras viva el proceso. `sweep_expired` se ejecuta periódicamente desde el lifespan de la app (ver `src/api/app.py`) y elimina conversaciones inactivas por más de `ttl` (6 horas por defecto).

### Cómo integra con la query

Cuando el API recibe una pregunta, la lista de colecciones para `search()` se ensambla así:

```
collections = context_manager.get_loaded_collections()   # de disco
ephemeral = ephemeral_store.get_collection(conversation_id)
if ephemeral is not None:
    collections.append(ephemeral)

results, confidence = search(question, mode, collections)
```

`search()` no sabe ni le importa si una colección viene de disco o de RAM. Ambas son `LoadedCollection`.

---

## 2. AttachmentStore — Archivos para inyectar raw

**Archivo fuente:** `src/context/attachments.py`

### La diferencia con EphemeralStore

`AttachmentStore` no hace chunking, ni embeddings, ni indexación FAISS. Guarda el **contenido de texto raw** de archivos de texto (`.py`, `.js`, `.ts`, `.json`, `.yaml`, `.md`, etc.) y lo inyecta directamente en el prompt cuando el usuario envía la siguiente pregunta.

¿Cuándo usar cada uno?

| Caso de uso | Almacén |
|---|---|
| "Quiero que busques en este PDF" | `EphemeralStore` |
| "Mira este módulo y explícalo" | `AttachmentStore` |
| "¿Qué dice este documento sobre X?" | `EphemeralStore` |
| "Revisa este snippet de código" | `AttachmentStore` |

### Formatos soportados

```python
SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".yaml", ".yml", ".toml", ".csv", ".sql", ".sh",
    ".env", ".cfg", ".ini", ".xml", ".css", ".html",
}
```

Solo texto plano. No PDFs, no imágenes — para eso está `EphemeralStore`. El límite es 500 KB por archivo.

### Métodos principales

#### `add_file()` — Guardar contenido

```python
def add_file(self, conversation_id: str, filename: str, content: str) -> AttachmentInfo:
    file_id = uuid.uuid4().hex[:12]
    store = self._conversations.setdefault(conversation_id, _ConversationAttachments())

    attachment = _Attachment(filename=filename, content=content)
    store.files[file_id] = attachment

    return AttachmentInfo(
        file_id=file_id,
        filename=filename,
        size_bytes=len(content.encode()),
        uploaded_at=attachment.uploadated_at,
    )
```

Guarda el contenido como string. No hay procesamiento — solo se almacena.

#### `list_contents()` — Para el inyector de prompts

```python
def list_contents(self, conversation_id: str | None) -> list[tuple[str, str]]:
```

Devuelve una lista de `(filename, content)` — lo que `inject_attachments()` en `src/prompts/builder.py` usa para construir el bloque de adjuntos en el prompt.

#### Consumo automático

Las attachments se consumen automáticamente después de cada query. El router de chat llama `remove_conversation()` después de procesar una pregunta que incluía adjuntos, para que la lista quede vacía para el siguiente mensaje. Esto es diferente a `EphemeralStore`, donde los archivos persisten durante toda la conversación.

---

## 3. Endpoints del API

**Archivo fuente:** `src/api/routers/files.py`

### POST `/api/v1/files` — Subir archivo

```python
@router.post("", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile,
    conversation_id: str = Form(...),
    attach_to_collection: bool = Form(False),
    collection: str | None = Form(default=None),
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> FileUploadResponse:
```

Dos destinos posibles según `attach_to_collection`:

| `attach_to_collection` | ¿Qué pasa? | ¿Sobrevive reinicio? |
|---|---|---|
| `False` (default) | Se indexa en RAM, scoped a `conversation_id` | No |
| `True` | Se procesa con el mismo pipeline y se guarda en disco en `collection` | Sí |

Soporta `.pdf`, `.html`, `.txt`. Otro formato devuelve 415.

Respuesta:

```python
class FileUploadResponse(BaseModel):
    conversation_id: str
    file_id: str
    filename: str
    chunk_count: int
    attached_to_collection: str | None = None   # None si es efímero
```

### GET `/api/v1/files/{conversation_id}` — Listar archivos efímeros

```python
@router.get("/{conversation_id}", response_model=EphemeralFilesResponse)
async def list_ephemeral_files(
    conversation_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> EphemeralFilesResponse:
```

Devuelve los archivos efímeros de una conversación:

```python
class EphemeralFilesResponse(BaseModel):
    conversation_id: str
    files: list[EphemeralFileInfo]
```

### DELETE `/api/v1/files/{conversation_id}/{file_id}` — Eliminar un archivo

```python
@router.delete("/{conversation_id}/{file_id}", response_model=DeleteResponse)
async def delete_ephemeral_file(
    conversation_id: str,
    file_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> DeleteResponse:
```

Elimina un archivo específico de la colección efímera. Si no existe (ya fue eliminado, expiró, o nunca se subió), devuelve 404. Si era el último archivo, elimina toda la conversación.

### DELETE `/api/v1/files/{conversation_id}` — Eliminar toda la conversación

```python
@router.delete("/{conversation_id}", response_model=DeleteResponse)
async def delete_ephemeral_conversation(
    conversation_id: str,
    ephemeral_store: EphemeralStore = Depends(get_ephemeral_store),
) -> DeleteResponse:
```

Limpia todos los archivos efímeros de una conversación de una vez. El cliente lo llama al cerrar o eliminar una conversación en la UI.

### Schemas de respuesta

```python
class DeleteResponse(BaseModel):
    deleted: bool
```

Simple y consistente para todas las operaciones de eliminación.

---

## Arquitectura completa

```
CLIENTE                  API                      ALMACENES
──────                  ───                      ────────

POST /files             upload_file()
  file + conversation_id ──→  _extract_pages()
                              │
                    ¿attach_to_collection?
                    │                    │
                   No                   Sí
                    │                    │
                    ▼                    ▼
              ephemeral_store.    save_collection()
              add_file()          (disco, persistente)
                    │
                    ▼
              _ConversationStore
              chunks + vectores + índice
                    │
                    ▼
              get_collection()
              → LoadedCollection
                    │
                    ▼
              search() ←────  context_manager.get_loaded_collections()
              (Módulo 4)       + ephemeral_store.get_collection()

DELETE /files/{cid}/{fid}
  → remove_file()
  → rebuild index without that file's chunks

DELETE /files/{cid}
  → remove_conversation()
  → free all RAM
```

### Limpieza automática

Tanto `EphemeralStore` como `AttachmentStore` tienen `sweep_expired()`, ejecutado periódicamente desde el lifespan de la app:

```python
# en src/api/app.py
async def _cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        ephemeral_store.sweep_expired(EPHEMERAL_TTL)   # 6 horas
        attachment_store.sweep_expired(ATTACHMENT_TTL)
```

Esto garantiza que las colecciones efímeras no crezcan sin límite en un proceso de larga duración.

### Concurrencia

Diseñado para un solo worker de uvicorn (el caso típico de este proyecto: local, un solo usuario). Con `--workers > 1` cada worker tendría su propio almacén en memoria, y una conversación podría "perder" sus archivos si el load balancer la routea a otro worker en el siguiente request.
