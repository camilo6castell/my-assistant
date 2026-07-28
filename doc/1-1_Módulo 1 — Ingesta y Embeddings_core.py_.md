# Módulo 1 — Ingesta y Embeddings

Antes de ver una sola línea de código, necesitas tener el modelo mental correcto. Si este concepto queda claro, todo lo demás se entiende solo.

---

## El problema fundamental

Imagina que tienes dos frases:

> "El perro persiguió al gato"
> "El can fue tras el felino"

Para un `String.contains()` o un `grep`, estas frases no tienen nada en común. Para un humano, son casi idénticas en significado.

**Los embeddings resuelven exactamente esto.**

Un modelo de embeddings es una función que toma texto y devuelve un array de números flotantes — un vector — donde **textos con significados similares producen vectores matemáticamente cercanos**, sin importar las palabras exactas usadas.

```
"El perro persiguió al gato"  →  [0.21, -0.54, 0.88, 0.03, ...]  ← 384 números
"El can fue tras el felino"   →  [0.19, -0.51, 0.90, 0.01, ...]  ← 384 números
"La bolsa subió en Wall St."  →  [-0.72, 0.33, -0.11, 0.95, ...]  ← 384 números
```

Las dos primeras filas son numéricamente cercanas. La tercera está lejos. Eso es todo el RAG: **distancias entre vectores**.

Piénsalo como coordenadas GPS del significado. Dos lugares con el mismo significado semántico viven en el mismo barrio del espacio vectorial.

---

## Paso 1 de 2 — ¿Por qué dividimos el texto en chunks?

Antes de generar vectores, el sistema divide cada documento en fragmentos. La función responsable es `chunk_text()` en `src/ingest/core.py`:

```python
def chunk_text(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE          # CHUNK_SIZE = 500 caracteres
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP   # CHUNK_OVERLAP = 100 caracteres
    
    return chunks
```

**¿Por qué no vectorizar el documento entero?**

Dos razones concretas:

**Razón 1 — Precisión de recuperación.** Un vector representa *el significado promedio* de todo el texto que codifica. Si vectorizas un libro entero, ese vector es un promedio tan diluido que no es útil para ninguna pregunta específica. Es como intentar buscar una dirección con las coordenadas GPS del centro geográfico de Colombia.

**Razón 2 — Límite del modelo.** Los modelos de embeddings tienen un límite de tokens de entrada (el tuyo, `BAAI/bge-small-en-v1.5`, acepta ~512 tokens ≈ 400 palabras). Un PDF de 300 páginas no cabe.

**¿Por qué el overlap?**

Visualiza el texto como una cinta:

```
│←────────── chunk 1 (500 chars) ──────────→│
                              │←── overlap (100) ──→│←────────── chunk 2 ──────────→│
```

Sin overlap, una idea que empieza al final del chunk 1 y termina al inicio del chunk 2 quedaría partida en dos vectores incompletos — ninguno capturaría la idea entera. El overlap garantiza que cada idea tenga al menos un chunk que la contenga completa.

---

## Paso 2 de 2 — Cómo se generan los vectores

```python
def encode_chunks(chunks: list[str]) -> np.ndarray:
    logger.info(f"Generando embeddings para {len(chunks)} chunks")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return _to_f32(np.array(embeddings))
```

`model` es una instancia de `SentenceTransformer("BAAI/bge-small-en-v1.5")` que vive en memoria desde que arranca el proceso. Recibe toda la lista de chunks de una vez y devuelve una matriz:

```
chunks = ["texto A", "texto B", "texto C"]   # lista de 3 strings

embeddings = model.encode(chunks)

# Resultado: matriz de shape (3, 384)
# Cada fila es el vector de un chunk
# [
#   [0.21, -0.54, 0.88, ...],   ← vector de "texto A"
#   [0.19, -0.51, 0.90, ...],   ← vector de "texto B"
#   [-0.72, 0.33, -0.11, ...],  ← vector de "texto C"
# ]
```

**`normalize_embeddings=True` — ¿qué hace exactamente?**

Normalizar un vector significa escalar sus valores para que su longitud (norma) sea exactamente 1.0. Todos los vectores quedan sobre la superficie de una esfera unitaria.

¿Por qué importa? Porque cuando todos los vectores tienen la misma longitud, **medir la distancia entre dos puntos es equivalente a medir el ángulo entre ellos** — que es exactamente la similitud coseno. Y la similitud coseno mide similitud de dirección, no de magnitud, que es lo que queremos: que "perro" y "can" apunten en la misma dirección semántica.

**`_to_f32()`** convierte el array a `float32` (32 bits por número en lugar de 64). FAISS trabaja internamente con `float32`. Sin esta conversión, FAISS falla en runtime aunque el array parezca idéntico visualmente.

---

Estos dos mecanismos — chunking con overlap + embeddings normalizados — son la base de todo el sistema. El resto del módulo 1 es la tubería que orquesta estas funciones para distintas fuentes (PDF, TXT, HTML, URL).

¿Pasamos a ver cómo `ingest.py` orquesta este pipeline para archivos locales, o quieres profundizar en alguno de estos dos conceptos primero?
