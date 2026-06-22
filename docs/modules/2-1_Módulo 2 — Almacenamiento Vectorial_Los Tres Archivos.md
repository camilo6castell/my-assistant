# Módulo 2 — Almacenamiento Vectorial: Los Tres Archivos

Cuando termina el ingest de una colección, el sistema escribe exactamente tres archivos en disco:

```
vector_stores/
└── sociologia/
    └── debord/
        ├── index.faiss    ← estructura de búsqueda
        ├── metadata.pkl   ← información de cada chunk
        └── vectors.npy    ← los vectores crudos
```

La primera pregunta natural es: **¿por qué tres archivos y no uno?** Cada uno tiene un rol distinto e irremplazable. Entender por qué existen los tres juntos es entender la arquitectura de almacenamiento completa.

---

## `metadata.pkl` — El índice humano

```python
# Lo que contiene: una lista de dicts, uno por chunk
[
    {
        "source": "La-sociedad-del-espectaculo.pdf",
        "source_type": "file",
        "page": 12,
        "text": "El espectáculo no es un conjunto de imágenes...",
        "chunk_index": 0,
        "collection": "sociologia/debord"
    },
    {
        "source": "La-sociedad-del-espectaculo.pdf",
        "source_type": "file",
        "page": 12,
        "text": "...sino una relación social entre personas mediatizada",
        "chunk_index": 1,
        "collection": "sociologia/debord"
    },
    # ... un dict por cada chunk indexado
]
```

Es una lista Python serializada con `pickle`. La posición en la lista **es** el identificador del chunk — no hay IDs explícitos. El chunk en `metadata[47]` corresponde al vector en la fila 47 de la matriz de vectores.

Puedes pensar en esto como la tabla de una base de datos donde la primary key es el índice de posición.

---

## `vectors.npy` — La memoria cruda

```python
# Shape: (N, 384) donde N = número total de chunks
# Ejemplo con 3 chunks:
array([
    [ 0.21, -0.54,  0.88,  0.03, ... ],  # fila 0 → vector de metadata[0]
    [ 0.19, -0.51,  0.90,  0.01, ... ],  # fila 1 → vector de metadata[1]
    [-0.72,  0.33, -0.11,  0.95, ... ],  # fila 2 → vector de metadata[2]
], dtype=float32)
```

Es la matriz de embeddings guardada en formato numpy binario. Numpy tiene su propio formato `.npy` que es extremadamente eficiente — carga la matriz entera en memoria con un solo `mmap` sin parsear nada.

**¿Por qué existe si FAISS ya guarda los vectores internamente?**

Porque FAISS en su formato `.faiss` no garantiza que puedas extraer los vectores originales de vuelta de forma fiable. `vectors.npy` es la fuente de verdad para operaciones como `rebuild_index()` y `vacuum_collection()` en `delete.py` — si el índice FAISS se corrompe, se puede reconstruir desde cero con este archivo.

---

## `index.faiss` — La estructura de búsqueda

Este es el más importante y el más interesante de entender.

**El problema que resuelve:** tienes 50.000 vectores de 384 dimensiones. Llega una pregunta convertida en vector. Necesitas encontrar los 5 vectores más similares. ¿Cómo?

La respuesta naive es comparar tu vector de pregunta contra los 50.000 uno por uno. Eso es búsqueda lineal O(N). Con N grande, es demasiado lento.

**La analogía con estructuras que ya conoces:**

Un índice de base de datos relacional (como un B-Tree en PostgreSQL) resuelve un problema similar para búsquedas exactas — en lugar de recorrer toda la tabla fila por fila, organiza los datos de forma que puedas llegar al resultado en O(log N).

FAISS hace algo análogo pero para búsqueda por similitud en espacios de alta dimensión.

**El tipo específico que usa este proyecto es `IndexFlatIP`:**

```python
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
```

`Flat` significa que guarda todos los vectores y hace comparación exacta — sin aproximaciones. Es el índice más simple de FAISS y el correcto para colecciones medianas (hasta ~100k vectores).

`IP` significa Inner Product — producto interno. Cuando los vectores están normalizados (que es exactamente lo que hace `normalize_embeddings=True`), el producto interno entre dos vectores es idéntico matemáticamente a la similitud coseno.

**La similitud coseno explicada sin fórmulas:**

Imagina cada vector como una flecha en el espacio. Dos flechas que apuntan en la misma dirección tienen similitud 1.0 (idénticas en significado). Dos flechas perpendiculares tienen similitud 0 (sin relación). Dos flechas opuestas tienen similitud -1.0 (significados opuestos).

```
"perro"  →  ↗  similitud alta con "can" ↗
"gato"   →  ↘  similitud baja con "perro" ↗
"bolsa"  →  ←  similitud casi cero con animales
```

FAISS calcula ese ángulo entre tu pregunta y todos los chunks, y te devuelve los más cercanos.

---

## Cómo los tres archivos trabajan juntos

```python
# En load_collection():

# 1. Carga el índice de búsqueda
index = faiss.read_index(str(paths["index_file"]))

# 2. Carga la metadata (lista de dicts)
with open(paths["metadata_file"], "rb") as f:
    metadata = pickle.load(f)

# 3. Carga los vectores crudos (para operaciones de mantenimiento)
vectors = np.load(paths["vectors_file"])
```

En tiempo de consulta, el flujo es:

```
Tu pregunta → vector (384 nums)
                    │
                    ▼
            index.search(vector, k=5)
                    │
                    ▼
            [47, 203, 891, 12, 556]  ← índices de los 5 más similares
                    │
                    ▼
            metadata[47], metadata[203], ...  ← texto + fuente + página
```

FAISS devuelve índices enteros. Esos índices son exactamente las posiciones en la lista `metadata`. La búsqueda vectorial de alta velocidad y la recuperación de información legible son dos operaciones separadas conectadas únicamente por el número de posición.

---

## La relación entre los tres archivos visualizada

```
metadata.pkl          vectors.npy           index.faiss
────────────          ───────────           ───────────
[0] {text, page...}   fila 0: [...]         Estructura
[1] {text, page...}   fila 1: [...]         optimizada
[2] {text, page...}   fila 2: [...]         para buscar
[3] {text, page...}   fila 3: [...]         por similitud
...                   ...                   
                                            search(q, 5)
                       ◄──────────────────  devuelve [2, 0, ...]
metadata[2] ◄─────────┘
metadata[0] ◄─────────┘
```

El contrato que mantiene todo coherente: **la posición es la clave**. Es la razón por la que `vacuum_collection()` existe — si borras filas del medio, la sincronización se rompe y hay que reindexar todo.

---

¿Avanzamos al Módulo 3 — cómo el `ContextManager` organiza múltiples colecciones en memoria y por qué la arquitectura de namespaces está diseñada así?
