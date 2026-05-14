from pathlib import Path
import pickle
import faiss
import numpy as np

from FlagEmbedding import FlagModel

from src.utils.logger import logger

from src.utils.env import (
    BASE_VECTOR_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

model = FlagModel(
    EMBED_MODEL,
    use_fp16=False,
    trust_remote_code=False,
)


def chunk_text(text: str) -> list[str]:

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunks.append(text[start:end])

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def get_collection_paths(
    collection: str,
) -> dict[str, Path]:

    vector_path = BASE_VECTOR_PATH / collection

    vector_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "vector_path": vector_path,
        "index_file": (vector_path / "index.faiss"),
        "metadata_file": (vector_path / "metadata.pkl"),
        "vectors_file": (vector_path / "vectors.npy"),
    }


def load_collection(collection: str):

    paths = get_collection_paths(collection)

    index_file = paths["index_file"]

    metadata_file = paths["metadata_file"]

    vectors_file = paths["vectors_file"]

    if index_file.exists():

        logger.info(f"Cargando colección: " f"{collection}")

        index = faiss.read_index(str(index_file))

        with open(
            metadata_file,
            "rb",
        ) as f:
            metadata = pickle.load(f)

        if vectors_file.exists():

            vectors = np.load(vectors_file)

            logger.info("vectors.npy cargado.")

        else:

            logger.warning("vectors.npy no encontrado.")

            vectors = None

    else:

        logger.info(f"Creando nueva colección: " f"{collection}")

        index = None
        metadata = []
        vectors = None

    return {
        "index": index,
        "metadata": metadata,
        "vectors": vectors,
        "paths": paths,
    }


def encode_chunks(
    chunks: list[str],
) -> np.ndarray:

    logger.info(f"Generando embeddings " f"({len(chunks)} chunks)")

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings


def create_faiss_index(
    dimension: int,
):

    logger.info(f"Creando índice FAISS " f"(dim={dimension})")

    return faiss.IndexFlatIP(dimension)


def save_collection(
    collection_data,
    new_embeddings,
    new_metadata,
):

    logger.info("Guardando colección...")

    index = collection_data["index"]

    metadata = collection_data["metadata"]

    existing_vectors = collection_data["vectors"]

    paths = collection_data["paths"]

    if existing_vectors is not None:

        all_vectors = np.vstack(
            [
                existing_vectors,
                new_embeddings,
            ]
        )

    else:

        all_vectors = new_embeddings

    if index is None:

        dimension = new_embeddings.shape[1]

        index = create_faiss_index(dimension)

    index.add(new_embeddings)

    metadata.extend(new_metadata)

    faiss.write_index(
        index,
        str(paths["index_file"]),
    )

    np.save(
        paths["vectors_file"],
        all_vectors,
    )

    with open(
        paths["metadata_file"],
        "wb",
    ) as f:
        pickle.dump(metadata, f)

    logger.info("Colección guardada correctamente.")


def build_metadata(
    source: str,
    page: int,
    chunk: str,
    chunk_index: int,
    collection: str,
):

    return {
        "source": source,
        "page": page,
        "text": chunk,
        "chunk_index": chunk_index,
        "collection": collection,
    }
