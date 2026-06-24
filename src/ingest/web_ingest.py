# python -m src.ingest.web_ingest <categoria> <coleccion> <url>

import sys

import numpy as np
import requests
from bs4 import BeautifulSoup
from readability import Document

from src.ingest.core import (
    ChunkMetadata,
    RawCollection,
    build_metadata,
    chunk_text,
    encode_chunks,
    load_collection,
    save_collection,
)
from src.utils.logger import logger


def extract_main_content(url: str) -> str | None:
    try:
        logger.info(f"Descargando URL: {url}")
        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()
        doc: Document = Document(response.text)
        soup: BeautifulSoup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        logger.error(f"Error extrayendo contenido de {url}: {e}")
        return None


def main() -> None:
    if len(sys.argv) != 4:
        print("Uso: python -m src.ingest.web_ingest <categoria> <coleccion> <url>")
        return

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    url: str = sys.argv[3]
    collection: str = f"{category}/{collection_name}"

    logger.info(f"Iniciando web ingest: {collection}")

    collection_data: RawCollection = load_collection(collection)

    # Acceso por atributo — ChunkMetadata es BaseModel
    existing_sources: set[str] = {m.source for m in collection_data["metadata"]}

    if url in existing_sources:
        logger.warning("URL ya indexada.")
        print("URL ya indexada.")
        return

    text: str | None = extract_main_content(url)

    if not text:
        print("No se pudo extraer contenido.")
        return

    chunks: list[str] = chunk_text(text)
    logger.info(f"Chunks generados: {len(chunks)}")

    new_metadata: list[ChunkMetadata] = [
        build_metadata(
            source=url,
            source_type="url",
            page=1,
            chunk=chunk,
            chunk_index=i,
            collection=collection,
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings: np.ndarray = encode_chunks(chunks)

    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )

    print(f"\nSe indexaron {len(chunks)} chunks en '{collection}'.\n")


if __name__ == "__main__":
    main()
