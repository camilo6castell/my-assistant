# python -m src.ingest.web_ingest programacion ddd-destilado https://example.com

import sys
from typing import Any

import requests

from bs4 import BeautifulSoup
from readability import Document

from src.utils.logger import logger

from src.ingest.core import (
    chunk_text,
    load_collection,
    encode_chunks,
    save_collection,
    build_metadata,
)


def extract_main_content(
    url: str,
) -> str | None:

    try:

        logger.info(f"Descargando URL: {url}")

        response: requests.Response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        doc: Document = Document(response.text)

        soup: BeautifulSoup = BeautifulSoup(
            doc.summary(),
            "html.parser",
        )

        text: str = soup.get_text(separator="\n")

        return text

    except Exception as e:

        logger.error(f"Error extrayendo contenido " f"de {url}: {e}")

        return None


def main() -> None:

    if len(sys.argv) != 4:

        print(
            "Uso: python -m " "src.ingest.web_ingest " "<categoria> <coleccion> <url>"
        )

        return

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    url: str = sys.argv[3]

    collection: str = f"{category}/{collection_name}"

    logger.info(f"Iniciando web ingest: {collection}")

    collection_data: dict[str, Any] = load_collection(collection)

    metadata: list[dict[str, Any]] = collection_data["metadata"]

    existing_sources: set[str] = {m["source"] for m in metadata}

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

    new_metadata: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):

        new_metadata.append(
            build_metadata(
                source=url,
                source_type="url",
                page=1,
                chunk=chunk,
                chunk_index=i,
                collection=collection,
            )
        )

    embeddings: Any = encode_chunks(chunks)

    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )

    print()

    print(f"Se indexaron " f"{len(chunks)} chunks " f"en '{collection}'.")

    print()


if __name__ == "__main__":
    main()
