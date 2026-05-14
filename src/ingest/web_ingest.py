# python -m src.ingest.web_ingest programacion ddd-destilado https://example.com

import sys

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
):

    try:

        logger.info(f"Descargando URL: {url}")

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        doc = Document(response.text)

        soup = BeautifulSoup(
            doc.summary(),
            "html.parser",
        )

        text = soup.get_text(separator="\n")

        return text

    except Exception as e:

        logger.error(f"Error extrayendo contenido " f"de {url}: {e}")

        return None


def main():

    if len(sys.argv) != 4:

        print(
            "Uso: python -m " "src.ingest.web_ingest " "<categoria> <coleccion> <url>"
        )

        return

    category = sys.argv[1]
    collection_name = sys.argv[2]
    url = sys.argv[3]

    collection = f"{category}/{collection_name}"

    logger.info(f"Iniciando web ingest: {collection}")

    collection_data = load_collection(collection)

    metadata = collection_data["metadata"]

    existing_sources = {m["source"] for m in metadata}

    if url in existing_sources:

        logger.warning("URL ya indexada.")

        print("URL ya indexada.")

        return

    text = extract_main_content(url)

    if not text:

        print("No se pudo extraer contenido.")

        return

    chunks = chunk_text(text)

    logger.info(f"Chunks generados: {len(chunks)}")

    new_metadata = []

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

    embeddings = encode_chunks(chunks)

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
