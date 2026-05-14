# python -m src.ingest.web_ingest react https://sitio.com

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

        logger.info(f"Contenido extraído correctamente.")

        return text

    except requests.RequestException as e:

        logger.error(f"Error HTTP en {url}: {e}")

        return None

    except Exception as e:

        logger.error(f"Error extrayendo " f"contenido de {url}: {e}")

        return None


def main():

    if len(sys.argv) < 3:

        logger.error("Argumentos insuficientes.")

        print("Uso: python -m " "src.ingest.web_ingest " "<collection> <URL>")

        return

    collection = sys.argv[1]

    url = sys.argv[2]

    logger.info(f"Iniciando web ingest | " f"collection={collection}")

    collection_data = load_collection(collection)

    metadata = collection_data["metadata"]

    existing_sources = {m["source"] for m in metadata}

    if url in existing_sources:

        logger.warning(f"URL ya indexada: {url}")

        print("URL ya indexada.")

        return

    logger.info(f"Procesando URL: {url}")

    text = extract_main_content(url)

    if not text:

        logger.error("No se pudo extraer contenido.")

        print("No se pudo extraer contenido.")

        return

    chunks = chunk_text(text)

    logger.info(f"Chunks generados: " f"{len(chunks)}")

    new_chunks = []

    new_metadata = []

    for i, chunk in enumerate(chunks):

        new_chunks.append(chunk)

        new_metadata.append(
            build_metadata(
                source=url,
                page=1,
                chunk=chunk,
                chunk_index=i,
                collection=collection,
            )
        )

    logger.info("Generando embeddings...")

    new_embeddings = encode_chunks(new_chunks)

    save_collection(
        collection_data,
        new_embeddings,
        new_metadata,
    )

    logger.info(f"Web ingest finalizado | " f"chunks={len(chunks)}")

    print(f"Se indexaron " f"{len(chunks)} chunks " f"en '{collection}'.")


if __name__ == "__main__":
    main()
