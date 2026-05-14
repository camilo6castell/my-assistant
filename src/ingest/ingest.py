# python -m src.ingest.ingest react

import sys

from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup

from src.utils.logger import logger

from src.utils.env import DATA_PATH

from src.ingest.core import (
    chunk_text,
    load_collection,
    encode_chunks,
    save_collection,
    build_metadata,
)


def read_pdf(path: Path):

    reader = PdfReader(str(path))

    pages = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append((i + 1, text))

    return pages


def read_html(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        soup = BeautifulSoup(
            f,
            "html.parser",
        )

    return [
        (
            1,
            soup.get_text(separator="\n"),
        )
    ]


def read_txt(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return [(1, f.read())]


def main():

    if len(sys.argv) < 2:

        logger.error("Collection no especificada.")

        print("Uso: python -m " "src.ingest.ingest " "<collection>")

        return

    collection = sys.argv[1]

    logger.info(f"Iniciando ingest " f"para '{collection}'")

    collection_data = load_collection(collection)

    metadata = collection_data["metadata"]

    existing_sources = {m["source"] for m in metadata}

    new_chunks = []

    new_metadata = []

    for file in DATA_PATH.iterdir():

        if file.name in existing_sources:

            logger.info(f"Omitiendo ya indexado: " f"{file.name}")

            continue

        if file.suffix == ".pdf":

            pages = read_pdf(file)

        elif file.suffix == ".html":

            pages = read_html(file)

        elif file.suffix == ".txt":

            pages = read_txt(file)

        else:

            logger.warning(f"Formato no soportado: " f"{file.name}")

            continue

        logger.info(f"Procesando: {file.name}")

        for page_number, text in pages:

            chunks = chunk_text(text)

            logger.info(f"Chunks generados: " f"{len(chunks)}")

            for i, chunk in enumerate(chunks):

                new_chunks.append(chunk)

                new_metadata.append(
                    build_metadata(
                        source=file.name,
                        page=page_number,
                        chunk=chunk,
                        chunk_index=i,
                        collection=collection,
                    )
                )

    if not new_chunks:

        logger.warning("No hay documentos nuevos.")

        print("No hay documentos nuevos.")

        return

    new_embeddings = encode_chunks(new_chunks)

    save_collection(
        collection_data,
        new_embeddings,
        new_metadata,
    )

    logger.info(f"Ingest finalizado " f"({len(new_chunks)} chunks)")

    print(f"Se añadieron " f"{len(new_chunks)} nuevos chunks " f"a '{collection}'.")


if __name__ == "__main__":
    main()
