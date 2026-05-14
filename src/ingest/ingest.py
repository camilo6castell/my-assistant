# python -m src.ingest.ingest sociologia la-sociedad-del-espectaculo

import sys

from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup

from src.utils.logger import logger

from src.config.settings import DATA_PATH

from src.ingest.core import (
    chunk_text,
    load_collection,
    encode_chunks,
    save_collection,
    build_metadata,
)


def read_pdf(path: Path):

    logger.info(f"Leyendo PDF: {path.name}")

    pages = []

    try:

        reader = PdfReader(str(path))

    except Exception as e:

        logger.error(f"No se pudo abrir PDF: {path.name} | {e}")

        return pages

    for i, page in enumerate(reader.pages):

        try:

            text = page.extract_text()

            if text and text.strip():

                pages.append(
                    (
                        i + 1,
                        text,
                    )
                )

            else:

                logger.warning(f"Página vacía | " f"{path.name} | " f"page={i + 1}")

        except Exception as e:

            logger.warning(
                f"No se pudo leer página | "
                f"{path.name} | "
                f"page={i + 1} | "
                f"error={e}"
            )

            continue

    return pages


def read_html(path: Path):

    logger.info(f"Leyendo HTML: {path.name}")

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        soup = BeautifulSoup(
            f,
            "html.parser",
        )

    text = soup.get_text(separator="\n")

    return [(1, text)]


def read_txt(path: Path):

    logger.info(f"Leyendo TXT: {path.name}")

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        text = f.read()

    return [(1, text)]


def read_file(path: Path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".html":
        return read_html(path)

    if suffix == ".txt":
        return read_txt(path)

    logger.warning(f"Formato no soportado: {path.name}")

    return []


def main():

    if len(sys.argv) != 3:

        print("Uso: python -m src.ingest.ingest " "<categoria> <coleccion>")

        return

    category = sys.argv[1]
    collection_name = sys.argv[2]

    collection = f"{category}/{collection_name}"

    logger.info(f"Iniciando ingest: {collection}")

    collection_data = load_collection(collection)

    metadata = collection_data["metadata"]

    existing_sources = {m["source"] for m in metadata}

    new_chunks = []
    new_metadata = []

    files = list(DATA_PATH.iterdir())

    if not files:

        logger.warning("No hay archivos en data/")

        print("No hay archivos en data/")

        return

    for file in files:

        if file.name in existing_sources:

            logger.info(f"Omitiendo ya indexado: {file.name}")

            continue

        pages = read_file(file)

        if not pages:

            logger.warning(f"No se pudo extraer contenido: {file.name}")

            continue

        logger.info(f"Procesando: {file.name}")

        for page_number, text in pages:

            chunks = chunk_text(text)

            logger.info(f"Chunks generados: {len(chunks)} " f"| page={page_number}")

            for i, chunk in enumerate(chunks):

                new_chunks.append(chunk)

                new_metadata.append(
                    build_metadata(
                        source=file.name,
                        source_type="file",
                        page=page_number,
                        chunk=chunk,
                        chunk_index=i,
                        collection=collection,
                    )
                )

    if not new_chunks:

        logger.warning("No hay contenido nuevo.")

        print("No hay contenido nuevo.")

        return

    embeddings = encode_chunks(new_chunks)

    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )

    logger.info(f"Ingest finalizado " f"| chunks={len(new_chunks)}")

    print()

    print(f"Se añadieron " f"{len(new_chunks)} chunks " f"a '{collection}'.")

    print()


if __name__ == "__main__":
    main()
