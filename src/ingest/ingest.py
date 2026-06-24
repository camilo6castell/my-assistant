# python -m src.ingest.ingest <categoria> <coleccion>

import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from bs4 import BeautifulSoup

from src.config.settings import settings
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


def read_pdf(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Leyendo PDF: {path.name}")
    pages: list[tuple[int, str]] = []

    try:
        reader: PdfReader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"No se pudo abrir PDF: {path.name} | {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text: str = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text))
            else:
                logger.warning(f"Pagina vacia | {path.name} | page={i + 1}")
        except Exception as e:
            logger.warning(
                f"No se pudo leer pagina | {path.name} | page={i + 1} | error={e}"
            )
            continue

    return pages


def read_html(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Leyendo HTML: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        soup: BeautifulSoup = BeautifulSoup(f, "html.parser")
    return [(1, soup.get_text(separator="\n"))]


def read_txt(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Leyendo TXT: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        text: str = f.read()
    return [(1, text)]


def read_file(path: Path) -> list[tuple[int, str]]:
    suffix: str = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".html":
        return read_html(path)
    if suffix == ".txt":
        return read_txt(path)
    logger.warning(f"Formato no soportado: {path.name}")
    return []


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python -m src.ingest.ingest <categoria> <coleccion>")
        return

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    collection: str = f"{category}/{collection_name}"

    logger.info(f"Iniciando ingest: {collection}")

    collection_data: RawCollection = load_collection(collection)

    # Acceso por atributo — ChunkMetadata es BaseModel
    existing_sources: set[str] = {m.source for m in collection_data["metadata"]}

    new_chunks: list[str] = []
    new_metadata: list[ChunkMetadata] = []

    files: list[Path] = list(settings.data_path.iterdir())

    if not files:
        logger.warning("No hay archivos en data/")
        print("No hay archivos en data/")
        return

    for file in files:
        if file.name in existing_sources:
            logger.info(f"Omitiendo ya indexado: {file.name}")
            continue

        pages: list[tuple[int, str]] = read_file(file)

        if not pages:
            logger.warning(f"No se pudo extraer contenido: {file.name}")
            continue

        logger.info(f"Procesando: {file.name}")

        for page_number, text in pages:
            chunks: list[str] = chunk_text(text)
            logger.info(f"Chunks generados: {len(chunks)} | page={page_number}")

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

    embeddings: np.ndarray = encode_chunks(new_chunks)

    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )

    logger.info(f"Ingest finalizado | chunks={len(new_chunks)}")
    print(f"\nSe anadieron {len(new_chunks)} chunks a '{collection}'.\n")


if __name__ == "__main__":
    main()
