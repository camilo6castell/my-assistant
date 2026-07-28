# python -m src.ingest.ingest <category> <collection>

import sys
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config.settings import settings
from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
from src.utils.logger import logger


def read_pdf(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Reading PDF: {path.name}")
    pages: list[tuple[int, str]] = []

    try:
        reader: PdfReader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"Could not open PDF: {path.name} | {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text: str = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text))
            else:
                logger.warning(f"Empty page | {path.name} | page={i + 1}")
        except Exception as e:
            logger.warning(f"Could not read page | {path.name} | page={i + 1} | error={e}")
            continue

    return pages


def read_html(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Reading HTML: {path.name}")
    with open(path, encoding="utf-8") as f:
        soup: BeautifulSoup = BeautifulSoup(f, "html.parser")
    return [(1, soup.get_text(separator="\n"))]


def read_txt(path: Path) -> list[tuple[int, str]]:
    logger.info(f"Reading TXT: {path.name}")
    with open(path, encoding="utf-8") as f:
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
    logger.warning(f"Unsupported format: {path.name}")
    return []


def main() -> None:

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    collection: str = f"{category}/{collection_name}"

    logger.info(f"Starting ingest: {collection}")

    collection_data: RawCollection = load_collection(collection)

    existing_sources: set[str] = {m.source for m in collection_data["metadata"]}

    new_chunks: list[str] = []
    new_metadata: list[ChunkMetadata] = []

    files: list[Path] = list(settings.data_path.iterdir())

    if not files:
        logger.warning("No files in data/")
        print("No files in data/")
        return

    for file in files:
        if file.name in existing_sources:
            logger.info(f"Skipping already indexed: {file.name}")
            continue

        pages: list[tuple[int, str]] = read_file(file)

        if not pages:
            logger.warning(f"Could not extract content: {file.name}")
            continue

        logger.info(f"Processing: {file.name}")

        for page_number, text in pages:
            chunks: list[str] = chunk_text(text)
            logger.info(f"Chunks generated: {len(chunks)} | page={page_number}")

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
        logger.warning("No new content.")
        print("No new content.")
        return

    embeddings: np.ndarray = encode_chunks(new_chunks)

    save_collection(
        collection_data=collection_data,
        new_embeddings=embeddings,
        new_metadata=new_metadata,
    )

    logger.info(f"Ingest finished | chunks={len(new_chunks)}")
    print(f"\nAdded {len(new_chunks)} chunks to '{collection}'.\n")


if __name__ == "__main__":
    main()
