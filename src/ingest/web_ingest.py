# python -m src.ingest.web_ingest <category> <collection> <url>

import sys

import numpy as np

from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.ingest.http import extract_main_content
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
from src.utils.logger import logger


def main() -> None:
    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    url: str = sys.argv[3]
    collection: str = f"{category}/{collection_name}"

    logger.info(f"Starting web ingest: {collection}")

    collection_data: RawCollection = load_collection(collection)

    existing_sources: set[str] = {m.source for m in collection_data["metadata"]}

    if url in existing_sources:
        logger.warning("URL already indexed.")
        print("URL already indexed.")
        return

    text: str | None = extract_main_content(url)

    if not text:
        print("Could not extract content.")
        return

    chunks: list[str] = chunk_text(text)
    logger.info(f"Chunks generated: {len(chunks)}")

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

    print(f"\nIndexed {len(chunks)} chunks in '{collection}'.\n")


if __name__ == "__main__":
    main()
